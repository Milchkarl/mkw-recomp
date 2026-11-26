#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "nvn_api.h"
#include "vk_mem_internal.h"

// ---------------------------------------------------------------- the CPU's writes as at each submit (MKW_SHADOW_SUBMIT)
// On the console the GPU reads the memory a submission uses right after nvnQueueSubmitCommands; the asynchronous
// replay records ~3 submits behind the game (lf6, 06/11/2026), and its shadow sync copied guest memory as it was then:
// the game had already rewritten the next frame's indirect arguments and per-instance data, and a tree drew from
// another tree's (one dark frame; lf9: 37 such frames in 1,024 without waiting, 1 in 776 when every submit waited for
// its replay). So at each submit (vk_mem_shadow_capture, game thread) the written pages of the shadow pools the replay
// uses are queried and copied into a capture tagged with the submit number; the sync of that submission applies the
// captures up to it (cap_apply) and no longer reads guest memory for those pools.
// - Captured pools: those a sync touched, from the next submit on (cap_from); one the replay has not used for
//   CAP_IDLE submits leaves the list, and a sync reads it from guest memory again until it is back.
// - Between activation and the first capture covering it, a pool keeps what its activating sync copied (older bytes,
//   never newer ones that a capture would then take back).
// - Read-trapped pages hold the shadow's bytes already: left out. Pages of untracked memory (no write watch): the pool
//   leaves the list and is read as before.
// MKW_SHADOW_SUBMIT (a measurement, default 0; "ab"): 1 captures. fe2 (06/11/2026): no fewer dark trees (55 in 770
// frames with it, 46 in 1,030 without) and 2.3 ms a submit on the game thread (685 pools, 3.5 MB each). Switching in the A/B blocks drops
// every pool at the edge: a block's first syncs may see the bytes older captures hold.
#define CAP_IDLE 64

typedef struct { PoolRec *pool; uint64_t addr, n; uint8_t *data; } CapRun;
typedef struct { uint64_t sub; CapRun *runs; int nruns; } CapPack;

// The list and the captures have a lock of their own: under shadow_lock the game thread waited for every shadow sync
// of the replay, 0.19 ms per submit with nothing to capture (fe1, 06/11/2026).
uint64_t vk_replay_submit;
static int mode = -1, active;                  // active: the last submit captured
static PlatMutex cap_lock = PLAT_MUTEX_INIT;   // innermost: after shadow_lock and gen_lock
static PoolRec **list;                         // cap_lock: captured pools
static int nlist, clist;
static volatile int32_t nlist_v;               // nlist, read without the lock
static CapPack *packs;                         // cap_lock: captures not applied yet, in submit order
static int npacks, cpacks;
static PoolRec **snap;                         // capture: the list, copied (the front end's submit lock)
static uint8_t *tracked;
static int csnap;
static uint64_t c_subs, c_pools, c_bytes, c_ns, c_dropped;   // cap_lock
static uint64_t c_query_ns, c_alloc_ns, c_copy_ns, c_runs;   // cap_lock (added after the capture): where c_ns goes
static uint64_t t_alloc_ns, t_copy_ns;                       // the capturing thread's, of the submit being captured
static uint64_t c_applied;                                  // shadow_lock

static void log_locked(void);

static void list_drop(int i) {   // cap_lock held
    list[i]->cap_on = 0;
    list[i] = list[--nlist];
    __atomic_store_n(&nlist_v, nlist, __ATOMIC_RELAXED);
}

void cap_forget(PoolRec *r) {
    plat_mutex_lock(&cap_lock);
    for (int i = 0; i < nlist; i++)
        if (list[i] == r) { list_drop(i); break; }
    for (int i = 0; i < npacks; i++)
        for (int k = 0; k < packs[i].nruns; k++)
            if (packs[i].runs[k].pool == r) packs[i].runs[k].pool = NULL;
    plat_mutex_unlock(&cap_lock);
}

// The captures leave the list under cap_lock and are applied after it: shadow_lock, held by the caller, keeps their
// pools alive (shadow_forget takes it).
void cap_apply(uint64_t sub) {
    if (!__atomic_load_n(&npacks, __ATOMIC_RELAXED)) return;
    plat_mutex_lock(&cap_lock);
    int done = 0;
    while (done < npacks && packs[done].sub <= sub) done++;
    CapPack *take = done ? malloc(sizeof *take * (size_t)done) : NULL;
    if (done && !take) rt_fatal("vk: out of memory (shadow capture)");
    if (done) memcpy(take, packs, sizeof *take * (size_t)done);
    if (done) memmove(packs, packs + done, sizeof *packs * (size_t)(npacks - done)), npacks -= done;
    plat_mutex_unlock(&cap_lock);
    for (int i = 0; i < done; i++) {
        for (int k = 0; k < take[i].nruns; k++) {
            CapRun *c = &take[i].runs[k];
            if (c->pool) shadow_apply_run(c->pool, c->addr, c->addr + c->n, c->data), c_applied += c->n;
            free(c->data);
        }
        free(take[i].runs);
    }
    free(take);
}

int cap_split(PoolRec **rs, int n) {
    if (!__atomic_load_n(&nlist_v, __ATOMIC_RELAXED)) return n;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (!rs[i]->cap_on) {
            PoolRec *t = rs[k];
            rs[k++] = rs[i], rs[i] = t;
        }
    return k;
}

void cap_activate(PoolRec **rs, int n) {
    if (!__atomic_load_n(&active, __ATOMIC_RELAXED) || !n) return;
    plat_mutex_lock(&cap_lock);
    for (int i = 0; i < n; i++) {
        PoolRec *r = rs[i];
        if (nlist == clist) vk_mem_grow((void **)&list, &clist, sizeof *list);
        list[nlist++] = r;
        r->cap_on = 1, r->cap_from = UINT64_MAX, r->cgen = r->sgen;   // the sync just copied up to sgen
    }
    __atomic_store_n(&nlist_v, nlist, __ATOMIC_RELAXED);
    plat_mutex_unlock(&cap_lock);
}

// MKW_CAP_SPAN_KIB (default 16; -1: one query per pool, as the sync's shadow_query_pools): the captured pools are
// asked for by spans of neighbours no further apart than this. sim2-capcost (26/11/2026, 2-player race): a capture is
// 2.58 ms a submit for 756 pools and 4.5 MiB, 1.71 of them asking. Three points (759 queries over 293 MiB 1.71 ms,
// 434 over 293 MiB 1.29 with 16 KiB, 173 over 355 MiB 1.10 with 1 MiB; sim2-capspan16, sim2-capspan1m) fit 1.3 us a
// query plus 2.5 us a MiB walked: the spans take the calls' part, and what is left is walking ~290 MiB of pools to
// find 4.5 written. The pages a span reports between its pools are stamped like any other query's (query_stamp) and
// protected again, which costs their writer a fault: none with 16 KiB, 160 a submit with 1 MiB (c_stray).
// The hot history (MKW_SHADOW_HOT) is kept per pool as shadow_query_pools keeps it, from what the span stamped in it.
#define CAP_HOT_SYNCS 32   // shadow_query_pools' HOT_SYNCS
typedef struct { uint64_t lo; int i; } CapOrd;
static CapOrd *ord;
static int cord, span_kib = -2;
static uint64_t c_spans, c_stray;   // cap_lock (added after the capture)
static uint64_t t_spans, t_stray;
static uint64_t c_walk, t_walk;   // bytes of the spans asked for

static int ord_cmp(const void *a, const void *b) {
    uint64_t x = ((const CapOrd *)a)->lo, y = ((const CapOrd *)b)->lo;
    return x < y ? -1 : x > y;
}

// gen_lock held: as shadow_query, [lo, hi) stamped with g, the frozen regions left out. 0: some part is not watched.
static int span_query(uint64_t lo, uint64_t hi, uint32_t g) {
    int tracked = 1;
    for (uint64_t a = lo; a < hi;) {
        uint64_t b = (a | ((1ull << REG_SHIFT) - 1)) + 1;
        if (b > hi) b = hi;
        if (mem_frozen(a)) { a = b; continue; }
        while (b < hi && !mem_frozen(b)) b = b + (1ull << REG_SHIFT) < hi ? b + (1ull << REG_SHIFT) : hi;
        if (!query_stamp(a, b, g)) tracked = 0;
        a = b;
    }
    return tracked;
}

// gen_lock held: any page of the pool stamped g.
static int pool_stamped(const PoolRec *r, uint32_t g) {
    for (uint64_t x = r->import_lo & ~((1ull << GEN_PAGE_SHIFT) - 1); x < r->import_hi; x += 1ull << GEN_PAGE_SHIFT) {
        if (reg_max[x >> REG_SHIFT] < g) {   // nothing that new in the whole region
            x = (x | ((1ull << REG_SHIFT) - 1)) + 1 - (1ull << GEN_PAGE_SHIFT);
            continue;
        }
        const uint32_t *ch = gen_chunk[x >> GEN_CHUNK_SHIFT];
        if (ch && ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] == g) return 1;
    }
    return 0;
}

// gen_lock held: shadow_query_pools for the capture's list, by spans.
static void cap_query(PoolRec **rs, uint8_t *trk, int n, int hot_on) {
    if (span_kib == -2) {
        const char *e = cfg_get("MKW_CAP_SPAN_KIB");
        span_kib = e ? atoi(e) : 16;
        rt_log("mem: submit captures ask for the written pages %s (MKW_CAP_SPAN_KIB=%d)",
               span_kib < 0 ? "pool by pool" : "by spans of neighbouring pools", span_kib);
    }
    if (span_kib < 0) { shadow_query_pools(rs, trk, n, hot_on), t_spans += (uint64_t)n; return; }
    if (n > cord) {
        cord = n * 2;
        free(ord);
        if (!(ord = malloc(sizeof *ord * (size_t)cord))) rt_fatal("vk: out of memory (shadow capture)");
    }
    for (int i = 0; i < n; i++) ord[i] = (CapOrd){rs[i]->import_lo, i};
    qsort(ord, (size_t)n, sizeof *ord, ord_cmp);
    const uint64_t gap = (uint64_t)span_kib << 10;
    for (int a = 0; a < n;) {
        const PoolRec *r0 = rs[ord[a].i];
        int peek = hot_on && r0->shot, b = a + 1;
        uint64_t lo = r0->import_lo, hi = r0->import_hi, inside = hi - lo;
        for (; b < n; b++) {
            const PoolRec *r = rs[ord[b].i];
            if ((hot_on && r->shot ? 1 : 0) != (peek ? 1 : 0) || r->import_lo > hi + gap) break;
            if (r->import_lo > hi) inside += r->import_hi - r->import_lo;
            else if (r->import_hi > hi) inside += r->import_hi - hi;
            if (r->import_hi > hi) hi = r->import_hi;
        }
        uint32_t g = ++gen_now;
        uint64_t p0 = ww_pages;
        ww_peek = peek;
        int tracked = span_query(lo, hi, g);
        ww_peek = 0;
        t_spans++, t_walk += hi - lo;
        if (!tracked && b - a > 1) {   // something between the pools is not watched: each on its own
            for (int k = a; k < b; k++) {
                PoolRec *r = rs[ord[k].i];
                ww_peek = peek;
                trk[ord[k].i] = (uint8_t)span_query(r->import_lo, r->import_hi, ++gen_now);
                ww_peek = 0;
                t_spans++;
            }
            g = 0;   // the history below from the pool's own stamp
        } else {
            for (int k = a; k < b; k++) trk[ord[k].i] = (uint8_t)tracked;
            uint64_t pages = ww_pages - p0, in_pages = inside >> GEN_PAGE_SHIFT;
            if (!peek && pages > in_pages) t_stray += pages - in_pages;   // (a floor: the pools' unwritten pages hide some)
        }
        for (int k = a; k < b; k++) {
            PoolRec *r = rs[ord[k].i];
            if (peek) { r->shot--; continue; }
            int found = pool_stamped(r, g ? g : gen_now - (uint32_t)(b - 1 - k));
            r->shist = (uint8_t)(r->shist << 1 | found);
            if (hot_on && __builtin_popcount(r->shist) >= 4) r->shot = CAP_HOT_SYNCS, r->shist = 0;
        }
        a = b;
    }
}

static void capture_run(PoolRec *r, uint64_t a, uint64_t b, CapRun **runs, int *nruns, int *cruns, uint64_t *bytes) {
    if (*nruns == *cruns) vk_mem_grow((void **)runs, cruns, sizeof **runs);
    uint64_t t0 = qpc();
    uint8_t *d = malloc(b - a);
    if (!d) rt_fatal("vk: out of memory (shadow capture)");
    uint64_t t1 = qpc();
    mem_copy_out(d, a, b - a);   // MKW_EARLY_READ: on the game's thread, a watched page must not count as its read
    t_alloc_ns += t1 - t0, t_copy_ns += qpc() - t1;
    (*runs)[(*nruns)++] = (CapRun){r, a, b - a, d};
    *bytes += b - a;
}

// gen_lock held: the pages of r stamped after its last capture, appended to the runs (as shadow_copy_newer walks them).
static void capture_pool(PoolRec *r, CapRun **runs, int *nruns, int *cruns, uint64_t *bytes) {
    uint64_t lo = r->import_lo, hi = r->import_hi, run = 0;
    uint32_t since = r->cgen;
    r->cgen = gen_now;
    for (uint64_t k = lo >> REG_SHIFT; k <= (hi - 1) >> REG_SHIFT; k++) {
        uint64_t a = k << REG_SHIFT, b = a + (1ull << REG_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (reg_max[k] <= since || !ch) {
            if (run) capture_run(r, run, a, runs, nruns, cruns, bytes), run = 0;
            continue;
        }
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            int dirty = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since && !mem_trapped(x);
            if (dirty && !run) run = x;
            else if (!dirty && run) capture_run(r, run, x, runs, nruns, cruns, bytes), run = 0;
        }
    }
    if (run) capture_run(r, run, hi, runs, nruns, cruns, bytes);
}

void vk_mem_shadow_capture(uint64_t sub) {
    if (!shadow_live) return;
    if (mode < 0) {
        mode = rt_knob("SHADOW_SUBMIT", "0");
        rt_log("mem: the CPU's writes captured at each submit %s (MKW_SHADOW_SUBMIT)",
               mode == 2 ? "in the B blocks" : mode ? "on" : "off");
    }
    uint64_t t0 = qpc();
    int on = RT_KNOB_ON(mode);
    __atomic_store_n(&active, on, __ATOMIC_RELAXED);
    if (!on && !__atomic_load_n(&nlist_v, __ATOMIC_RELAXED)) return;
    plat_mutex_lock_shared(&pool_lock);   // the records live while it is held (vk_pool_fini takes it exclusively)
    plat_mutex_lock(&cap_lock);
    int n = 0;
    for (int i = 0; i < nlist;)
        if (!on || (list[i]->touch_sub && sub > list[i]->touch_sub + CAP_IDLE)) list_drop(i), c_dropped++;
        else i++;
    if (nlist > csnap) {
        csnap = nlist * 2;
        free(snap), free(tracked);
        snap = malloc(sizeof *snap * (size_t)csnap), tracked = malloc((size_t)csnap);
        if (!snap || !tracked) rt_fatal("vk: out of memory (shadow capture)");
    }
    memcpy(snap, list, sizeof *snap * (size_t)nlist);
    n = nlist;
    plat_mutex_unlock(&cap_lock);
    CapRun *runs = NULL;
    int nruns = 0, cruns = 0;
    uint64_t bytes = 0;
    uint64_t query_ns = 0;
    t_alloc_ns = t_copy_ns = t_spans = t_stray = t_walk = 0;   // (captures are serial: the front end's submit lock)
    if (n) {
        gen_lock_x();
        uint64_t q0 = qpc();
        cap_query(snap, tracked, n, RT_KNOB_ON(hot_mode));
        query_ns = qpc() - q0;
        for (int i = 0; i < n; i++)
            if (tracked[i]) {
                capture_pool(snap[i], &runs, &nruns, &cruns, &bytes);
                if (snap[i]->cap_from == UINT64_MAX) snap[i]->cap_from = sub;
            }
        plat_mutex_unlock(&gen_lock);
    }
    plat_mutex_lock(&cap_lock);
    for (int i = 0; i < n; i++)
        if (!tracked[i] && snap[i]->cap_on)   // untracked: read at the sync as before
            for (int k = 0; k < nlist; k++)
                if (list[k] == snap[i]) { list_drop(k); break; }
    if (nruns) {
        if (npacks == cpacks) vk_mem_grow((void **)&packs, &cpacks, sizeof *packs);
        packs[npacks++] = (CapPack){sub, runs, nruns};
    }
    c_subs++, c_pools += (uint64_t)n, c_bytes += bytes, c_ns += qpc() - t0;
    c_query_ns += query_ns, c_alloc_ns += t_alloc_ns, c_copy_ns += t_copy_ns, c_runs += (uint64_t)nruns;
    c_spans += t_spans, c_stray += t_stray, c_walk += t_walk;
    if (c_subs >= 1500) log_locked();   // ~5 s of race; shadow_log only runs at a few frames
    plat_mutex_unlock(&cap_lock);
    plat_mutex_unlock_shared(&pool_lock);
}

static void log_locked(void) {   // cap_lock held; c_applied is the replay's, read as it is
    rt_log("mem: submit captures since the last line: %llu submits, %.1f pools each, %.1f MiB captured, %.1f MiB "
           "applied, %.2f ms on the game thread (%.2f asking which pages were written in %.1f queries over %.1f MiB "
           "a submit, with "
           "%llu written pages between the pools; %.2f allocating and %.2f copying %llu runs), %d pools listed, %llu "
           "dropped, %d captures pending",
           (unsigned long long)c_subs, c_subs ? (double)c_pools / (double)c_subs : 0.0, c_bytes / 1048576.0,
           c_applied / 1048576.0, qpc_ms(c_ns), qpc_ms(c_query_ns), c_subs ? (double)c_spans / (double)c_subs : 0.0,
           c_subs ? (double)c_walk / 1048576.0 / (double)c_subs : 0.0,
           (unsigned long long)c_stray, qpc_ms(c_alloc_ns), qpc_ms(c_copy_ns),
           (unsigned long long)c_runs, nlist, (unsigned long long)c_dropped, npacks);
    c_subs = c_pools = c_bytes = c_ns = c_applied = c_dropped = 0;
    c_query_ns = c_alloc_ns = c_copy_ns = c_runs = c_spans = c_stray = c_walk = 0;
}

void cap_log(void) {   // shadow_lock held
    if (mode <= 0) return;
    plat_mutex_lock(&cap_lock);
    log_locked();
    plat_mutex_unlock(&cap_lock);
}

int vk_mem_shadow_state(uint64_t addr) {
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, addr, 1);
    const PoolRec *r = e ? e->buf : NULL;
    int st = !r ? 0 : !r->smap ? 1 : !r->cap_on ? 2 : r->cap_from > __atomic_load_n(&vk_replay_submit, __ATOMIC_RELAXED) ? 3 : 4;
    plat_mutex_unlock_shared(&pool_lock);
    return st;
}
