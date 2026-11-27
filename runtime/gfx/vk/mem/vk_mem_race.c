#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "platform/platform.h"
#include "vk_exec.h"
#include "vk_mem_internal.h"

// ---------------------------------------------------------------- MKW_SHADOW_RACE (diagnosis, default 0)
// The GPU reads a shadow pool (vk_mem_shadow.c) straight from its host-visible buffer, and the sync before the next
// submission copies the CPU's newer bytes into that same buffer: if the GPU is still running an earlier submission that
// reads them, that one sees the newer bytes. A capture's serial replay drew a black cactus in one frame of some replays
// and not in others, with every buffer of the draws the same at the pass end. Here every shadow
// range a recording resolves is noted with its submission's fence; a sync that copies bytes different from the shadow's
// over a noted range whose fence is not done yet and is not the one being synced logs it ("srace:" lines, the first
// RACE_LINES, then a count per frame line with the shadow line). MKW_SHADOW_RACE=frame: by frame instead (below).

#define RACE_LINES 20000

typedef struct { const PoolRec *r; uint64_t lo, hi, fence, frame; } RNote;

int race_on = -1;
static PlatMutex race_lock = PLAT_MUTEX_INIT;
static RNote *notes;   // race_lock: MKW_SHADOW_RACE=frame's notes, all pools
static int nnotes, cnotes;
// The other modes keep each pool's notes in its bucket (by PoolRec address), in the order they were noted, and drop
// the finished ones from the front: one list of every pool, walked and compacted whole at each copy, grew with the game
// ahead and its loops were most of race_check's 20-38 % of the replay's samples (sim2-erdry2, sim2-erdry3, 27/11/2026).
#define NBUCKET 256
static struct { RNote *v; int n, c, head; } bucket[NBUCKET];   // race_lock
static int live_notes(void) {
    int s = 0;
    for (int i = 0; i < NBUCKET; i++) s += bucket[i].n - bucket[i].head;
    return s;
}
static int bucket_of(const PoolRec *r) { return (int)(((uintptr_t)r >> 4) * 0x9E3779B97F4A7C15ull >> 56); }
static uint64_t r_hits, r_bytes, r_lines;   // race_lock: since the last log, and lines written
static uint8_t *tmp;
static uint64_t ctmp;

// MKW_FENCE_EARLY with its precise guard (MKW_FENCE_EARLY_GUARD unset or 2; vk_sync.c prologue): the same notes, and a
// copy that would change bytes a running submission reads first waits for that submission's GPU work (race_on 2: no
// srace lines). The game, told a fence is done once recorded, rewrote such bytes when the GPU ran behind (fe3: 20
// copies at the first frame of a B block, the GPU 6 submissions behind); a wait before every part cost 7.7 ms each (fe1).
static uint64_t r_waits, r_wait_ns;   // shadow_lock: since the last log

// MKW_SHADOW_RACE=frame (race_on 3): could the game build frame N+1 while frame N is still being recorded and drawn?
// Today it cannot: it waits for N's fence first (nvnSyncWait, ~14 ms a frame in split screen, sim2-crit). Here the
// notes are kept by frame, finished or not, and each copy counts the bytes it changes over ranges that the frame
// before (age 1) and the one before that (age 2) read, and over earlier submissions of the same frame (age 0): what a
// game released early would have rewritten under them. Zero at age 1: the game keeps a frame's data apart from the
// next one's, and may run a frame ahead with no copy made at the submit. "srace-frame:" lines for the first ranges.
#define AGES 3
static uint64_t f_bytes[AGES], f_hits[AGES], f_copied, f_noted, f_lines;   // race_lock: since the last log
// The pages changed under the frame before, since the last line (the first F_TOP; the rest count as "other").
#define F_TOP 256
static struct { uint64_t page, pool_lo, pool_hi, bytes, hits; } f_top[F_TOP];
static int f_ntop;
static uint64_t f_other, f_last_log;
static void frame_log(uint64_t frame);
static uint8_t *mark;
static uint64_t cmark;

int race_level(void) {
    if (race_on < 0) {
        const char *e = cfg_get("MKW_SHADOW_RACE"), *fe = cfg_get("MKW_FENCE_EARLY"), *g = cfg_get("MKW_FENCE_EARLY_GUARD");
        race_on = e && !strcmp(e, "frame") ? 3 : e && *e && strcmp(e, "0") ? 1
                  : fe && *fe && strcmp(fe, "0") && (!g || atoi(g) == 2) ? 2 : 0;
        if (race_on == 3) rt_log("mem: shadow copies over ranges the frames before read are counted (MKW_SHADOW_RACE=frame)");
        if (race_on == 1) rt_log("mem: shadow copies over ranges a running submission reads are logged (MKW_SHADOW_RACE)");
        if (race_on == 2) rt_log("mem: shadow copies over ranges a running submission reads wait for it while fences end early (MKW_FENCE_EARLY)");
    }
    return race_on;
}

// shadow_lock held: race_check found the copy changes bytes that fence, not done, reads; with fences ending early, the
// copy waits for its GPU work (the timeline itself: done also waits for write-backs, which may need shadow_lock).
void race_wait(uint64_t fence) {
    if (!fence || !mem_fence_early) return;
    uint64_t t0 = kern_now_ns();
    VkCtx *c = vk_ctx();
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                              .pSemaphores = &c->timeline, .pValues = &fence};
    vkWaitSemaphores(c->device, &wi, 2000000000ull);
    r_waits++, r_wait_ns += kern_now_ns() - t0;
}

// Recording thread, at a resolve of shadow pool r.
void race_note(const PoolRec *r, uint64_t lo, uint64_t hi) {
    uint64_t f = vk_exec_recording_fence();
    if (!f || hi <= lo) return;
    plat_mutex_lock(&race_lock);
    RNote **v = &notes;
    int *n = &nnotes, *c = &cnotes, first = 0;
    if (race_on != 3) {
        int b = bucket_of(r);
        v = &bucket[b].v, n = &bucket[b].n, c = &bucket[b].c, first = bucket[b].head;
    }
    for (int i = *n - 1; i >= first && i >= *n - 8; i--)
        if ((*v)[i].r == r && (*v)[i].fence == f && (*v)[i].lo <= lo && (*v)[i].hi >= hi) {
            plat_mutex_unlock(&race_lock);
            return;
        }
    if (*n == *c) vk_mem_grow((void **)v, c, sizeof **v);
    (*v)[(*n)++] = (RNote){r, lo, hi, f, vk_exec_presented};
    f_noted += hi - lo;
    plat_mutex_unlock(&race_lock);
}

// race_lock held: MKW_SHADOW_RACE=frame, the copy (or merge) of [a, b) for a submission of frame `now`.
static void frame_check(const PoolRec *r, uint64_t a, uint64_t b, uint64_t sync, const uint8_t *old) {
    uint64_t now = vk_exec_presented, len = b - a;
    int k = 0, any = 0;
    if (now >= f_last_log + 120) frame_log(now), f_last_log = now;
    for (int i = 0; i < nnotes; i++)
        if (notes[i].frame + AGES > now) notes[k++] = notes[i];   // older frames go
    nnotes = k;
    f_copied += len;
    if (len > cmark && !(mark = realloc(mark, cmark = len))) rt_fatal("vk: out of memory (MKW_SHADOW_RACE)");
    for (int i = 0; i < nnotes; i++) {
        const RNote *n = &notes[i];
        if (n->r != r || n->fence >= sync || n->lo >= b || n->hi <= a) continue;
        uint64_t lo = n->lo > a ? n->lo : a, hi = n->hi < b ? n->hi : b;
        if (!any) memset(mark, 0, len), any = 1;
        uint8_t bit = (uint8_t)(1u << (now - n->frame));
        for (uint64_t x = lo - a; x < hi - a; x++) mark[x] |= bit;
    }
    if (!any) return;
    if (len > ctmp && !(tmp = realloc(tmp, ctmp = len))) rt_fatal("vk: out of memory (MKW_SHADOW_RACE)");
    if (old) memcpy(tmp, old, len);
    else mem_copy_out(tmp, a, len);
    const uint8_t *s = r->smap + (a - r->import_lo);
    uint64_t d[AGES] = {0}, first[AGES], last[AGES] = {0};
    for (int g = 0; g < AGES; g++) first[g] = UINT64_MAX;
    for (uint64_t x = 0; x < len; x++) {
        if (!mark[x] || tmp[x] == s[x]) continue;
        for (int g = 0; g < AGES; g++)
            if (mark[x] >> g & 1) d[g]++, last[g] = x, first[g] = first[g] == UINT64_MAX ? x : first[g];
    }
    for (int g = 0; g < AGES; g++) {
        if (!d[g]) continue;
        f_hits[g]++, f_bytes[g] += d[g];
        if (g == 1) {
            uint64_t page = (a + first[g]) & ~0xfffull;
            int j = 0;
            while (j < f_ntop && f_top[j].page != page) j++;
            if (j == F_TOP) f_other += d[g];
            else {
                if (j == f_ntop) f_top[f_ntop++] = (typeof(f_top[0])){page, r->import_lo, r->import_hi, 0, 0};
                f_top[j].bytes += d[g], f_top[j].hits++;
            }
        }
        if (g && f_lines++ < 60)
            rt_log("srace-frame: f%llu %s for submission %llu changes %llu bytes in [%#llx, %#llx] that frame f%llu read "
                   "(pool [%#llx, %#llx))", (unsigned long long)now, old ? "merge" : "copy", (unsigned long long)sync,
                   (unsigned long long)d[g], (unsigned long long)(a + first[g]), (unsigned long long)(a + last[g]),
                   (unsigned long long)(now - (uint64_t)g), (unsigned long long)r->import_lo, (unsigned long long)r->import_hi);
    }
}

// shadow_lock held, before [a, b) of r is copied in from guest memory for the submission with fence sync; or, old not
// NULL, after a merge by bytes changed it (vk_mem_merge.c): old holds the shadow's bytes from before.
uint64_t race_check(const PoolRec *r, uint64_t a, uint64_t b, uint64_t sync, const uint8_t *old) {
    uint64_t done = vk_sync_fence_done(), wait = 0;
    plat_mutex_lock(&race_lock);
    if (race_on == 3) {
        frame_check(r, a, b, sync, old);
        plat_mutex_unlock(&race_lock);
        return 0;
    }
    __typeof__(bucket[0]) *bk = &bucket[bucket_of(r)];
    while (bk->head < bk->n && bk->v[bk->head].fence <= done) bk->head++;   // the finished ones go
    if (bk->head == bk->n) bk->head = bk->n = 0;
    else if (bk->head > 64 && bk->head * 2 > bk->n) {   // noted out of order (chunk threads): a done one may linger
        memmove(bk->v, bk->v + bk->head, (size_t)(bk->n - bk->head) * sizeof *bk->v);
        bk->n -= bk->head, bk->head = 0;
    }
    for (int i = bk->head; i < bk->n; i++) {
        const RNote *n = &bk->v[i];
        if (n->r != r || n->fence <= done || n->fence >= sync || n->lo >= b || n->hi <= a) continue;
        // The guard (race_on 2) only needs the newest fence whose range changes, so an older note is skipped and the
        // bytes are compared with memcmp: counting them byte by byte under shadow_lock was 30-38 % of the replay's
        // samples with the game ahead, which holds more notes (sim2-erdry2, 27/11/2026). Its log line counts the
        // copies then, not the bytes (0).
        if (race_on == 2 && n->fence <= wait) continue;
        uint64_t lo = n->lo > a ? n->lo : a, hi = n->hi < b ? n->hi : b, len = hi - lo;
        const uint8_t *s = r->smap + (lo - r->import_lo);
        if (race_on == 2 && old) {
            if (memcmp(old + (lo - a), s, len)) r_hits++, wait = n->fence;
            continue;
        }
        if (len > ctmp && !(tmp = realloc(tmp, ctmp = len))) rt_fatal("vk: out of memory (MKW_SHADOW_RACE)");
        if (old) memcpy(tmp, old + (lo - a), len);
        else mem_copy_out(tmp, lo, len);
        if (race_on == 2) {
            if (memcmp(tmp, s, len)) r_hits++, wait = n->fence;
            continue;
        }
        uint64_t diff = 0, first = UINT64_MAX, last = 0;
        for (uint64_t x = 0; x < len; x++)
            if (tmp[x] != s[x]) diff++, last = x, first = first == UINT64_MAX ? x : first;
        if (!diff) continue;
        r_hits++, r_bytes += diff;
        if (n->fence > wait) wait = n->fence;
        if (race_on == 1 && r_lines++ < RACE_LINES)
            rt_log("srace: f%llu %s sync %llu (done %llu) over [%#llx, %#llx) read by %llu: %llu bytes differ in [%#llx, %#llx]",
                   (unsigned long long)vk_exec_presented, old ? "merge" : "copy", (unsigned long long)sync, (unsigned long long)done,
                   (unsigned long long)n->lo, (unsigned long long)n->hi, (unsigned long long)n->fence,
                   (unsigned long long)diff, (unsigned long long)(lo + first), (unsigned long long)(lo + last));
    }
    plat_mutex_unlock(&race_lock);
    return wait;
}

// race_lock held: MKW_SHADOW_RACE=frame, every 120 frames.
static void frame_log(uint64_t frame) {
    {
        rt_log("mem: shadow race by frame at frame %llu: %.1f KiB copied in, %.1f KiB noted as read; changed under an "
               "earlier submission of the same frame %llu bytes (%llu copies), under the frame before %llu bytes (%llu), "
               "under the one before that %llu bytes (%llu); %d ranges noted", (unsigned long long)frame,
               (double)f_copied / 1024, (double)f_noted / 1024, (unsigned long long)f_bytes[0],
               (unsigned long long)f_hits[0], (unsigned long long)f_bytes[1], (unsigned long long)f_hits[1],
               (unsigned long long)f_bytes[2], (unsigned long long)f_hits[2], nnotes);
        memset(f_bytes, 0, sizeof f_bytes), memset(f_hits, 0, sizeof f_hits), f_copied = f_noted = 0;
    }
    for (int n = 0; n < 16; n++) {   // the 16 pages with most bytes
        int best = -1;
        for (int j = 0; j < f_ntop; j++)
            if (f_top[j].bytes && (best < 0 || f_top[j].bytes > f_top[best].bytes)) best = j;
        if (best < 0) break;
        rt_log("srace-top: f%llu page %#llx of pool [%#llx, %#llx) (%llu KiB): %llu bytes in %llu copies under the frame before",
               (unsigned long long)frame, (unsigned long long)f_top[best].page, (unsigned long long)f_top[best].pool_lo,
               (unsigned long long)f_top[best].pool_hi, (unsigned long long)((f_top[best].pool_hi - f_top[best].pool_lo) >> 10),
               (unsigned long long)f_top[best].bytes, (unsigned long long)f_top[best].hits);
        f_top[best].bytes = 0;
    }
    if (f_ntop) rt_log("srace-top: f%llu %d pages in all, %llu bytes in pages past the first %d", (unsigned long long)frame,
                       f_ntop, (unsigned long long)f_other, F_TOP);
    f_ntop = 0, f_other = 0;
}

void race_log(uint64_t frame) {
    plat_mutex_lock(&race_lock);
    if (race_on == 3) {   // its own lines, every 120 frames (frame_check)
        plat_mutex_unlock(&race_lock);
        return;
    }
    rt_log("mem: shadow race at frame %llu: %llu copies over ranges a running submission reads, %llu bytes changed; "
           "%d ranges noted; %llu copies waited for it, %.1f ms", (unsigned long long)frame, (unsigned long long)r_hits,
           (unsigned long long)r_bytes, live_notes(), (unsigned long long)r_waits, r_wait_ns / 1e6);
    r_hits = r_bytes = r_waits = r_wait_ns = 0;
    plat_mutex_unlock(&race_lock);
}

// ---------------------------------------------------------------- MKW_SHADOW_WATCH=<lo>-<hi>[,...] (hex; diagnosis)
// Every change the shadow syncs make to the shadow inside [lo, hi), every pending GPU write they skip there and every
// write-back over it, one "swatch:" line each with the frame and the submission: the order of CPU copies and GPU
// results over one buffer (the per-instance records the culling dispatch writes).
uint64_t watch_hi;   // the highest end: 0 without the switch
static uint64_t wr[4][2];
static int nwr;

void watch_init(void) {
    const char *e = cfg_get("MKW_SHADOW_WATCH");
    for (const char *p = e; p && nwr < 4 && strchr(p, '-');) {
        char *end;
        wr[nwr][0] = strtoull(p, &end, 16);
        wr[nwr][1] = strtoull(end + 1, &end, 16);
        if (wr[nwr][1] > watch_hi) watch_hi = wr[nwr][1];
        rt_log("mem: shadow changes over [%#llx, %#llx) logged (MKW_SHADOW_WATCH)", (unsigned long long)wr[nwr][0],
               (unsigned long long)wr[nwr][1]);
        nwr++;
        p = *end == ',' ? end + 1 : NULL;
    }
}

static int watched_range(uint64_t a, uint64_t b) {
    for (int i = 0; i < nwr; i++)
        if (a < wr[i][1] && b > wr[i][0]) return 1;
    return 0;
}

static void watch_one(const PoolRec *r, uint64_t lo, uint64_t hi, uint64_t sync);

// shadow_lock held, before [a, b) of r is copied in from guest memory for the submission with fence sync.
void watch_copy(const PoolRec *r, uint64_t a, uint64_t b, uint64_t sync) {
    for (int i = 0; i < nwr; i++) {
        uint64_t lo = a > wr[i][0] ? a : wr[i][0], hi = b < wr[i][1] ? b : wr[i][1];
        if (lo < hi) watch_one(r, lo, hi, sync);
    }
}

static void watch_one(const PoolRec *r, uint64_t lo, uint64_t hi, uint64_t sync) {
    uint64_t len = hi - lo;
    plat_mutex_lock(&race_lock);
    if (len > ctmp && !(tmp = realloc(tmp, ctmp = len))) rt_fatal("vk: out of memory (MKW_SHADOW_WATCH)");
    mem_copy_out(tmp, lo, len);
    const uint8_t *s = r->smap + (lo - r->import_lo);
    uint64_t diff = 0, zero = 0, first = 0;
    for (uint64_t x = 0; x < len; x++)
        if (tmp[x] != s[x]) first = diff++ ? first : x, zero += !tmp[x];
    if (diff)
        rt_log("swatch: f%llu sync %llu (done %llu) copy [%#llx, %#llx): %llu bytes change from %#llx, %llu of them to zero",
               (unsigned long long)vk_exec_presented, (unsigned long long)sync,
               (unsigned long long)vk_sync_fence_done(), (unsigned long long)lo, (unsigned long long)hi,
               (unsigned long long)diff, (unsigned long long)(lo + first), (unsigned long long)zero);
    plat_mutex_unlock(&race_lock);
}

void watch_event(const char *what, uint64_t lo, uint64_t hi, uint64_t fence, uint64_t bytes) {
    if (!watched_range(lo, hi)) return;
    rt_log("swatch: f%llu %s [%#llx, %#llx) fence %llu (done %llu): %llu bytes", (unsigned long long)vk_exec_presented,
           what, (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)fence,
           (unsigned long long)vk_sync_fence_done(), (unsigned long long)bytes);
}
