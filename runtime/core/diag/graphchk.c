// Job graph check (set graphchk, MKW_GRAPH_CHECK=1; 1.0.0): why the race hangs at MKW_LOGIC_HZ=120 (fps-libres).
//
// The ModuleSystem's nested calcs (main+0x1ca594, main+0x256417c) reset a job graph context (main+0x1ca6d8), fill it
// from a template (main+0x1cadc0, main+0x1321c0), start it (main+0x1ca878) and wait for it (main+0x15a3c0). At 120 Hz
// a race hangs 50-190 frames after ReadyRace with that wait never ending (lhz120tt7, 19/11/2026: of 2669 jobs, 16 wait
// on one dependency with no predecessor left and 6 ready ones were never queued). Two readings of the code fit:
//   - overflow: main+0x1cadc0 never checks its arrays. A job index past the job capacity [ctx+0x00] is written to job
//     0 (csel ... wzr, hi), an edge past [ctx+0x10] is dropped or written through a null pointer, a pointer slot past
//     [ctx+0x20] goes to slot 0; edges then name the wrong job and some never see their predecessor finish.
//   - reset in flight: the same context reset while its jobs still run (two callers sharing it) orphans them.
// So at each build ("built") the fill of the three arrays against their capacity is logged whenever it reaches a new
// high for that context (a 60 Hz run shows the headroom; a 120 Hz one whether it runs out), and at each reset any job
// without its done bit (state +0x28, 0x40000000) is logged with the context's counts.
//
// Context fields (main+0x1cadc0 and main+0x1ca6d8): [+0x00] job capacity, [+0x08] jobs (0x38 bytes each), [+0x10] edge
// capacity, [+0x20] pointer capacity, [+0xa0] jobs used, [+0xa4] edges used, [+0xa8] pointer slots used.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cpu/cpu.h"
#include "cpu/dispatch.h"
#include "host.h"
#include "platform/platform.h"

#define GRAPH_BUILT_HOST 0xdc0   // patch page offset (recomp/patches-mkw.txt, set graphchk): main+0x1ca594's built graph
#define GRAPH_BUILT2_HOST 0xdb0  // main+0x256417c's built graph, before it starts it
#define GRAPH_RESET_HOST 0xda0   // the three resets (main+0x1ca6d8), before them
#define RING_W10_HOST 0xd90      // main+0x68a88's ring checks: fill w9 against capacity - 6 in w10
#define RING_W3_HOST 0xd80       // main+0x1cab40's: fill w2 against capacity - 6 in w3
#define RING_W11_HOST 0xd70      // the worker's (main+0x663e0) pushes to ring 3: fill w10, capacity - 6 in w11
#define STALL_NS 3000000000ull   // a graph unchanged this long after its build is reported as stalled
#define STALL_LINES 160
#define PRED_MAX 4
#define RING_DROP_LINES 40
#define GUEST_MAIN 0x7000000000ull   // main's base (recomp/layout.py)
#define GRAPH_CTXS 32
#define GRAPH_RESET_LINES 40     // reset-in-flight lines at most: one hang repeats it every frame
#define JOB_SIZE 0x38
#define JOB_DONE 0x40000000u

typedef struct {
    uint64_t ctx;
    uint32_t jobs, edges, ptrs;   // the highest fill seen
} GraphSeen;

static GraphSeen seen[GRAPH_CTXS];
static uint8_t seen_lock;
static int reset_lines;
static uint64_t last_ctx, last_build_ns;   // the latest graph built, for the stall report
static uint32_t last_built_jobs;           // its job count when built: jobs past it were added while it ran
static uint8_t watch_started;
static void stall_watch(void *arg);

static uint32_t rd32(uint64_t a) { return *(const volatile uint32_t *)(uintptr_t)a; }
static uint64_t rd64(uint64_t a) { return *(const volatile uint64_t *)(uintptr_t)a; }

static void lock(void) {
    while (__atomic_test_and_set(&seen_lock, __ATOMIC_ACQUIRE)) plat_pause();
}
static void unlock(void) { __atomic_clear(&seen_lock, __ATOMIC_RELEASE); }

// The context built so far whose job array holds job, with its index there (graphchk_edge.c); 0 when none does.
uint64_t graphchk_ctx_of(uint64_t job, uint32_t *index) {
    for (int i = 0; i < GRAPH_CTXS; i++) {
        uint64_t ctx = __atomic_load_n(&seen[i].ctx, __ATOMIC_RELAXED), base;
        if (!ctx) break;
        if (!plat_mem_read(ctx + 8, &base, 8) || job < base || (job - base) % JOB_SIZE) continue;
        if ((job - base) / JOB_SIZE < rd32(ctx)) return *index = (uint32_t)((job - base) / JOB_SIZE), ctx;
    }
    return 0;
}

// main+0x1ca878 starts a graph by pushing its root jobs (no pending dependency) with main+0x1cab40 into one of three
// rings of the context, and drops a job without a word when its ring holds capacity - 6 already: that job never runs
// and every job after it waits forever. Ring of a root: a callback ([job+0x30]) runs it first; else [job+0xa] == 0 and
// a second ring ([ctx+0x78]) -> ring 1, [job+0xa] == -2 and a third ([ctx+0x90]) -> ring 2, else ring 0. Capacities
// [ctx+0x40], [ctx+0x70], [ctx+0x88]; job pointers [ctx+0x28], min([ctx+0x20], [ctx+0xa8]) of them.
static int count_roots(uint64_t ctx, uint32_t roots[3], uint32_t caps[3]) {
    static const uint32_t cap_off[3] = {0x40, 0x70, 0x88};
    for (int r = 0; r < 3; r++) caps[r] = rd32(ctx + cap_off[r]), roots[r] = 0;
    uint64_t ring1 = rd64(ctx + 0x78), ring2 = rd64(ctx + 0x90), list = rd64(ctx + 0x28);
    uint32_t n = rd32(ctx + 0xa8), pcap = rd32(ctx + 0x20), callbacks = 0;
    if (n > pcap) n = pcap;
    for (uint32_t i = 0; list && i < n; i++) {
        uint64_t job = rd64(list + (uint64_t)i * 8);
        if (!job || (rd32(job + 0x28) & 0xffff)) continue;
        if (rd64(job + 0x30)) { callbacks++; continue; }
        int16_t cls = *(const volatile int16_t *)(uintptr_t)(job + 0xa);
        roots[ring1 && !cls ? 1 : ring2 && cls == -2 ? 2 : 0]++;
    }
    int drop = 0;
    for (int r = 0; r < 3; r++) drop |= caps[r] < 6 ? roots[r] > 0 : roots[r] > caps[r] - 6;
    return drop ? -1 : (int)callbacks;
}

static void check_built(uint64_t ctx, uint64_t lr) {
    if (!ctx) return;
    uint32_t jobs = rd32(ctx + 0xa0), edges = rd32(ctx + 0xa4), ptrs = rd32(ctx + 0xa8);
    uint32_t jcap = rd32(ctx), ecap = rd32(ctx + 0x10), pcap = rd32(ctx + 0x20);
    uint32_t roots[3], caps[3];
    int callbacks = count_roots(ctx, roots, caps);
    int over = jobs > jcap || edges > ecap || ptrs > pcap || callbacks < 0;
    __atomic_store_n(&last_ctx, ctx, __ATOMIC_RELAXED);
    __atomic_store_n(&last_built_jobs, jobs, __ATOMIC_RELAXED);
    __atomic_store_n(&last_build_ns, plat_now_ns(), __ATOMIC_RELEASE);
    if (!__atomic_test_and_set(&watch_started, __ATOMIC_RELAXED)) plat_thread_spawn(stall_watch, 0, "graphchk", 0);
    lock();
    GraphSeen *s = 0;
    for (int i = 0; i < GRAPH_CTXS && !s; i++)
        if (seen[i].ctx == ctx || !seen[i].ctx) s = &seen[i], s->ctx = ctx;
    int high = s && (jobs > s->jobs || edges > s->edges || ptrs > s->ptrs);
    if (high) s->jobs = jobs, s->edges = edges, s->ptrs = ptrs;
    unlock();
    if (high || over)
        rt_log("graphchk: ctx %#llx from main+%#llx: jobs %u/%u edges %u/%u ptrs %u/%u roots %u/%u %u/%u %u/%u "
               "ring3 cap %u%s (60 Hz ticks %llu)",
               (unsigned long long)ctx, (unsigned long long)(lr - GUEST_MAIN), jobs, jcap, edges, ecap, ptrs, pcap,
               roots[0], caps[0], roots[1], caps[1], roots[2], caps[2], rd32(ctx + 0x58),
               callbacks < 0 ? " DROP" : over ? " OVERFLOW" : " new high", (unsigned long long)frame_step_ticks());
}

// The done bit of every job of the previous build; a template's job that never ran (an empty graph start skips them)
// would show here too, so the count of jobs without it is logged with the first one's state.
static void check_reset(uint64_t ctx, uint64_t lr) {
    if (!ctx || __atomic_load_n(&reset_lines, __ATOMIC_RELAXED) >= GRAPH_RESET_LINES) return;
    uint32_t jobs = rd32(ctx + 0xa0), jcap = rd32(ctx), n = jobs < jcap ? jobs : jcap, open = 0, first = 0, state = 0;
    uint64_t base = rd64(ctx + 8);
    if (!base) return;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t st = rd32(base + (uint64_t)i * JOB_SIZE + 0x28);
        if (st & JOB_DONE) continue;
        if (!open++) first = i, state = st;
    }
    if (!open || __atomic_fetch_add(&reset_lines, 1, __ATOMIC_RELAXED) >= GRAPH_RESET_LINES) return;
    rt_log("graphchk: ctx %#llx reset from main+%#llx with %u of %u jobs not done (first %u, state %#x; 60 Hz ticks %llu)",
           (unsigned long long)ctx, (unsigned long long)(lr - GUEST_MAIN), open, n, first, state,
           (unsigned long long)frame_step_ticks());
}

// main+0x1ca650: ldp x1, x0, [x21, #8] (context, scheduler). Hosts write only x0-x18: the lifted caller keeps x19-x28
// in locals across a bl and never sees a host's write to them (lhz120gc1 wrote x21 here and crashed in frame 0).
static void host_graph_built(Cpu *c) {
    c->x[1] = rd64(c->x[21] + 8);
    c->x[0] = rd64(c->x[21] + 0x10);
    check_built(c->x[1], c->x[30]);
}

// main+0x2564270 and 0x2564334: mov x0, x20 (the scheduler); x21 is the context.
static void host_graph_built2(Cpu *c) {
    c->x[0] = c->x[20];
    check_built(c->x[21], c->x[30]);
}

// main+0x1ca60c, 0x256424c and 0x25642fc: mov x0, x21 before bl main+0x1ca6d8; x21 is the context.
static void host_graph_reset(Cpu *c) {
    c->x[0] = c->x[21];
    check_reset(c->x[21], c->x[30]);
}

// The ring checks: w = capacity - 6 as the replaced sub, and a line for each job the ring is about to drop (signed
// compare, as the b.ge that follows), with the capacity naming the ring (32768, 4096, 32 in lhz120gc3).
static uint64_t ring_drops;

static uint32_t ring_check(uint32_t cap, uint32_t fill, uint64_t lr) {
    uint32_t limit = cap - 6;
    if ((int32_t)fill >= (int32_t)limit) {
        uint64_t n = __atomic_add_fetch(&ring_drops, 1, __ATOMIC_RELAXED);
        if (n <= RING_DROP_LINES)
            rt_log("graphchk: DROP %llu: ring of %u full (%u) at main+%#llx (60 Hz ticks %llu)", (unsigned long long)n,
                   cap, fill, (unsigned long long)(lr - 4 - GUEST_MAIN), (unsigned long long)frame_step_ticks());
    }
    return limit;
}

static void host_ring_w10(Cpu *c) { c->x[10] = ring_check((uint32_t)c->x[10], (uint32_t)c->x[9], c->x[30]); }
static void host_ring_w3(Cpu *c) { c->x[3] = ring_check((uint32_t)c->x[3], (uint32_t)c->x[2], c->x[30]); }
static void host_ring_w11(Cpu *c) { c->x[11] = ring_check((uint32_t)c->x[11], (uint32_t)c->x[10], c->x[30]); }

// The stall report. Job state ([job+0x28], main+0x663e0 and main+0x68a88): bits 0-15 dependencies left, 16-22
// instances not started, 23-29 instances running, 30 done, 31 set by a callback job ([job+0x30]) that runs. A job's
// successors are a list: sentinel at job+0x10, first node [job+0x18], node = {job, next}. A finishing job releases
// them only when it was the last running instance and (bit 31 or no instance left to start), so a predecessor with
// its done bit can still hold a successor; every job not done is logged with its predecessors' states to tell which.
static int rd64_safe(uint64_t a, uint64_t *v) { return plat_mem_read(a, v, 8); }

int graphchk_edge_note(uint64_t job, char *buf, int size);   // graphchk_edge.c: the job's last late edge
void graphchk_edge_counts(void);
void graphchk_edge_hosts(uint64_t patch_page);

static void report_stall(uint64_t ctx) {
    uint64_t base = rd64(ctx + 8);
    uint32_t n = rd32(ctx + 0xa0), cap = rd32(ctx);
    if (n > cap) n = cap;
    if (!base || !n || !plat_mem_readable(base, (uint64_t)n * JOB_SIZE)) return;
    uint32_t (*pred)[PRED_MAX] = calloc(n, sizeof *pred);
    uint8_t *npred = calloc(n, 1), *nopen = calloc(n, 1);   // predecessors, and those not done
    if (!pred || !npred || !nopen) { free(pred); free(npred); free(nopen); return; }
    for (uint32_t i = 0; i < n; i++) {
        uint64_t job = base + (uint64_t)i * JOB_SIZE, sent = job + 0x10, node = rd64(job + 0x18);
        for (int k = 0; node && node != sent && k < 4096; k++) {
            uint64_t s, next;
            if (!rd64_safe(node, &s) || !rd64_safe(node + 8, &next)) break;
            if (s >= base && s < base + (uint64_t)n * JOB_SIZE) {
                uint32_t j = (uint32_t)((s - base) / JOB_SIZE);
                if (npred[j] < PRED_MAX) pred[j][npred[j]] = i;
                if (npred[j] < 255) npred[j]++;
                if (!(rd32(job + 0x28) & JOB_DONE) && nopen[j] < 255) nopen[j]++;
            }
            node = next;
        }
    }
    uint32_t open = 0, waiting = 0, lines = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t st = rd32(base + (uint64_t)i * JOB_SIZE + 0x28);
        open += !(st & JOB_DONE), waiting += !(st & JOB_DONE) && (st & 0xffff);
    }
    if (!waiting) goto out;   // skipped jobs never get the done bit; a graph is stuck only with a job left waiting
    uint32_t built = __atomic_load_n(&last_built_jobs, __ATOMIC_RELAXED);
    rt_log("graphchk: STALL ctx %#llx: %u of %u jobs not done (%u when built); rings head-tail %u-%u %u-%u %u-%u %u-%u",
           (unsigned long long)ctx, open, n, built, rd32(ctx + 0x50), rd32(ctx + 0x54), rd32(ctx + 0x68), rd32(ctx + 0x6c),
           rd32(ctx + 0x80), rd32(ctx + 0x84), rd32(ctx + 0x98), rd32(ctx + 0x9c));
    // The frontier only: a job waiting on more dependencies than it has predecessors not done (an edge nobody will
    // release), or one with none left that no ring holds (the rings are empty in a stall); "+" marks a job added
    // after the build. Owner: [job+0] (a tagged pointer) and its vtable.
    // Two passes, the waiting ones first: jobs skipped by their callback never get the done bit either.
    for (uint32_t pass = 0, i = 0; pass < 2 && lines < STALL_LINES; i = i + 1 < n ? i + 1 : (pass++, 0)) {
        uint64_t job = base + (uint64_t)i * JOB_SIZE;
        uint32_t st = rd32(job + 0x28), deps = st & 0xffff;
        if ((st & JOB_DONE) || (pass == 0 ? deps <= nopen[i] : deps != 0)) continue;
        char preds[160];
        int len = 0;
        for (int k = 0; k < npred[i] && k < PRED_MAX; k++)
            len += snprintf(preds + len, sizeof preds - len, " %u%s:%#x", pred[i][k], pred[i][k] >= built ? "+" : "",
                            rd32(base + (uint64_t)pred[i][k] * JOB_SIZE + 0x28));
        preds[len] = 0;
        uint64_t owner = rd64(job), vt = 0;
        if (owner & ~1ull) rd64_safe(owner & ~1ull, &vt);   // a tagged owner (bit 0) points to its object as well
        char late[400];
        if (graphchk_edge_note(job, late, sizeof late) <= 0) late[0] = 0;
        rt_log("graphchk: STALL job %u%s state %#x class %d kind %#x callback %#llx owner %#llx vt main+%#llx "
               "preds %u (%u not done)%s%s", i, i >= built ? "+" : "", st,
               *(const volatile int16_t *)(uintptr_t)(job + 0xa), rd32(job + 0xc), (unsigned long long)rd64(job + 0x30),
               (unsigned long long)owner, (unsigned long long)(vt ? vt - GUEST_MAIN : 0), npred[i], nopen[i], preds,
               late);
        lines++;
    }
    graphchk_edge_counts();
out:
    free(pred);
    free(npred);
    free(nopen);
}

static void stall_watch(void *arg) {
    (void)arg;
    uint64_t reported = 0;
    for (;;) {
        plat_sleep_ns(500000000ull);
        uint64_t t = __atomic_load_n(&last_build_ns, __ATOMIC_ACQUIRE), ctx = __atomic_load_n(&last_ctx, __ATOMIC_RELAXED);
        if (!t || t == reported || plat_now_ns() - t < STALL_NS) continue;
        reported = t;
        report_stall(ctx);
    }
}

void graphchk_hosts(uint64_t patch_page) {
    rt_register_host(patch_page + RING_W10_HOST, host_ring_w10);
    rt_register_host(patch_page + RING_W3_HOST, host_ring_w3);
    rt_register_host(patch_page + RING_W11_HOST, host_ring_w11);
    rt_register_host(patch_page + GRAPH_BUILT_HOST, host_graph_built);
    rt_register_host(patch_page + GRAPH_BUILT2_HOST, host_graph_built2);
    rt_register_host(patch_page + GRAPH_RESET_HOST, host_graph_reset);
    graphchk_edge_hosts(patch_page);
}
