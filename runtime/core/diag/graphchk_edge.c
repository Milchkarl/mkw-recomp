// Late edge check (set graphchk, MKW_GRAPH_CHECK=1; 1.0.0): which adder links a successor onto a predecessor that has
// already finished, the edge the 120 Hz race hang loses (fps-libres).
//
// A finishing job (main+0x663e0, 0x66464-0x66528) sets its done bit, reads the head of its successor list once,
// decrements each successor's dependency count and stores the empty sentinel back. The two adders that run while a
// graph runs, main+0x22612cc (ctx, pred, succ) and main+0x13e480's loop (pred x3, a job it creates), push the node onto
// the predecessor's list and only then increment the successor's count, and neither looks at the predecessor's state.
// A push after that single read is never walked and is wiped by the sentinel store: the successor waits forever on one
// dependency with no predecessor (lhz120gc6: 16 such jobs). A push before the read whose increment lands after the
// walk's decrement wraps the count to 0xffff and back to 0 without queueing it (gc6: 6 jobs, deps 0, never run).
// Both need the predecessor done or finishing when the node is pushed, so each hook reads its state right after the
// push: done there means the edge raced the release. Counts per adder, state and whether both jobs are in the same
// graph (a job pointer kept from an earlier logic step would not be), the first late edges with both jobs' owners,
// and the last late edge of each successor for the stall report (graphchk_edge_note).
//
// lhz120gc7 saw no late edge, so the hooks above cannot see the other order: the push lands while the predecessor runs
// and its walk decrements the new job before the adder's increment, so the count goes 0 -> 0xffff (no dispatch) and
// back to 0 (gc7: 6 jobs in state 0x10000, never queued). The three places that decrement a job's count also watch for
// a count already at 0: the finishing job's walk (main+0x664d8), the empty job's walk in the dispatch
// (main+0x68c3c) and the gate (main+0x1c7d48: main+0x1c7d14 opens jobs[base + index] of [obj+0x120], the
// one dependency a template job gets from bit 5 of its template byte 7, when an asynchronous task of main+0x4fbb84's
// object ends). A gate opened before the graph is built again finds last step's done job, and the build then closes
// it for good (gc7: 16 jobs with one dependency and no predecessor). The stall line says which of those its job saw.
#include <stdint.h>
#include <stdio.h>

#include "cpu/cpu.h"
#include "host.h"
#include "platform/platform.h"

#define EDGE_HOST 0xd60    // main+0x2261324: ldr w9, [x2, #0x28] in the add-edge leaf, after the push
#define EDGE2_HOST 0xd50   // main+0x13e5ac: ldr w0, [x16] in main+0x13e480's loop, after the push onto x3
#define DEC_WALK_HOST 0xd40   // main+0x664d8: ldr w9, [x1, #0x28] in the finishing job's walk (x23 the job)
#define DEC_EMPTY_HOST 0xd30  // main+0x68c3c: ldr w9, [x1, #0x28] in the dispatch's walk of an empty job (x19)
#define DEC_GATE_HOST 0xd20   // main+0x1c7d48: ldr w9, [x1, #0x28] in the gate (x19 the graph, w8 the index)
#define EDGE_LINES 60
#define EDGE_SLOTS 4096    // late edges kept by successor; a race keeps a few thousand jobs at most
#define GUEST_MAIN 0x7000000000ull
#define JOB_SIZE 0x38
#define JOB_DONE 0x40000000u
#define JOB_RUNNING 0x3f800000u

enum { ST_IDLE, ST_RUNNING, ST_DONE, ST_N };
static const char *const st_name[ST_N] = {"idle", "running", "done"};
static const char *const adder_name[2] = {"main+0x22612cc", "main+0x13e480"};

typedef struct {
    uint64_t succ, pred;
    uint32_t pred_state;
    uint8_t adder, same_graph;
} LateEdge;

enum { DEC_WALK, DEC_EMPTY, DEC_GATE, DEC_N };
static const char *const dec_name[DEC_N] = {"walk main+0x663e0", "empty job main+0x68a88", "gate main+0x1c7d14"};

// The last decrement from 0 a job saw, and its last gate opening.
typedef struct {
    uint64_t job, pred;          // pred: the walking job, or the graph for the gate
    uint64_t zero_tick, gate_tick;
    uint32_t zero_state, gate_state, gates;
    uint8_t zero_by;
} JobDec;

static uint64_t counts[2][ST_N][2];   // adder, predecessor state, both jobs in the same graph
static uint64_t dec_counts[DEC_N][2];  // decrements: all, from a count of 0
static LateEdge late[EDGE_SLOTS];
static JobDec decs[EDGE_SLOTS];
static int dec_lines;
static uint8_t late_lock;
static int lines;

uint64_t graphchk_ctx_of(uint64_t job, uint32_t *index);   // graphchk.c

static uint32_t rd32(uint64_t a) { return *(const volatile uint32_t *)(uintptr_t)a; }
static int16_t job_class(uint64_t job) { return *(const volatile int16_t *)(uintptr_t)(job + 0xa); }

static uint64_t owner_vt(uint64_t job) {
    uint64_t owner = 0, vt = 0;
    plat_mem_read(job, &owner, 8);
    if (owner & ~1ull) plat_mem_read(owner & ~1ull, &vt, 8);   // a tagged owner (bit 0) points to its object as well
    return vt ? vt - GUEST_MAIN : 0;
}

// "ctx C index/used" for a job of a graph built so far, "no graph" otherwise.
static void where(uint64_t ctx, uint32_t i, char *buf, int size) {
    if (ctx) snprintf(buf, size, "ctx %#llx %u/%u", (unsigned long long)ctx, i, rd32(ctx + 0xa0));
    else snprintf(buf, size, "no graph");
}

static void lock(void) {
    while (__atomic_test_and_set(&late_lock, __ATOMIC_ACQUIRE)) plat_pause();
}
static void unlock(void) { __atomic_clear(&late_lock, __ATOMIC_RELEASE); }

static void note_edge(int adder, uint64_t pred, uint64_t succ) {
    if (!pred) return;
    uint32_t ps = rd32(pred + 0x28), pi = 0, si = 0;
    int st = ps & JOB_DONE ? ST_DONE : ps & JOB_RUNNING ? ST_RUNNING : ST_IDLE;
    uint64_t pctx = graphchk_ctx_of(pred, &pi), sctx = graphchk_ctx_of(succ, &si);
    int same = pctx && pctx == sctx;
    uint64_t n = __atomic_add_fetch(&counts[adder][st][same], 1, __ATOMIC_RELAXED);
    if (st != ST_DONE) return;
    lock();
    late[(succ / JOB_SIZE) % EDGE_SLOTS] = (LateEdge){succ, pred, ps, (uint8_t)adder, (uint8_t)same};
    unlock();
    if (__atomic_fetch_add(&lines, 1, __ATOMIC_RELAXED) >= EDGE_LINES) return;
    char pw[64], sw[64];
    where(pctx, pi, pw, sizeof pw);
    where(sctx, si, sw, sizeof sw);
    rt_log("graphchk: LATE edge #%llu by %s: pred %s state %#x class %d vt main+%#llx -> succ %s state %#x class %d "
           "vt main+%#llx (thread %u, 60 Hz ticks %llu)",
           (unsigned long long)n, adder_name[adder], pw, ps, job_class(pred), (unsigned long long)owner_vt(pred), sw,
           rd32(succ + 0x28), job_class(succ), (unsigned long long)owner_vt(succ), plat_thread_id(),
           (unsigned long long)frame_step_ticks());
}

// A decrement about to happen at site `by` on `job`, whose state word is `state`.
static void note_dec(int by, uint64_t pred, uint64_t job, uint32_t state) {
    int zero = !(state & 0xffff);
    __atomic_add_fetch(&dec_counts[by][zero], 1, __ATOMIC_RELAXED);
    if (!zero && by != DEC_GATE) return;
    uint64_t tick = frame_step_ticks();
    lock();
    JobDec *d = &decs[(job / JOB_SIZE) % EDGE_SLOTS];
    if (d->job != job) *d = (JobDec){.job = job};
    if (zero) {
        d->pred = pred;
        d->zero_tick = tick;
        d->zero_state = state;
        d->zero_by = (uint8_t)by;
    }
    if (by == DEC_GATE) {
        d->gate_tick = tick;
        d->gate_state = state;
        d->gates++;
    }
    unlock();
    if (!zero || __atomic_fetch_add(&dec_lines, 1, __ATOMIC_RELAXED) >= EDGE_LINES) return;
    uint32_t i = 0;
    char jw[64];
    where(graphchk_ctx_of(job, &i), i, jw, sizeof jw);
    rt_log("graphchk: decrement from 0 by %s on job %s state %#x class %d vt main+%#llx, from %#llx vt main+%#llx "
           "(thread %u, 60 Hz ticks %llu)",
           dec_name[by], jw, state, job_class(job), (unsigned long long)owner_vt(job), (unsigned long long)pred,
           (unsigned long long)(by == DEC_GATE ? 0 : owner_vt(pred)), plat_thread_id(), (unsigned long long)tick);
}

// The stall report's note for a waiting job: its last late edge, decrement from 0 and gate opening. Returns the length.
int graphchk_edge_note(uint64_t job, char *buf, int size) {
    lock();
    LateEdge e = late[(job / JOB_SIZE) % EDGE_SLOTS];
    JobDec d = decs[(job / JOB_SIZE) % EDGE_SLOTS];
    unlock();
    int n = 0;
    if (e.succ == job)
        n += snprintf(buf + n, size - n, "; late edge by %s from %#llx (state then %#x, %s graph, vt main+%#llx)",
                      adder_name[e.adder], (unsigned long long)e.pred, e.pred_state, e.same_graph ? "same" : "other",
                      (unsigned long long)owner_vt(e.pred));
    if (d.job != job || n >= size) return n;
    if (d.zero_tick)
        n += snprintf(buf + n, size - n, "; decrement from 0 by %s from %#llx (state then %#x, tick %llu)",
                      dec_name[d.zero_by], (unsigned long long)d.pred, d.zero_state,
                      (unsigned long long)d.zero_tick);
    if (n < size)
        n += snprintf(buf + n, size - n, "; gate opened %u times, last at tick %llu on state %#x (now tick %llu)",
                      d.gates, (unsigned long long)d.gate_tick, d.gate_state,
                      (unsigned long long)frame_step_ticks());
    return n < size ? n : size - 1;
}

void graphchk_edge_counts(void) {
    for (int a = 0; a < 2; a++)
        for (int s = 0; s < ST_N; s++)
            rt_log("graphchk: edges by %s onto a %s predecessor: %llu in the same graph, %llu not",
                   adder_name[a], st_name[s], (unsigned long long)__atomic_load_n(&counts[a][s][1], __ATOMIC_RELAXED),
                   (unsigned long long)__atomic_load_n(&counts[a][s][0], __ATOMIC_RELAXED));
    for (int b = 0; b < DEC_N; b++)
        rt_log("graphchk: decrements by %s: %llu, %llu of them from 0", dec_name[b],
               (unsigned long long)__atomic_load_n(&dec_counts[b][0], __ATOMIC_RELAXED) +
                   __atomic_load_n(&dec_counts[b][1], __ATOMIC_RELAXED),
               (unsigned long long)__atomic_load_n(&dec_counts[b][1], __ATOMIC_RELAXED));
}

// main+0x2261324: ldr w9, [x2, #0x28]; x1 the predecessor, x2 the successor. Hosts write only x0-x18.
static void host_edge(Cpu *c) {
    c->x[9] = rd32(c->x[2] + 0x28);
    note_edge(0, c->x[1], c->x[2]);
}

// main+0x13e5ac: ldr w0, [x16]; x3 the predecessor, x16 the new job's state word.
static void host_edge2(Cpu *c) {
    c->x[0] = rd32(c->x[16]);
    note_edge(1, c->x[3], c->x[16] - 0x28);
}

// The three decrement sites, all ldr w9, [x1, #0x28] with x1 the job about to be decremented.
static void host_dec_walk(Cpu *c) {
    c->x[9] = rd32(c->x[1] + 0x28);
    note_dec(DEC_WALK, c->x[23], c->x[1], (uint32_t)c->x[9]);
}
static void host_dec_empty(Cpu *c) {
    c->x[9] = rd32(c->x[1] + 0x28);
    note_dec(DEC_EMPTY, c->x[19], c->x[1], (uint32_t)c->x[9]);
}
static void host_dec_gate(Cpu *c) {
    c->x[9] = rd32(c->x[1] + 0x28);
    note_dec(DEC_GATE, c->x[19], c->x[1], (uint32_t)c->x[9]);
}

void graphchk_edge_hosts(uint64_t patch_page) {
    rt_register_host(patch_page + DEC_WALK_HOST, host_dec_walk);
    rt_register_host(patch_page + DEC_EMPTY_HOST, host_dec_empty);
    rt_register_host(patch_page + DEC_GATE_HOST, host_dec_gate);
    rt_register_host(patch_page + EDGE_HOST, host_edge);
    rt_register_host(patch_page + EDGE2_HOST, host_edge2);
}
