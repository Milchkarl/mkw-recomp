// MKW_FRAMES=<file>: one tab-separated row per present (frame, world clock, A/B side, interval, replay and the render
// thread's waits), read by tools/runner/abstat.py. Why: the perf lines average 60 frames, and a 20 s race at the 660M's
// pace gave 4-6 such windows per A/B side, too few to tell a change from noise (85 A/B runs to 25/11/2026: median
// standard error of B - A 1.3 ms, 18 of them past two). Rows give the blocks of MKW_AB whole, their first frames
// droppable. Buffered and flushed every 60 frames: a killed run loses at most that, and the render thread pays a
// formatted write into memory per frame and one small write() per second or so.
// host_ms: CLOCK_MONOTONIC at the row (plat_now_ns), the clock tools/runner records perf with (-k CLOCK_MONOTONIC), so
// a run's profile splits by A/B block (tools/run/perf_ab.py).
#include <stdio.h>

#include "cfg.h"
#include "host.h"
#include "prof.h"
#include "platform/platform.h"

static FILE *out;
static const char *const cls[CPUCTR_N] = {"rep", "rout", "chk", "cout", "game"};
// per value (cpuctr.c): column, divisor (µs, thousands of cycles and instructions, misses)
static const char *const val[CPUCTR_V] = {"cpu_us", "kcyc", "kins", "miss", "rq_us", "wall_us", "wait_us", "held_us"};
static const unsigned div_by[CPUCTR_V] = {1000, 1000, 1000, 1, 1000, 1000, 1000, 1000};
static int state;   // 0: MKW_FRAMES not read yet; 1: writing; -1: off

static int frames_open(void) {
    const char *path = cfg_get("MKW_FRAMES");
    if (!path || !*path) return -1;
    out = fopen(path, "w");
    if (!out) {
        rt_log("frames: cannot write %s: no per-frame rows", path);
        return -1;
    }
    setvbuf(out, NULL, _IOFBF, 1 << 16);
    fputs("frame\tworld_ms\thost_ms\tab\tinterval_us\treplay_us\tpipe_us\ttex_us\tfence_us\tpace_us", out);
    // MKW_CPU_COUNTERS: per class of thread, CPU µs, thousands of cycles and of instructions, cache misses, run queue
    // µs, the spans measured (wall µs, waits µs)
    if (cpuctr_on())
        for (int c = 0; c < CPUCTR_N; c++)
            for (int v = 0; v < CPUCTR_V; v++) fprintf(out, "\t%s_%s", cls[c], val[v]);
    fputc('\n', out);
    rt_log("frames: one row per present to %s", path);
    return 1;
}

void frames_row(uint64_t frame, uint64_t interval_ns, const uint64_t wait_ns[PERF_N], uint64_t replay_ns) {
    if (!state) state = frames_open();
    if (state < 0 || !interval_ns) return;   // the first present has no interval
    int64_t world = frame_step_world_us();
    fprintf(out, "%llu\t%.1f\t%.3f\t%d\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu", (unsigned long long)frame,
            world < 0 ? -1.0 : world / 1e3, (double)plat_now_ns() / 1e6, rt_ab(), (unsigned long long)(interval_ns / 1000),
            (unsigned long long)(replay_ns / 1000), (unsigned long long)(wait_ns[PERF_PIPE] / 1000),
            (unsigned long long)((wait_ns[PERF_TEX] + wait_ns[PERF_ASTC]) / 1000),
            (unsigned long long)((wait_ns[PERF_FENCE] + wait_ns[PERF_DRAIN]) / 1000),
            (unsigned long long)(wait_ns[PERF_PACE] / 1000));
    if (cpuctr_on()) {
        uint64_t v[CPUCTR_N][CPUCTR_V];
        cpuctr_take(v);
        for (int c = 0; c < CPUCTR_N; c++)
            for (int k = 0; k < CPUCTR_V; k++) fprintf(out, "\t%llu", (unsigned long long)(v[c][k] / div_by[k]));
    }
    fputc('\n', out);
    if (frame % 60 == 0) fflush(out);
}

void frames_flush(void) {
    if (state > 0) fflush(out);
}
