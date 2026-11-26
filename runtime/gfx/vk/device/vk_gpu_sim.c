// MKW_GPU_SIM=<ms per 1000 draws> (measurement, off by default): a GPU of a chosen speed in place of this machine's.
// The game and the replay run as always (every command is recorded, uploads and sparse binds are real), but the
// recorded command buffers are not handed to the queue, and a submission counts as finished when a simulated GPU
// would have finished it: one job at a time, in submit order, each costing its draws x the given time. Every fence
// wait, the completion callbacks and the frees behind them follow that clock.
//
// Why: in split screen our GPUs are the limit and hide how the game, the replay and the GPU wait for each other (the
// 660M's frame did not move with or without MKW_ACQUIRE_GPU, ss2-acq). With the GPU's time set to a player's (RTX
// 4080, 4 players: 8.6 ms for ~6,400 draws, 1.35 ms per 1000) the same switches can be compared on this CPU.
//
// The model: a submission's GPU work starts when its first part reaches the queue (the chunked recording sends parts
// early) or when the GPU finishes the one before, whichever is later, and it cannot end before its last part is sent.
// What it does not show: the picture (nothing is drawn; the window shows whatever the images hold), occlusion query
// results (unavailable, counted as visible) and anything the GPU writes for the CPU to read.
#include <stdlib.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk_gpu_sim.h"

#define RING 4096   // values in flight are far fewer: the completion thread reads a value's entry before RING more exist

static int on = -1;
static double ns_per_draw;
static uint64_t due[RING];        // when the simulated GPU finishes each value (kern_now_ns)
static uint64_t last_due;         // of the newest issued value
static uint64_t first_part_ns;    // when the first part of the submission being recorded was sent, 0 none yet
static uint32_t pend_draws;
// The line every LOG_EVERY submissions with draws: how busy the simulated GPU was and how long work sat behind it.
#define LOG_EVERY 600
static uint64_t st_since, st_busy_ns, st_behind_ns, st_draws, st_n;

int vk_sim_on(void) {
    if (on < 0) {
        const char *e = cfg_get("MKW_GPU_SIM");
        double ms = e && *e ? strtod(e, NULL) : 0;
        ns_per_draw = ms * 1000.0;   // ms per 1000 draws = us per draw
        if (ms > 0)
            rt_log("vk: simulated GPU: %.2f ms per 1000 draws, recorded command buffers are not executed, the picture "
                   "is not valid (MKW_GPU_SIM)", ms);
        on = ms > 0;
    }
    return on;
}

void vk_sim_draws(uint32_t draws) { pend_draws += draws; }

void vk_sim_part(void) {
    if (!first_part_ns) first_part_ns = kern_now_ns();
}

void vk_sim_issue(uint64_t value) {
    uint64_t now = kern_now_ns(), start = first_part_ns ? first_part_ns : now;
    uint64_t cost = (uint64_t)(pend_draws * ns_per_draw);
    if (start < last_due) {
        if (pend_draws) st_behind_ns += last_due - start;
        start = last_due;
    }
    uint64_t end = start + cost;
    if (end < now) end = now;
    due[value % RING] = last_due = end;
    if (pend_draws) {
        if (!st_since) st_since = now;
        st_busy_ns += cost, st_draws += pend_draws;
        if (++st_n >= LOG_EVERY) {
            rt_log("perf: simulated GPU: busy %.0f ms per s, %.2f ms and %.0f draws a submission with draws, %.2f ms "
                   "each behind the one before (MKW_GPU_SIM)", (double)st_busy_ns / (double)(now - st_since) * 1e3,
                   (double)st_busy_ns / 1e6 / (double)st_n, (double)st_draws / (double)st_n,
                   (double)st_behind_ns / 1e6 / (double)st_n);
            st_since = now, st_busy_ns = st_behind_ns = st_draws = st_n = 0;
        }
    }
    first_part_ns = 0, pend_draws = 0;
}

uint64_t vk_sim_reached(uint64_t done, uint64_t reached) {
    if (reached <= done) return reached;
    uint64_t now = kern_now_ns(), r = done;
    while (r < reached && due[(r + 1) % RING] <= now) r++;
    if (r > done) return r;
    plat_sleep_ns(due[(done + 1) % RING] - now);
    return done + 1;
}
