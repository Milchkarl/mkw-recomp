// MKW_SIMTRACE=<file> (measurement, off by default): one line per event of the frame's pipeline, for the frame simulator
// tools/run/framesim.py: the game's submits and its waits for them, the replay's recording of each submit, the simulated
// GPU's run of it (MKW_GPU_SIM) and each present, each with CLOCK_MONOTONIC in ns and the thread.
// Why: from 25 to 28/11/2026 the split-screen work took 157 game runs, 7 of them ceilings (one stage made free) and a
// dozen «what if the game ran ahead» runs. With every stage's start and end of one run, the simulator replays the
// same frames under another dependency or cost («the replay overlaps the GPU», «the replay costs nothing») in seconds,
// after it reproduces the run it was taken from.
// Lines: `ns event tid a b c`, events: sub (fence, handles, held: the game's submit), wb/we (a wait, begin/end, b its
// kind: 0 for the GPU to reach fence a, any thread; 2 for the replay to issue fence a, MKW_FENCE_EARLY; 3 the frame
// bound, for the replay to handle present a; 4 a drain of everything queued, a the presents queued, c why), rb/re
// (the replay records fence a: begin with its handles, end with whether it recorded), rp (the replay handled present
// a), gpu (fence, start ns, end ns) and gpd (fence, draws, cost ns) of the simulated GPU, pr (the game presents frame
// a), gwb/gwe/gwk (guest thread a blocks on a guest object / its wait ends, woken by guest thread b or c=1 timed out /
// the calling thread wakes guest thread a, as guest thread b). A few hundred events a frame under one lock, buffered; without the variable each call is one branch.
#include <stdio.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

static int state;   // 0: not read yet; 1: on; -1: off
static FILE *out;
static PlatMutex lock = PLAT_MUTEX_INIT;

int simtrace_on(void) {
    if (!state) {
        plat_mutex_lock(&lock);
        if (!state) {
            const char *p = cfg_get("MKW_SIMTRACE");
            out = p && *p ? fopen(p, "w") : NULL;
            if (out) {
                setvbuf(out, NULL, _IOFBF, 1 << 20);
                fputs("ns\tevent\ttid\ta\tb\tc\n", out);
                rt_log("simtrace: pipeline events to %s (MKW_SIMTRACE, for tools/run/framesim.py)", p);
            } else if (p && *p) {
                rt_log("simtrace: cannot open %s: off", p);
            }
            __atomic_store_n(&state, out ? 1 : -1, __ATOMIC_RELEASE);
        }
        plat_mutex_unlock(&lock);
    }
    return state > 0;
}

void simtrace(const char *ev, uint64_t a, uint64_t b, uint64_t c) {
    if (__atomic_load_n(&state, __ATOMIC_ACQUIRE) < 0 || !simtrace_on()) return;
    uint64_t now = plat_now_ns();
    unsigned tid = (unsigned)plat_thread_id();
    plat_mutex_lock(&lock);
    fprintf(out, "%llu\t%s\t%u\t%llu\t%llu\t%llu\n", (unsigned long long)now, ev, tid, (unsigned long long)a,
            (unsigned long long)b, (unsigned long long)c);
    plat_mutex_unlock(&lock);
}

void simtrace_flush(void) {
    if (state <= 0) return;
    plat_mutex_lock(&lock);
    fflush(out);
    plat_mutex_unlock(&lock);
}
