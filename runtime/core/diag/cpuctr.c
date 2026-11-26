// MKW_CPU_COUNTERS=1: each thread's CPU time, cycles, instructions and cache misses (perf_event_open, counting only)
// and its time waiting for a core (run_delay of /proc/thread-self/schedstat), summed per class of thread (the replay
// thread recording a submission and the rest of its time, its chunk threads, the game's submitting threads), with
// spans the callers measure (the replay's recording wall time, its waits for chunks, each chunk's wall time); written
// per frame as extra MKW_FRAMES columns (Linux only).
// Why: with the game ahead of the replay (MKW_FENCE_AHEAD) the replay took 10-27 % longer per frame. sim2-ctr showed its
// CPU time, instructions, clock and instructions per cycle unchanged: the longer recording is time off the CPU, either
// waiting for a core, for its chunks or for a lock; wall minus CPU, run queue and chunk waits leaves the locks. Each
// thread reads its own counters when it ticks (a read() and a pread() per submission or chunk); without the variable a
// tick is one branch.
#include <stdint.h>

#include "cfg.h"
#include "host.h"
#include "prof.h"

static int state;   // 0: not read yet; 1: on; -1: off
static uint64_t acc[CPUCTR_N][CPUCTR_V];

int cpuctr_on(void) {
    if (!state) {
        const char *v = cfg_get("MKW_CPU_COUNTERS");
        state = v && *v && *v != '0' ? 1 : -1;
#ifndef __linux__
        if (state > 0) rt_log("cpuctr: MKW_CPU_COUNTERS needs Linux's perf_event_open: off"), state = -1;
#endif
    }
    return state > 0;
}

void cpuctr_take(uint64_t out[CPUCTR_N][CPUCTR_V]) {
    for (int c = 0; c < CPUCTR_N; c++)
        for (int v = 0; v < CPUCTR_V; v++) out[c][v] = __atomic_exchange_n(&acc[c][v], 0, __ATOMIC_RELAXED);
}

void cpuctr_add(int cls, int what, uint64_t ns) {
    if (__builtin_expect(state <= 0, 1) && !cpuctr_on()) return;
    __atomic_fetch_add(&acc[cls][what], ns, __ATOMIC_RELAXED);
}

#ifdef __linux__
#include <fcntl.h>
#include <linux/perf_event.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static __thread int fd = -1, sfd = -1, nev, tried;
static __thread uint64_t last[CPUCTR_RQ + 1];

static int open_ev(uint32_t type, uint64_t config, int group) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a, a.type = type, a.config = config;
    a.read_format = PERF_FORMAT_GROUP;
    a.disabled = group < 0;   // the leader starts the group
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, group, 0);
}

static uint64_t cpu_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static uint64_t run_delay_ns(void) {   // schedstat: "time on the CPU, time waiting on a run queue, timeslices"
    char b[96];
    ssize_t n = sfd < 0 ? -1 : pread(sfd, b, sizeof b - 1, 0);
    if (n <= 0) return 0;
    b[n] = 0;
    char *p = strchr(b, ' ');
    return p ? strtoull(p + 1, NULL, 10) : 0;
}

static void thread_open(void) {
    static int logged;
    tried = 1;
    fd = open_ev(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
    if (fd < 0) {
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log("cpuctr: perf_event_open failed: no counters");
        return;
    }
    nev = 1;
    if (open_ev(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, fd) >= 0) nev++;
    if (nev == 2 && open_ev(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, fd) >= 0) nev++;
    ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    sfd = open("/proc/thread-self/schedstat", O_RDONLY | O_CLOEXEC);
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("cpuctr: per-thread cycles%s%s%s to the MKW_FRAMES columns", nev > 1 ? ", instructions" : "",
               nev > 2 ? ", cache misses" : "", sfd >= 0 ? ", run queue waits" : "");
    last[CPUCTR_CPU] = cpu_ns(), last[CPUCTR_RQ] = run_delay_ns();
}

void cpuctr_tick(int cls) {
    if (__builtin_expect(state <= 0, 1) && !cpuctr_on()) return;
    if (!tried) thread_open();
    if (fd < 0) return;
    uint64_t buf[1 + CPUCTR_MISS] = {0}, now[CPUCTR_RQ + 1] = {cpu_ns()};
    if (read(fd, buf, sizeof buf) < (ssize_t)(sizeof(uint64_t) * (1 + (size_t)nev))) return;
    for (int v = CPUCTR_CYC; v <= CPUCTR_MISS; v++) now[v] = v <= nev ? buf[v] : 0;
    now[CPUCTR_RQ] = run_delay_ns();
    for (int v = 0; v <= CPUCTR_RQ; v++) __atomic_fetch_add(&acc[cls][v], now[v] - last[v], __ATOMIC_RELAXED), last[v] = now[v];
}
#else
void cpuctr_tick(int cls) { (void)cls; }
#endif
