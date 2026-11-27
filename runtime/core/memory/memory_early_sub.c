// MKW_EARLY_SUBMASK (default 1 with the early-read waits; 0 off, "ab": used in the B blocks): which 64-byte pieces a
// submit writes, from its submit (gfx/nvn/api/nvn_subw.c), for the early-read waits of memory_early.c. The replay's
// mask (mem_early_gpu_wrote) comes when it records the writer, after the game's read in 98 % of the waits with the
// game ahead (sim2-piece35). A submit whose writes are all known (nvn_subw.c tells the unknown ones apart) has its
// fence kept in a ring: for it, a page it has no piece of is not written by it at all.
#include <stdint.h>

#include "cfg.h"
#include "host.h"
#include "memory_int.h"

#define PAGE 0x1000ull
#define RING 1024

typedef struct { uint64_t f[2], m[2]; } SMask;   // per guest page: the two newest known submits writing it
static SMask *smask;
static uint64_t ring[RING];   // fence % RING: the fence when all its writes are known
static PlatMutex sm_lock = PLAT_MUTEX_INIT;
static int mode = -1;

int mem_early_sub_mode(void) {
    if (mode < 0) {
        int m = mem_early_waits() ? rt_knob("EARLY_SUBMASK", "1") : 0;
        if (m && !(smask = plat_mem_alloc(RT_ADDRESS_LIMIT / PAGE * sizeof *smask))) m = 0;
        rt_log("mem: what a submit writes is known at the submit for the early-read waits %s (MKW_EARLY_SUBMASK)",
               m == 2 ? "in the B blocks" : m ? "on" : "off");
        __atomic_store_n(&mode, m, __ATOMIC_RELEASE);
    }
    return mode;
}

static uint64_t bits_of(uint64_t p, uint64_t a, uint64_t end) {   // the pieces of page p that [a, end) covers
    uint64_t lo = p > a ? p : a, hi = p + PAGE < end ? p + PAGE : end;
    unsigned first = (unsigned)(lo % PAGE / 64), last = (unsigned)((hi - 1) % PAGE / 64);
    return (~0ull >> (63 - last)) & (~0ull << first);
}

void mem_early_sub_wrote(const uint64_t *r, int n, uint64_t fence) {
    if (mem_early_sub_mode() <= 0 || !fence) return;
    plat_mutex_lock(&sm_lock);
    for (int i = 0; i < n; i++) {
        uint64_t a = r[2 * i], end = a + r[2 * i + 1];
        if (a >= RT_ADDRESS_LIMIT) continue;
        if (end > RT_ADDRESS_LIMIT || end < a) end = RT_ADDRESS_LIMIT;
        for (uint64_t p = a & ~(PAGE - 1); p < end; p += PAGE) {
            SMask *s = &smask[p / PAGE];
            if (s->f[0] != fence) {
                if (fence < s->f[0]) continue;   // submits come in fence order; never expected
                s->f[1] = s->f[0], s->m[1] = s->m[0], s->f[0] = fence, s->m[0] = 0;
            }
            s->m[0] |= bits_of(p, a, end);
        }
    }
    __atomic_store_n(&ring[fence % RING], fence, __ATOMIC_RELEASE);
    plat_mutex_unlock(&sm_lock);
}

// 1 and *bits the pieces of page p fence w writes, when its submit knew all its writes and the knob is on for this
// block; 0 unknown.
int mem_early_sub_bits(uint64_t p, uint64_t w, uint64_t *bits) {
    int m = mode;
    if (m <= 0 || !RT_KNOB_ON(m) || !w || __atomic_load_n(&ring[w % RING], __ATOMIC_ACQUIRE) != w) return 0;
    plat_mutex_lock(&sm_lock);
    const SMask *s = &smask[p / PAGE];
    int known = 1;
    if (s->f[0] == w) *bits = s->m[0];
    else if (s->f[1] == w) *bits = s->m[1];
    else if (w > s->f[1]) *bits = 0;   // newer than the older slot and not in either: it writes nothing here
    else known = 0;                    // pushed out by newer writers
    plat_mutex_unlock(&sm_lock);
    return known;
}
