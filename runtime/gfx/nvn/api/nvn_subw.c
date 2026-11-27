// MKW_EARLY_SUBMASK (default 1; 0 off, "ab": used in the B blocks): what a submit's GPU work writes in guest memory,
// known at the submit, for the early-read waits (core/memory/memory_early.c). With the game ahead (MKW_FENCE_EARLY) a
// read of a rotating copy written two frames before shares its edge page with the copy the frame before writes; only
// the replay's recording told which 64-byte pieces a writer writes, and when the game reads, the newest writer is not
// recorded yet in 98 % of the waits (sim2-piece35): the read waited for bytes it does not read.
//
// Here the walk of resolve_deferred (nvn_api_cmd.c) lists the ranges as the replay writes them back: the copy and
// clear destinations, counters and uniform updates, and for each draw and dispatch the storage buffers its program
// writes (the slots per program and stage come from the replay's translation, nvn_progw_*), as vk_exec_bind.c
// ssbo_info binds them. The replay keeps its bind state across submits and per queue (nvn_qorder.c), so only binds
// made in the same submit are used: a draw writing through a bind from another submit, a program the replay has not
// translated yet, a texture-to-buffer copy (its pitch) or a storage buffer bound with no size make the submit unknown,
// and its reads wait as before. Game thread, under the front end's submit lock.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "platform/platform.h"

#define STAGES 8
#define CS 5        // NVN_STAGE_CS
#define SUBW_MAX 1024
#define PW_N 16384  // programs (a new serial at every nvnProgramSetShaders)
#define PW_KNOWN (1ull << 63)

uint32_t nvn_subw_align;   // vk_caps.c: storage buffers bind from their address rounded down to it (0: exact)

// Per program serial: the NVN stages it has (bit 8: known) and, per stage, NVN_PROGW_* bits | PW_KNOWN. Written by the
// replay's threads, read by the game thread.
typedef struct { uint64_t serial; uint32_t present; uint64_t w[STAGES]; } ProgW;
static ProgW *pw;
static PlatMutex pw_lock = PLAT_MUTEX_INIT;

static ProgW *pw_find(uint64_t serial, int add) {
    if (!serial || (!pw && !add)) return NULL;
    if (!pw) {
        plat_mutex_lock(&pw_lock);
        if (!pw) __atomic_store_n(&pw, calloc(PW_N, sizeof *pw), __ATOMIC_RELEASE);
        plat_mutex_unlock(&pw_lock);
        if (!pw) return NULL;
    }
    for (uint32_t i = (uint32_t)(serial * 0x9E3779B97F4A7C15ull >> 50), k = 0; k < 64; k++, i = (i + 1) & (PW_N - 1)) {
        uint64_t s = __atomic_load_n(&pw[i].serial, __ATOMIC_ACQUIRE);
        if (s == serial) return &pw[i];
        if (s) continue;
        if (!add) return NULL;
        if (__atomic_compare_exchange_n(&pw[i].serial, &s, serial, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) || s == serial)
            return &pw[i];
    }
    return NULL;   // full around it: unknown
}

void nvn_progw_present(uint64_t serial, uint32_t stages) {
    ProgW *e = pw_find(serial, 1);
    if (e) __atomic_store_n(&e->present, (stages & 0xff) | 0x100, __ATOMIC_RELEASE);
}

void nvn_progw_stage(uint64_t serial, int stage, uint64_t bits) {
    ProgW *e = (unsigned)stage < STAGES ? pw_find(serial, 1) : NULL;
    if (e) __atomic_store_n(&e->w[stage], bits | PW_KNOWN, __ATOMIC_RELEASE);
}

int nvn_subw_on(void) { return mem_early_sub_mode(); }

// This submit: its write ranges, binds made in it, and why it is unknown (UNK_*).
enum { UNK_PROG, UNK_BIND, UNK_T2B, UNK_SIZE, UNK_FULL, UNK_KINDS };
static const char *const unk_name[UNK_KINDS] = {"program not translated yet", "bind from another submit",
                                                "texture-to-buffer copy", "storage buffer without a size", "too many ranges"};
static struct {
    int on, unknown, n;
    uint64_t r[2 * SUBW_MAX];
    uint64_t serial;
    int prog_set;
    uint16_t ssbo_set[STAGES], ubo_set;
    p_bind_buf ssbo[STAGES][16], ubo[16];   // ubo: the compute stage's, read as storage at c0 0x210 (vk_exec_bind.c)
} S;
static uint64_t n_sub, n_known, n_unk[UNK_KINDS];

void nvn_subw_begin(void) {
    S.on = nvn_subw_on() != 0;
    S.unknown = 0, S.n = 0, S.serial = 0, S.prog_set = 0, S.ubo_set = 0;
    memset(S.ssbo_set, 0, sizeof S.ssbo_set);
}

static void unknown(int why) {
    if (!S.unknown) S.unknown = why + 1;
}

static void add(uint64_t a, uint64_t n) {
    if (!a || !n || nvn_gpu_only(a)) return;
    for (int i = S.n - 1; i >= 0 && i >= S.n - 16; i--)
        if (S.r[2 * i] == a && S.r[2 * i + 1] == n) return;
    if (S.n == SUBW_MAX) return unknown(UNK_FULL);
    S.r[2 * S.n] = a, S.r[2 * S.n + 1] = n, S.n++;
}

static void add_bind(const p_bind_buf *b) {
    if (!b->addr) return;   // the replay binds a zero buffer
    if (!b->size) return unknown(UNK_SIZE);   // bound to the end of its backing
    uint64_t lead = nvn_subw_align ? b->addr & (nvn_subw_align - 1) : 0;
    add(b->addr - lead, b->size + lead);
}

static void use(uint32_t stages) {
    if (!S.prog_set) return unknown(UNK_BIND);
    if (!S.serial) return;   // no program: the replay draws nothing
    ProgW *e = pw_find(S.serial, 0);
    uint32_t present = e ? __atomic_load_n(&e->present, __ATOMIC_ACQUIRE) : 0;
    if (!(present & 0x100)) return unknown(UNK_PROG);
    stages &= present;
    for (int st = 0; st < STAGES; st++) {
        if (!(stages >> st & 1)) continue;
        uint64_t w = __atomic_load_n(&e->w[st], __ATOMIC_ACQUIRE);
        if (!(w & PW_KNOWN) || (w & NVN_PROGW_OTHER)) return unknown(UNK_PROG);
        for (uint32_t m = (uint32_t)(w & 0xffff); m; m &= m - 1) {
            int k = __builtin_ctz(m);
            if (!(S.ssbo_set[st] >> k & 1)) return unknown(UNK_BIND);
            add_bind(&S.ssbo[st][k]);
        }
        for (uint32_t m = (uint32_t)(w >> 16 & 0xffff); m; m &= m - 1) {
            int k = __builtin_ctz(m);
            if (!(S.ubo_set >> k & 1)) return unknown(UNK_BIND);
            add_bind(&S.ubo[k]);
        }
    }
}

void nvn_subw_op(uint32_t op, const uint8_t *a) {
    if (!S.on || S.unknown) return;
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = (const p_bind_program *)a;
        S.serial = b->prog ? b->prog->serial : 0, S.prog_set = 1;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = (const p_bind_buf *)a;
        if (b->stage >= 0 && b->stage < STAGES && b->index >= 0 && b->index < 16)
            S.ssbo[b->stage][b->index] = *b, S.ssbo_set[b->stage] |= (uint16_t)(1u << b->index);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = (const p_bind_buf *)a;
        if (b->stage == CS && b->index >= 0 && b->index < 16) S.ubo[b->index] = *b, S.ubo_set |= (uint16_t)(1u << b->index);
        break;
    }
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: case OP_DRAW_MESH: use(0xff & ~(1u << CS)); break;
    case OP_DISPATCH: case OP_DISPATCH_IND: use(1u << CS); break;
    case OP_COPY_B2B: add(((const p_copy_b2b *)a)->dst, ((const p_copy_b2b *)a)->size); break;
    case OP_CLEAR_BUF: add(((const p_clear_buf *)a)->addr, ((const p_clear_buf *)a)->size); break;
    case OP_REPORT_COUNTER: add(((const p_report_counter *)a)->addr, 16); break;
    case OP_UPDATE_UB: add(((const p_update_ub *)a)->dst, ((const p_update_ub *)a)->size); break;
    case OP_COPY_T2B: unknown(UNK_T2B); break;
    default: break;
    }
}

void nvn_subw_commit(uint64_t fence) {
    if (!S.on) return;
    n_sub++;
    if (S.unknown) n_unk[S.unknown - 1]++;
    else n_known++, mem_early_sub_wrote(S.r, S.n, fence);
    S.on = 0;
    if (n_sub == 3000) {
        char b[256];
        int k = 0;
        for (int i = 0; i < UNK_KINDS; i++)
            if (n_unk[i]) k += snprintf(b + k, sizeof b - (size_t)k, ", %s %llu", unk_name[i], (unsigned long long)n_unk[i]);
        b[k] = 0;
        rt_log("nvn: submits since the last line: 3000, writes known at the submit for %llu; unknown%s",
               (unsigned long long)n_known, k ? b : ": none");
        n_sub = n_known = 0, memset(n_unk, 0, sizeof n_unk);
    }
}
