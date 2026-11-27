// MKW_REC_CENSUS=1 (diagnosis, default off): how long before its submit the game finished recording the draws it
// submits, and how many deferred binds read new handles at the submit. The replay translates a submission only once
// it is submitted; translating each recording when the game ends it instead (on its own thread, while the game goes on
// with the frame) can only gain the time between those two moments, and only for what does not depend on the submit
// (deferred texture and sampler handles, nvn_api_cmd.c resolve_deferred). CallCommands copies a recording into its
// caller, so the end times travel with the draw counts: each copy holds its draws, the sum of each draw's end time and
// the earliest and latest end among them (the caller's own draws end at the caller's EndRecording).
// One line every 1500 submits: draws per submit, the draw-weighted mean lead (submit time - end time), the leads of
// the earliest and latest recordings, and the deferred binds read and changed in that window.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_cmd_internal.h"
#include "nvn_hmap.h"
#include "nvn_obj.h"
#include "platform/platform.h"

// The draws of a copy by the end time of the recording they came from: RC_E entries, the nearest two merged when full
// (the end of the later one kept: a lead too short rather than too long). sum: draws x end time (ns).
#define RC_E 24
typedef struct { uint64_t draws, sum, lo, hi; int n; struct { uint64_t end, draws; } e[RC_E]; } RcInfo;

static int on = -1;
static PlatMutex rc_lock = PLAT_MUTEX_INIT;
static nvn_hmap by_handle = {.klen = 8};   // recording copy -> RcInfo (entries of freed copies stay until reused)
static nvn_hmap by_cmd = {.klen = 8};      // command buffer recording now -> RcInfo of the recordings it called
// rc_lock: this window. by_age: draws whose recording ended less than 40 ms (this frame or the last), 40 ms to 1 s,
// and over 1 s before the submit; recs: recordings (entries) per submit.
static struct { uint64_t submits, dsubs, draws, lead, first, last, def_reads, def_changed, by_age[3], recs; } w;

int nvn_rc_on(void) {
    if (on < 0) {
        const char *v = cfg_get("MKW_REC_CENSUS");   // a diagnosis: cfg_get, no cfg.c row
        on = v && *v && strcmp(v, "0");
        if (on) rt_log("nvn: recording census on (MKW_REC_CENSUS): lead of each submit's draws over their recording's end");
    }
    return on;
}

static uint64_t count_draws(const uint8_t *p) {
    uint64_t n = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.op == OP_END || h.size < 8) return n;
        n += h.op == OP_DRAW || h.op == OP_DRAW_IDX || h.op == OP_DRAW_IDX_IND || h.op == OP_DRAW_MESH ||
             h.op == OP_DISPATCH || h.op == OP_DISPATCH_IND;
        p += h.size;
    }
}

static void add_entry(RcInfo *a, uint64_t end, uint64_t draws) {
    if (a->n == RC_E) {   // merge the two nearest ends
        int best = 0;
        uint64_t gap = UINT64_MAX;
        for (int i = 0; i < a->n; i++)
            for (int j = i + 1; j < a->n; j++) {
                uint64_t g = a->e[i].end > a->e[j].end ? a->e[i].end - a->e[j].end : a->e[j].end - a->e[i].end;
                if (g < gap) gap = g, best = i * RC_E + j;
            }
        int i = best / RC_E, j = best % RC_E;
        a->e[i].end = a->e[i].end > a->e[j].end ? a->e[i].end : a->e[j].end;
        a->e[i].draws += a->e[j].draws;
        a->e[j] = a->e[--a->n];
    }
    a->e[a->n].end = end, a->e[a->n++].draws = draws;
}

static void merge(RcInfo *a, const RcInfo *b) {
    if (!b->draws) return;
    a->lo = a->draws && a->lo < b->lo ? a->lo : b->lo;
    a->hi = a->draws && a->hi > b->hi ? a->hi : b->hi;
    a->draws += b->draws, a->sum += b->sum;
    for (int i = 0; i < b->n; i++) add_entry(a, b->e[i].end, b->e[i].draws);
}

void nvn_rc_begin(const void *cmd) {
    if (!nvn_rc_on()) return;
    uint64_t k = (uintptr_t)cmd;
    plat_mutex_lock(&rc_lock);
    free(nvn_hm_del(&by_cmd, &k));
    plat_mutex_unlock(&rc_lock);
}

void nvn_rc_inline(const void *cmd, uint64_t handle) {
    if (!nvn_rc_on()) return;
    uint64_t k = (uintptr_t)cmd;
    plat_mutex_lock(&rc_lock);
    const RcInfo *leaf = nvn_hm_get(&by_handle, &handle);
    if (leaf) {
        RcInfo *acc = nvn_hm_get(&by_cmd, &k);
        if (!acc && (acc = calloc(1, sizeof *acc))) nvn_hm_put(&by_cmd, &k, acc);
        if (acc) merge(acc, leaf);
    }
    plat_mutex_unlock(&rc_lock);
}

void nvn_rc_end(const void *cmd, const uint8_t *copy) {
    if (!nvn_rc_on()) return;
    uint64_t k = (uintptr_t)cmd, handle = (uintptr_t)copy, now = plat_now_ns(), total = count_draws(copy);
    RcInfo *info = calloc(1, sizeof *info);
    if (!info) return;
    plat_mutex_lock(&rc_lock);
    RcInfo *acc = nvn_hm_del(&by_cmd, &k);
    uint64_t called = acc ? acc->draws : 0;
    if (acc) *info = *acc, free(acc);
    RcInfo own = {total > called ? total - called : 0, 0, now, now, 0, {{0, 0}}};
    own.sum = own.draws * now;
    if (own.draws) add_entry(&own, now, own.draws);
    merge(info, &own);
    if (by_handle.n > 200000) {   // freed copies' entries: start over rather than grow without bound
        for (uint32_t i = 0; i < by_handle.cap; i++) free(by_handle.vals[i]);
        free(by_handle.keys), free(by_handle.vals);
        nvn_hm_init(&by_handle, 8);
    }
    free(nvn_hm_put(&by_handle, &handle, info));
    plat_mutex_unlock(&rc_lock);
}

void nvn_rc_def(uint64_t reads, uint64_t changed) {   // a submit's deferred binds: read, and read with new handles
    if (!nvn_rc_on()) return;
    plat_mutex_lock(&rc_lock);
    w.def_reads += reads, w.def_changed += changed;
    plat_mutex_unlock(&rc_lock);
}

void nvn_rc_submit(const uint64_t *handles, int n) {
    if (!nvn_rc_on()) return;
    uint64_t now = plat_now_ns();
    plat_mutex_lock(&rc_lock);
    for (int i = 0; i < n; i++) {
        const RcInfo *r = nvn_hm_get(&by_handle, &handles[i]);
        if (!r || !r->draws) continue;
        w.dsubs++, w.draws += r->draws, w.lead += r->draws * now - r->sum, w.first += now - r->lo, w.last += now - r->hi;
        w.recs += (uint64_t)r->n;
        for (int j = 0; j < r->n; j++) {
            uint64_t age = now - r->e[j].end;
            w.by_age[age < 40000000 ? 0 : age < 1000000000 ? 1 : 2] += r->e[j].draws;
        }
    }
    if (++w.submits == 1500) {
        double s = (double)w.submits, d = w.dsubs ? (double)w.dsubs : 1;
        rt_log("nvn: recording census over %llu submits (%llu with draws): %.0f draws a submit with draws, their "
               "recordings ended %.2f ms before the submit (draw-weighted), the earliest %.2f ms and the latest %.2f ms "
               "before (per submit with draws); by the age of their recording %.0f %% under 40 ms, %.0f %% 40 ms-1 s, "
               "%.0f %% over 1 s, from %.1f recordings a submit; deferred binds %.1f a submit, %.2f with new handles",
               (unsigned long long)w.submits, (unsigned long long)w.dsubs, (double)w.draws / d,
               w.draws ? (double)w.lead / (double)w.draws / 1e6 : 0, (double)w.first / d / 1e6, (double)w.last / d / 1e6,
               w.draws ? 100.0 * (double)w.by_age[0] / (double)w.draws : 0,
               w.draws ? 100.0 * (double)w.by_age[1] / (double)w.draws : 0,
               w.draws ? 100.0 * (double)w.by_age[2] / (double)w.draws : 0, (double)w.recs / d, (double)w.def_reads / s,
               (double)w.def_changed / s);
        memset(&w, 0, sizeof w);
    }
    plat_mutex_unlock(&rc_lock);
}
