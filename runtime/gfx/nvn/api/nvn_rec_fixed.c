// MKW_REC_CENSUS=fixed (diagnosis, default off): what the fixed recordings hold, the ones the game calls again and
// again (CallCommands / CopyCommands of a handle first called over 1 s before). In a split-screen race 71-72 % of the
// draws of the big submits come from them; translating each once and reusing its Vulkan packets
// is only possible for what a recording sets itself before its first draw
// and for recordings without work the replay must redo or order every time. A copy never changes (a new recording is
// a new handle, nvn_api_cmd.c), so each handle is walked once, at its first call.
// One line per fixed handle, the first 40: its draws and dispatches, what it sets before its first draw, what it holds;
// the same census for new recordings (their first call: the pieces the game's threads record during the frame);
// and every 1500 submits the draws of fixed calls by those traits.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_cmd_internal.h"
#include "nvn_hmap.h"
#include "nvn_obj.h"
#include "platform/platform.h"

// Traits (bits): set before the first draw or dispatch, then held anywhere in the recording.
enum {
    F_PROG, F_RT, F_VIEW, F_STATE, F_VTX, F_UBO, F_SSBO, F_TEX,   // set before the first draw
    F_BARRIER, F_QUERY, F_COPY, F_EVENT, F_CLEAR, F_RTMID, F_PUSH, F_DEFER, F_N
};
static const char *const f_name[F_N] = {"program", "targets", "viewport", "fixed state", "vertex input", "uniform blocks",
                                        "storage buffers", "textures", "barrier", "query", "copy or update", "event",
                                        "clear", "targets after a draw", "push constants", "deferred binds"};

typedef struct { uint64_t first_ns, draws; uint32_t traits; } Fixed;
static nvn_hmap seen = {.klen = 8};   // handle -> Fixed (handles of freed copies stay until reused: diagnosis)
static PlatMutex fx_lock = PLAT_MUTEX_INIT;
// Per class: 0 fixed (called again over 1 s after its first call), 1 new (its first call: the frame's own recordings,
// the pieces the game's threads record during the frame and copy into the frame's submit).
static uint64_t w_calls[2], w_draws[2], w_by[2][F_N], w_clean[2], w_subs;
static int nlogged;

int nvn_rf_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = cfg_get("MKW_REC_CENSUS");   // a diagnosis: cfg_get, no cfg.c row
        on = v && !strcmp(v, "fixed");
        if (on) rt_log("nvn: fixed-recording census on (MKW_REC_CENSUS=fixed)");
    }
    return on;
}

static void profile(const uint8_t *p, Fixed *f) {
    int drew = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.op == OP_END || h.size < 8) return;
        uint32_t t = 0;
        switch (h.op) {
        case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: case OP_DRAW_MESH: case OP_DISPATCH: case OP_DISPATCH_IND:
            f->draws++, drew = 1;
            break;
        case OP_BIND_PROGRAM: t = 1u << F_PROG; break;
        case OP_SET_RT: t = drew ? 1u << F_RTMID : 1u << F_RT; break;
        case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: t = 1u << F_VIEW; break;
        case OP_BIND_STATE: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR: case OP_DEPTH_CLAMP:
        case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_ALPHA_REF:
            t = 1u << F_STATE;
            break;
        case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: t = 1u << F_VTX; break;
        case OP_BIND_UBO: t = 1u << F_UBO; break;
        case OP_BIND_SSBO: t = 1u << F_SSBO; break;
        case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: {
            const p_bind_handles *b = (const p_bind_handles *)(const void *)(p + 8);
            t = 1u << F_TEX;
            if ((h.op == OP_BIND_TEX || h.op == OP_BIND_IMG || h.op == OP_BIND_SAMP) && b->src) t |= 1u << F_DEFER;
            break;
        }
        case OP_BARRIER: t = 1u << F_BARRIER; break;
        case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_COND: case OP_RENDER_ENABLE: t = 1u << F_QUERY; break;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_BUF: case OP_UPDATE_UB:
            t = 1u << F_COPY;
            break;
        case OP_SIGNAL_EVENT: case OP_WAIT_EVENT: t = 1u << F_EVENT; break;
        case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_CLEAR_TEX: t = 1u << F_CLEAR; break;
        case OP_PUSH_CONST: t = 1u << F_PUSH; break;
        default: break;
        }
        // the first eight traits count only before the first draw: after it they are the recording's own changes
        if (drew) t &= ~((1u << F_BARRIER) - 1) | (1u << F_RTMID);
        f->traits |= t;
        p += h.size;
    }
}

void nvn_rf_call(uint64_t handle) {
    if (!nvn_rf_on() || !handle) return;
    uint64_t now = plat_now_ns();
    plat_mutex_lock(&fx_lock);
    Fixed *f = nvn_hm_get(&seen, &handle);
    int cls = -1;
    if (!f && (f = calloc(1, sizeof *f))) {
        f->first_ns = now;
        profile((const uint8_t *)(uintptr_t)handle, f);
        nvn_hm_put(&seen, &handle, f);
        cls = 1;
    }
    if (f && cls < 0 && now - f->first_ns > 1000000000ull) cls = 0;
    if (cls >= 0 && f->draws) {
        w_calls[cls]++, w_draws[cls] += f->draws;
        for (int i = 0; i < F_N; i++)
            if (f->traits >> i & 1) w_by[cls][i] += f->draws;
        uint32_t bad = 1u << F_BARRIER | 1u << F_QUERY | 1u << F_COPY | 1u << F_EVENT | 1u << F_DEFER;
        if ((f->traits & (1u << F_PROG)) && !(f->traits & bad)) w_clean[cls] += f->draws;
        if (!cls && nlogged < 40 && f->first_ns) {
            char b[512];
            int k = 0;
            for (int i = 0; i < F_N && k < (int)sizeof b - 32; i++)
                if (f->traits >> i & 1) k += snprintf(b + k, sizeof b - (size_t)k, "%s%s", k ? ", " : "", f_name[i]);
            b[k] = 0;
            rt_log("nvn: fixed recording %#llx: %llu draws and dispatches; %s", (unsigned long long)handle,
                   (unsigned long long)f->draws, k ? b : "sets nothing before its first draw and holds nothing else");
            nlogged++, f->first_ns = 0;   // logged once (first_ns 0 keeps it counted as fixed)
        }
    }
    plat_mutex_unlock(&fx_lock);
}

void nvn_rf_submit(void) {
    if (!nvn_rf_on()) return;
    plat_mutex_lock(&fx_lock);
    if (++w_subs == 1500) {
        for (int c = 0; c < 2; c++) {
            double d = w_draws[c] ? 100.0 / (double)w_draws[c] : 0;
            char b[768];
            int k = 0;
            for (int i = 0; i < F_N && k < (int)sizeof b - 48; i++)
                k += snprintf(b + k, sizeof b - (size_t)k, "; %s %.0f %%", f_name[i], (double)w_by[c][i] * d);
            rt_log("nvn: %s-recording census over 1500 submits: %llu calls, %llu draws; with a program and none of "
                   "barrier, query, copy, event, deferred binds %.0f %% of the draws; by trait (set before the first draw "
                   "for the first eight)%s", c ? "new" : "fixed", (unsigned long long)w_calls[c],
                   (unsigned long long)w_draws[c], (double)w_clean[c] * d, b);
        }
        w_subs = 0;
        memset(w_calls, 0, sizeof w_calls), memset(w_draws, 0, sizeof w_draws), memset(w_clean, 0, sizeof w_clean);
        memset(w_by, 0, sizeof w_by);
    }
    plat_mutex_unlock(&fx_lock);
}
