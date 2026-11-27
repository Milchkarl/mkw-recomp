// MKW_REC_CENSUS=seq (diagnosis, default off): how much of a frame's big submit repeats the same submit of the frame
// before, op for op. The game builds that submit just before sending it (~1.2 ms of CallCommands into small recordings
// it keeps for seconds), so its replay cannot start earlier from the game's own recording. If the frames repeat each
// other, the replay could translate frame N+1 from frame N's stream while the game is still computing N+1, and at the
// submit only redo what differs: this census says how much would differ.
//
// Each op of a submit (as resolve_deferred in nvn_api_cmd.c walks it: called recordings inline, the ops the replay
// hands to vk_exec_op, so the count matches its op positions) gets two hashes: the op with its payload (exact) and its
// shape (below). A submit of over BIG ops is aligned with earlier ones: each op matches the next equal one
// within LOOK ops ahead in the previous stream, else it is new there; against the frame before and the best of the last
// HIST (rotating buffers), exact and with buffer addresses and texture handles taken as patchable. Every LOG_EVERY big
// submits: ops, the shares that match (all ops and draws), and the kinds of the ops that still differ.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_cmd_internal.h"

#define BIG 20000
#define LOOK 256
#define LOG_EVERY 120

#define HIST 3

// Per op: exact hash (op and payload), shape hash (buffer binds without their address, texture binds without their
// handles: what a translation made ahead could patch at the submit; every other op exact) and the op.
typedef struct { uint64_t *x, *s; uint32_t *k; int n, cap; } Stream;
static Stream cur, hist[HIST];   // game thread, under the front end's submit lock; hist[0] the frame before
// [0] exact vs the frame before, [1] exact vs the best of the last HIST, [2] shape vs the frame before, [3] shape best
static uint64_t s_subs, s_ops, s_draws, s_eq[4], s_deq[4], s_diff_by[OP_COUNT_], s_any[2], s_dany[2], s_ord, s_dord;
static uint64_t *srt;   // a sorted copy of the frame before's hashes
static int srt_cap;

static int on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = cfg_get("MKW_REC_CENSUS");   // a diagnosis: cfg_get, no cfg.c row
        v = e && !strcmp(e, "seq");
        if (v) rt_log("nvn: big-submit repetition census on (MKW_REC_CENSUS=seq)");
    }
    return v;
}

static int is_draw(uint32_t op) {
    return op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND || op == OP_DRAW_MESH || op == OP_DISPATCH ||
           op == OP_DISPATCH_IND;
}

static uint64_t mix(uint64_t x, const uint8_t *p, uint32_t n) {
    for (uint32_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        x = (x ^ w) * 1099511628211ull;
        x ^= x >> 29;
    }
    return x;
}

void nvn_rs_op(const uint8_t *p) {
    if (!on()) return;
    op_hdr h;
    memcpy(&h, p, sizeof h);
    if (h.op == OP_BEGIN || h.op == OP_END || h.op == OP_JUMP || h.op == OP_CALL) return;
    const uint8_t *a = p + 8;
    uint32_t n = h.size - 8;
    uint64_t x = mix(1469598103934665603ull ^ h.op, a, n), s = x;
    if (h.op == OP_BIND_UBO || h.op == OP_BIND_SSBO) {   // stage, index and size: the address patched
        const p_bind_buf *b = (const p_bind_buf *)(const void *)a;
        uint64_t f[2] = {(uint64_t)(uint32_t)b->stage << 32 | (uint32_t)b->index, b->size};
        s = mix(7 ^ h.op, (const uint8_t *)f, sizeof f);
    } else if (h.op == OP_BIND_TEX || h.op == OP_BIND_SAMP || h.op == OP_BIND_IMG) {   // stage, slots: the handles patched
        const p_bind_handles *b = (const p_bind_handles *)(const void *)a;
        uint64_t f[2] = {(uint64_t)(uint32_t)b->stage << 32 | (uint32_t)b->first, (uint64_t)(uint32_t)b->n};
        s = mix(7 ^ h.op, (const uint8_t *)f, sizeof f);
    }
    if (cur.n == cur.cap) {
        cur.cap = cur.cap ? cur.cap * 2 : 1 << 16;
        if (!(cur.x = realloc(cur.x, sizeof *cur.x * (size_t)cur.cap)) ||
            !(cur.s = realloc(cur.s, sizeof *cur.s * (size_t)cur.cap)) ||
            !(cur.k = realloc(cur.k, sizeof *cur.k * (size_t)cur.cap)))
            rt_fatal("nvn: out of memory (MKW_REC_CENSUS=seq)");
    }
    cur.x[cur.n] = x, cur.s[cur.n] = s, cur.k[cur.n] = h.op, cur.n++;
}

// Ops of a matched in b, aligned greedily: each op of a takes the next equal one of b within LOOK ahead. Returns the
// ops matched (*draws: the draws among them); diff, if set, counts the unmatched by kind.
static uint64_t align(const uint64_t *a, int na, const uint64_t *b, int nb, const uint32_t *kind, uint64_t *draws,
                      uint64_t *diff) {
    uint64_t eq = 0;
    *draws = 0;
    for (int i = 0, j = 0; i < na; i++) {
        int hit = -1;
        for (int d = 0; d < LOOK && j + d < nb; d++)
            if (b[j + d] == a[i]) { hit = j + d; break; }
        if (hit >= 0) j = hit + 1, eq++, *draws += (uint64_t)is_draw(kind[i]);
        else if (diff) diff[kind[i] < OP_COUNT_ ? kind[i] : 0]++;
    }
    return eq;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

// Order-free bound: the ops of cur found anywhere in the frame before (w 0 exact, 1 shape).
static void anywhere(int w) {
    const Stream *p = &hist[0];
    if (p->n > srt_cap && !(srt = realloc(srt, sizeof *srt * (size_t)(srt_cap = p->n))))
        rt_fatal("nvn: out of memory (MKW_REC_CENSUS=seq)");
    memcpy(srt, w ? p->s : p->x, sizeof *srt * (size_t)p->n);
    qsort(srt, (size_t)p->n, sizeof *srt, cmp_u64);
    for (int i = 0; i < cur.n; i++) {
        uint64_t v = w ? cur.s[i] : cur.x[i];
        if (bsearch(&v, srt, (size_t)p->n, sizeof *srt, cmp_u64)) s_any[w]++, s_dany[w] += (uint64_t)is_draw(cur.k[i]);
    }
}

// Ordered bound (patience diff): ops whose exact hash is unique in both frames are anchors; the longest run of anchors in
// the same order (LIS) keeps the order, and between two kept anchors the ops count as the same in order when both
// gaps hold the same hashes one for one. Ops of cur in that ordered match.
typedef struct { uint64_t h; int i; } HI;
static HI *ha, *hb;
static int hcap, *lis_b, *lis_p, *lis_t, *pos;
static int cmp_hi(const void *a, const void *b) {
    const HI *x = a, *y = b;
    return x->h < y->h ? -1 : x->h > y->h ? 1 : x->i - y->i;
}
static int uniq(HI *v, int n, int k) { return (k == 0 || v[k - 1].h != v[k].h) && (k == n - 1 || v[k + 1].h != v[k].h); }

static uint64_t ordered(const uint64_t *a, int na, const uint64_t *b, int nb, const uint32_t *kind, uint64_t *draws) {
    int m = na > nb ? na : nb;
    if (m > hcap) {
        hcap = m;
        if (!(ha = realloc(ha, sizeof *ha * (size_t)m)) || !(hb = realloc(hb, sizeof *hb * (size_t)m)) ||
            !(lis_b = realloc(lis_b, sizeof(int) * (size_t)m)) || !(lis_p = realloc(lis_p, sizeof(int) * (size_t)m)) ||
            !(lis_t = realloc(lis_t, sizeof(int) * (size_t)m)) || !(pos = realloc(pos, sizeof(int) * (size_t)m)))
            rt_fatal("nvn: out of memory (MKW_REC_CENSUS=seq)");
    }
    for (int i = 0; i < na; i++) ha[i] = (HI){a[i], i};
    for (int i = 0; i < nb; i++) hb[i] = (HI){b[i], i};
    qsort(ha, (size_t)na, sizeof *ha, cmp_hi), qsort(hb, (size_t)nb, sizeof *hb, cmp_hi);
    for (int i = 0; i < na; i++) pos[i] = -1;   // a's op -> b's op, for anchors
    for (int x = 0, y = 0; x < na && y < nb;) {
        if (ha[x].h < hb[y].h) x++;
        else if (ha[x].h > hb[y].h) y++;
        else {
            if (uniq(ha, na, x) && uniq(hb, nb, y)) pos[ha[x].i] = hb[y].i;
            x++, y++;
        }
    }
    int len = 0;   // LIS over pos in a's order: lis_t[k] the a-index ending the best run of length k+1
    for (int i = 0; i < na; i++) {
        if (pos[i] < 0) continue;
        int lo = 0, hi = len;
        while (lo < hi) { int md = (lo + hi) / 2; if (pos[lis_t[md]] < pos[i]) lo = md + 1; else hi = md; }
        lis_p[i] = lo ? lis_t[lo - 1] : -1, lis_t[lo] = i;
        if (lo == len) len++;
    }
    int nk = 0;
    for (int i = len ? lis_t[len - 1] : -1; i >= 0; i = lis_p[i]) lis_b[nk++] = i;   // kept anchors, last first
    uint64_t eq = 0;
    *draws = 0;
    int pa = -1, pb = -1;
    for (int k = nk; k >= 0; k--) {   // each gap before an anchor (and the tail after the last)
        int ea = k ? lis_b[k - 1] : na, eb = k ? pos[lis_b[k - 1]] : nb;
        if (ea - pa == eb - pb) {
            for (int g = 1; g < ea - pa; g++)
                if (a[pa + g] == b[pb + g]) eq++, *draws += (uint64_t)is_draw(kind[pa + g]);
        }
        if (k) eq++, *draws += (uint64_t)is_draw(kind[ea]);
        pa = ea, pb = eb;
    }
    return eq;
}

static void census_log(void) {
    int ord[OP_COUNT_];
    for (int i = 0; i < OP_COUNT_; i++) ord[i] = i;
    for (int i = 1; i < OP_COUNT_; i++)
        for (int j = i; j > 0 && s_diff_by[ord[j]] > s_diff_by[ord[j - 1]]; j--) { int t = ord[j]; ord[j] = ord[j - 1], ord[j - 1] = t; }
    char b[512];
    int k = 0;
    uint64_t nd = s_ops - s_eq[2];
    for (int i = 0; i < 8 && s_diff_by[ord[i]] && k < (int)sizeof b - 48; i++)
        k += snprintf(b + k, sizeof b - (size_t)k, "%s%s %.0f %%", i ? ", " : "", nvn_op_name((uint32_t)ord[i]),
                      100.0 * (double)s_diff_by[ord[i]] / (double)(nd ? nd : 1));
    b[k] = 0;
    double o = 100.0 / (double)s_ops, d = 100.0 / (double)(s_draws ? s_draws : 1);
    rt_log("nvn: big submits repeating earlier frames, over %llu (%.0f ops, %.0f draws each): exact, the frame before "
           "%.1f %% of ops (%.1f %% of draws), the best of the last %d %.1f %% (%.1f %%); with buffer addresses and "
           "texture handles patched, the frame before %.1f %% (%.1f %%), the best of the last %d %.1f %% (%.1f %%); "
           "what still differs from the frame before by kind: %s; anywhere in the frame before (order-free bound): "
           "exact %.1f %% (%.1f %% of draws), patched %.1f %% (%.1f %%); in the same order (anchored on unique ops, "
           "exact) %.1f %% (%.1f %% of draws)",
           (unsigned long long)s_subs, (double)s_ops / (double)s_subs, (double)s_draws / (double)s_subs,
           (double)s_eq[0] * o, (double)s_deq[0] * d, HIST, (double)s_eq[1] * o, (double)s_deq[1] * d,
           (double)s_eq[2] * o, (double)s_deq[2] * d, HIST, (double)s_eq[3] * o, (double)s_deq[3] * d, k ? b : "none",
           (double)s_any[0] * o, (double)s_dany[0] * d, (double)s_any[1] * o, (double)s_dany[1] * d,
           (double)s_ord * o, (double)s_dord * d);
    s_subs = s_ops = s_draws = s_ord = s_dord = 0;
    memset(s_any, 0, sizeof s_any), memset(s_dany, 0, sizeof s_dany);
    memset(s_eq, 0, sizeof s_eq), memset(s_deq, 0, sizeof s_deq), memset(s_diff_by, 0, sizeof s_diff_by);
}

void nvn_rs_end(void) {
    if (!on()) return;
    if (cur.n < BIG) { cur.n = 0; return; }
    if (hist[HIST - 1].n >= BIG) {
        s_subs++, s_ops += (uint64_t)cur.n;
        for (int i = 0; i < cur.n; i++) s_draws += (uint64_t)is_draw(cur.k[i]);
        uint64_t best[2] = {0, 0}, bestd[2] = {0, 0};
        for (int h = 0; h < HIST; h++)
            for (int w = 0; w < 2; w++) {
                uint64_t dr, eq = align(w ? cur.s : cur.x, cur.n, w ? hist[h].s : hist[h].x, hist[h].n, cur.k, &dr,
                                        !h && w ? s_diff_by : NULL);
                if (!h) s_eq[2 * w] += eq, s_deq[2 * w] += dr;
                if (eq > best[w]) best[w] = eq, bestd[w] = dr;
            }
        s_eq[1] += best[0], s_deq[1] += bestd[0], s_eq[3] += best[1], s_deq[3] += bestd[1];
        anywhere(0), anywhere(1);
        uint64_t od;
        s_ord += ordered(cur.x, cur.n, hist[0].x, hist[0].n, cur.k, &od), s_dord += od;
        if (s_subs == LOG_EVERY) census_log();
    }
    Stream t = hist[HIST - 1];
    memmove(hist + 1, hist, sizeof hist[0] * (HIST - 1));
    hist[0] = cur, cur = t, cur.n = 0;
}
