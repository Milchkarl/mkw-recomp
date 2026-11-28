// MKW_SEG=1 (default off; "check": every hit compared with the same draws recorded as usual): inside the passes a chunk
// records (MKW_REPLAY_PAR, in secondaries as with MKW_SEC_DRAWS), stretches of up to SEG_DRAWS draws are recorded once
// into a secondary command buffer of their own and executed again by later submissions whose stretch repeats.
//
// Why: replaying a draw costs the chunk threads ~2.7 us, about half of it what such a stretch repeats from two frames
// before (pipeline, dynamic state, vertex and index buffers, the draw), the rest the per-draw data (sim2-stage). In a
// 2-player race 88.7 % of the draws lie in 16-draw segments found whole two frames back once the data the bindless rows
// carry is left out (sim2-keyed). Step b of pantalla-partida; with MKW_BINDLESS=1 and programs of SM86_BINDLESS=2.
//
// How:
// - A segment begins at a draw inside a pass recorded in secondaries and holds the state and bind ops up to its
//   SEG_DRAWS-th draw (vk_exec_par.c seg_gather). Its key hashes those ops without the addresses, handles and pushed
//   bytes the rows carry, and the replay state they start from (targets, pass formats, fixed state, program, vertex
//   buffers): equal keys record equal commands but for the rows.
// - Miss: the draws are recorded as usual into a new entry's secondary (no ONE_TIME_SUBMIT), the bindless rows into the
//   entry's own buffer so the pushed row addresses stay valid, the support buffer into a persistent copy. A draw that
//   records what a later submission could not repeat (a barrier, a pass end, scratch indices, descriptors other than
//   the rows, the temporal jitter) or is skipped makes the capture bad: the secondary then goes on as a plain one.
// - Hit: an entry of the same key the GPU is done with. Its ops are applied to the replay state as the light walk does,
//   each draw's side effects done and its rows written (vk_exec_seg_fill), and the secondary executed. A draw that
//   would now record otherwise (skipped, a feedback barrier, another pipeline, another vertex buffer) undoes it all and
//   the stretch is recorded as usual.
// Entries live for the run in a table of SEG_MAX, the least recently used taken back once the GPU is done with them.
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_draw_internal.h"
#include "vk_exec.h"
#include "vk_passhz.h"
#include "vk_reuse.h"
#include "vk_seg.h"
#define XXH_INLINE_ALL
#include <xxhash.h>

#define SEG_MAX 4096
#define SEG_TAB 4096          // hash buckets, power of two
#define SEG_SPANS 48          // vertex and index buffers a segment binds
#define SEG_IDLE_FENCES 16    // an entry unused for this many submissions may be taken back (a race frame sends ~4)
#define SUP_SLOTS 1024
#define SEEN_TAB 65536        // keys seen once (admission), power of two
#define SUP_SLOT 1536         // SUPPORT_BYTES rounded up to 256

typedef struct { uint64_t addr; VkMemSpan sp; VkBuffer rec; } SegSpan;
enum { SE_FREE, SE_CAP, SE_LIVE };
typedef struct SegEnt {
    uint64_t key, fence, used, pgen;   // fence: last submission it was executed in; pgen: vk_mem_pool_gen at capture
    struct SegEnt *next;               // bucket chain (live) or free list
    int state, ndraws, nspan;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkBuffer rbuf;
    VmaAllocation ralloc;
    uint8_t *rmap;
    VkDeviceAddress raddr;
    SegDraw d[SEG_DRAWS];
    SegSpan span[SEG_SPANS];
} SegEnt;

struct SegCap {
    SegEnt *e;
    int bad, nd, ndraws, rows;
    uint16_t pend_row[2];
    uint8_t pend_span[2];
};

static int seg_mode = -1;
static struct {
    PlatMutex lock;
    SegEnt *tab[SEG_TAB], *free, *ent;
    int nent, hand;
    uint64_t n_try, n_hit, n_cap, n_pub, n_undo, n_kill, n_full, n_chk, n_chk_bad, n_first, draws_hit, bad[SEGB_COUNT];
} C = {.lock = PLAT_MUTEX_INIT};
static struct {
    PlatMutex lock;
    VkBuffer buf;
    uint8_t *map;
    int n;
    uint64_t hash[SUP_SLOTS];
} SUP = {.lock = PLAT_MUTEX_INIT};

_Thread_local SegCap *seg_cap;
static _Thread_local SegCap cap_tls;
static _Thread_local struct { int on; VkExec filled; } CHK;   // check: the hit's state, compared at vk_exec_seg_end
static _Thread_local VkExec snap;

int vk_exec_seg_mode(void) {
    if (seg_mode < 0) {
        const char *e = cfg_get("MKW_SEG"), *cr = cfg_get("MKW_GPU_CRUMBS_DRAW");
        int m = e && !strcmp(e, "check") ? 2 : e && atoi(e) > 0;
        if (m && !vk_bindless_on()) rt_log("vk: MKW_SEG needs MKW_BINDLESS=1: off"), m = 0;
        if (m && cr && atoi(cr)) rt_log("vk: MKW_SEG is off with MKW_GPU_CRUMBS_DRAW (marks per draw)"), m = 0;
        if (m) rt_log("vk: MKW_SEG: chunk draws in reusable segments of %d%s", SEG_DRAWS, m == 2 ? ", each hit checked" : "");
        __atomic_store_n(&seg_mode, m, __ATOMIC_RELEASE);
    }
    return seg_mode;
}

int vk_exec_seg_op(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX:
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
    case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_PUSH_CONST:
    case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR:
    case OP_ALPHA_REF: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_COPY_STRIDE:
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
        return 1;
    default:
        return 0;
    }
}

// Miss census (28/11/2026: 7 % hits on race2pl where sim2-keyed found 88.7 %): per miss, which part of the key changed
// against the last segment with the same ops, and how many misses had their key live but still in use by the GPU.
enum { KP_RT, KP_PASS, KP_FIXED, KP_PIPE, KP_PROG, KP_VBUF, KP_REST, KP_N };
static const char *const kp_name[KP_N] = {"rt", "pass", "fixed", "pipe", "prog", "vbuf", "rest"};
#define KD_TAB 65536
static struct { uint64_t ops, part[KP_N]; } KD[KD_TAB];
static uint64_t kd_diff[KP_N], kd_new, kd_same, kd_busy;
static _Thread_local uint64_t kd_ops, kd_part[KP_N];
static _Thread_local const VkExec *kd_ex;

static uint64_t h64(const void *p, size_t n) { return XXH3_64bits(p, n); }

static void key_parts(const VkExec *ex, const op_hdr *const *ops, int nops) {
    kd_ex = ex;
    kd_part[KP_RT] = h64(&ex->rt, sizeof ex->rt);
    kd_part[KP_PASS] = h64(&ex->pass_w, offsetof(VkExec, vp) - offsetof(VkExec, pass_w));
    kd_part[KP_FIXED] = h64(&ex->vp, offsetof(VkExec, pipe) - offsetof(VkExec, vp));
    kd_part[KP_PIPE] = h64(&ex->pipe, sizeof ex->pipe);
    kd_part[KP_PROG] = h64(&ex->prog, offsetof(VkExec, vbuf) - offsetof(VkExec, prog));
    kd_part[KP_VBUF] = h64(&ex->vbuf, sizeof ex->vbuf);
    uint64_t r[2] = {(uint64_t)ex->render_disabled, (uint64_t)ex->fbl_mask};
    kd_part[KP_REST] = h64(r, sizeof r);
    XXH3_state_t st;
    XXH3_64bits_reset(&st);
    for (int i = 0; i < nops; i++) {
        const op_hdr *h = ops[i];
        size_t n = h->size - sizeof *h;
        if (h->op == OP_BIND_UBO || h->op == OP_BIND_SSBO) n = 8;
        else if (h->op == OP_BIND_TEX || h->op == OP_BIND_SAMP || h->op == OP_PUSH_CONST) n = 12;
        XXH3_64bits_update(&st, h, sizeof *h);
        XXH3_64bits_update(&st, nvn_op_payload(h), n);
    }
    kd_ops = XXH3_64bits_digest(&st);
}

// Which field of the fixed state and of the pipeline state changed (byte census against the last segment of the same
// ops): the fields by name, from VkExec vp..pipe and VkPipeState.
#define FD(s, f) {#f, offsetof(s, f), sizeof(((s *)0)->f)}
typedef struct { const char *name; size_t off, size; } FieldDef;
static const FieldDef fd_fixed[] = {
    FD(VkExec, vp), FD(VkExec, has_vp), FD(VkExec, scissor), FD(VkExec, has_scissor), FD(VkExec, depth_range),
    FD(VkExec, st_vmask), FD(VkExec, st_wmask), FD(VkExec, st_ref), FD(VkExec, point_size), FD(VkExec, poly_offset),
    FD(VkExec, prim_restart), FD(VkExec, blend_color), FD(VkExec, alpha_ref), FD(VkExec, copy_stride),
    FD(VkExec, tex_pool), FD(VkExec, samp_pool)};
static const FieldDef fd_pipe[] = {
    FD(VkPipeState, vs), FD(VkPipeState, fs), FD(VkPipeState, color_fmt), FD(VkPipeState, depth_fmt),
    FD(VkPipeState, samples), FD(VkPipeState, prim), FD(VkPipeState, depth_minus_w), FD(VkPipeState, depth_clamp),
    FD(VkPipeState, raster_discard), FD(VkPipeState, y_flip), FD(VkPipeState, prim_restart), FD(VkPipeState, feedback),
    FD(VkPipeState, blend), FD(VkPipeState, color), FD(VkPipeState, chmask), FD(VkPipeState, ms), FD(VkPipeState, poly),
    FD(VkPipeState, ds), FD(VkPipeState, vattr), FD(VkPipeState, vstream), FD(VkPipeState, mesh)};
#define NFIX (sizeof fd_fixed / sizeof *fd_fixed)
#define NPIPE (sizeof fd_pipe / sizeof *fd_pipe)
// Each try against the most alike of the last KDB_WAYS starting states of the same ops (the last one alone is often
// the other player's view): exact repeats, and the fields that still differ from the most alike.
#define KDB_TAB 8192
#define KDB_WAYS 4
#define FIX_BYTES (offsetof(VkExec, pipe) - offsetof(VkExec, vp))
#define PASS_BYTES (offsetof(VkExec, vp) - offsetof(VkExec, pass_w))
typedef struct { uint8_t pass[PASS_BYTES], fixed[FIX_BYTES]; VkPipeState pipe; NVNbufferRange vbuf[16]; } KdState;
static struct { uint64_t ops[KDB_WAYS]; KdState st[KDB_WAYS]; int next; } KDB[KDB_TAB];
static uint64_t fd_fixed_n[NFIX], fd_pipe_n[NPIPE], fd_pass_n, fd_vbuf_n, fd_exact, fd_alike, fd_tries;

static int state_diff(const KdState *a, const KdState *b, int count) {
    int n = 0;
    for (size_t k = 0; k < NFIX; k++) {
        size_t o = fd_fixed[k].off - offsetof(VkExec, vp);
        if (memcmp(a->fixed + o, b->fixed + o, fd_fixed[k].size)) n++, fd_fixed_n[k] += (uint64_t)count;
    }
    for (size_t k = 0; k < NPIPE; k++)
        if (memcmp((const uint8_t *)&a->pipe + fd_pipe[k].off, (const uint8_t *)&b->pipe + fd_pipe[k].off, fd_pipe[k].size))
            n++, fd_pipe_n[k] += (uint64_t)count;
    if (memcmp(a->pass, b->pass, sizeof a->pass)) n++, fd_pass_n += (uint64_t)count;
    if (memcmp(a->vbuf, b->vbuf, sizeof a->vbuf)) n++, fd_vbuf_n += (uint64_t)count;
    return n;
}

static void field_census(void) {   // C.lock held, every try
    uint32_t i = (uint32_t)(kd_ops >> 16) & (KDB_TAB - 1);
    KdState cur;
    memcpy(cur.pass, &kd_ex->pass_w, sizeof cur.pass);
    memcpy(cur.fixed, &kd_ex->vp, sizeof cur.fixed);
    cur.pipe = kd_ex->pipe;
    memcpy(cur.vbuf, kd_ex->vbuf, sizeof cur.vbuf);
    fd_tries++;
    int best = -1, bn = 1 << 30;
    for (int w = 0; w < KDB_WAYS; w++)
        if (KDB[i].ops[w] == kd_ops) {
            int n = state_diff(&KDB[i].st[w], &cur, 0);
            if (n < bn) bn = n, best = w;
        }
    if (best >= 0 && bn == 0) fd_exact++;
    else if (best >= 0) fd_alike++, state_diff(&KDB[i].st[best], &cur, 1);
    if (best < 0 || bn) {
        int w = KDB[i].next;
        KDB[i].next = (w + 1) % KDB_WAYS;
        KDB[i].ops[w] = kd_ops, KDB[i].st[w] = cur;
    }
}

// Repeat census per key variant (every try): how many tries' key was never seen before in the run, for the key as is
// and without each part of the starting state, so the part that changes every frame shows (seg5, 28/11/2026: 75 % of
// keys first seen while recseq.py --keyed finds 82 % of draws in segments whose ops repeat).
enum { KV_FULL, KV_NO_VBUF, KV_NO_VIEW, KV_NO_PIPE, KV_NO_PROG, KV_OPS, KV_N };
static const char *const kv_name[KV_N] = {"as is", "without vbuf", "without pass and fixed", "without pipe",
                                          "without prog", "ops only"};
static uint64_t kv_seen[KV_N][65536], kv_first[KV_N], kv_tries;

static void variant_census(void) {   // C.lock held
    static const uint8_t drop[KV_N] = {0, 1u << KP_VBUF, (1u << KP_PASS) | (1u << KP_FIXED), 1u << KP_PIPE,
                                       1u << KP_PROG, 0xff & ~(1u << KP_RT)};
    kv_tries++;
    for (int v = 0; v < KV_N; v++) {
        uint64_t h = kd_ops * 0x9E3779B97F4A7C15ull;
        for (int k = 0; k < KP_N; k++)
            if (k != KP_RT && !(drop[v] >> k & 1)) h = (h ^ kd_part[k]) * 0x100000001B3ull;
        uint64_t *s = &kv_seen[v][(h >> 20) & 65535];
        if (*s != h) kv_first[v]++, *s = h;
    }
}

static void key_miss(void) {   // C.lock held
    uint32_t i = (uint32_t)kd_ops & (KD_TAB - 1);
    if (KD[i].ops != kd_ops) {
        kd_new++;
    } else {
        int any = 0;
        for (int k = 0; k < KP_N; k++)
            if (KD[i].part[k] != kd_part[k]) kd_diff[k]++, any = 1;
        if (!any) kd_same++;
    }
    KD[i].ops = kd_ops;
    memcpy(KD[i].part, kd_part, sizeof kd_part);
}

// The ops and the starting state, without what the rows carry (UBO and SSBO addresses, texture and sampler handles,
// pushed bytes) nor what the primary holds (attachment views, the open pass).
static uint64_t seg_key(const VkExec *ex, const op_hdr *const *ops, int nops) {
    XXH3_state_t st;
    XXH3_64bits_reset(&st);
    // Not the targets (ex->rt): a secondary records no attachment, only the formats the pass part below holds; the game
    // alternates its targets between frames (seg2, 28/11/2026: 6034 misses by rt alone). A draw reading a target it
    // writes undoes the hit (feedback_scan in vk_exec_seg_fill).
    XXH3_64bits_update(&st, &ex->pass_w, offsetof(VkExec, vp) - offsetof(VkExec, pass_w));
    XXH3_64bits_update(&st, &ex->vp, offsetof(VkExec, ubo) - offsetof(VkExec, vp));
    XXH3_64bits_update(&st, &ex->render_disabled, sizeof ex->render_disabled);
    XXH3_64bits_update(&st, &ex->fbl_mask, sizeof ex->fbl_mask);
    for (int i = 0; i < nops; i++) {
        const op_hdr *h = ops[i];
        const uint8_t *a = nvn_op_payload(h);
        size_t n = h->size - sizeof *h;
        XXH3_64bits_update(&st, h, sizeof *h);
        if (h->op == OP_BIND_UBO || h->op == OP_BIND_SSBO) n = 8;                            // stage, index
        else if (h->op == OP_BIND_TEX || h->op == OP_BIND_SAMP || h->op == OP_PUSH_CONST) n = 12;   // stage, first/offset, n/size
        XXH3_64bits_update(&st, a, n);
    }
    return XXH3_64bits_digest(&st);
}

static uint64_t gpu_done(void) {
    VkCtx *c = vk_ctx();
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &v);
    return v;
}

static void ent_init(SegEnt *e) {
    VkCtx *c = vk_ctx();
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->gfx_family};
    if (vkCreateCommandPool(c->device, &pi, NULL, &e->pool) != VK_SUCCESS) rt_fatal("vk: MKW_SEG command pool");
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = e->pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &ai, &e->cb) != VK_SUCCESS) rt_fatal("vk: MKW_SEG command buffer");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = SEG_ROW_BYTES,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VmaAllocationCreateInfo ac = {.usage = VMA_MEMORY_USAGE_AUTO, .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo inf;
    if (vmaCreateBuffer(c->vma, &bi, &ac, &e->rbuf, &e->ralloc, &inf) != VK_SUCCESS) rt_fatal("vk: MKW_SEG rows");
    e->rmap = inf.pMappedData;
    e->raddr = vk_buf_addr(e->rbuf);
}

static void unlink_locked(SegEnt *e) {
    SegEnt **p = &C.tab[e->key & (SEG_TAB - 1)];
    while (*p && *p != e) p = &(*p)->next;
    if (*p) *p = e->next;
    e->state = SE_FREE;
    e->next = C.free, C.free = e;
}

// An entry to capture into, claimed for the submission of fence; NULL when every one is in use.
static SegEnt *take_locked(uint64_t fence, uint64_t gpu) {
    for (SegEnt **p = &C.free; *p; p = &(*p)->next)
        if ((*p)->fence <= gpu) {
            SegEnt *e = *p;
            *p = e->next;
            return e;
        }
    if (C.nent < SEG_MAX) {
        if (!C.ent && !(C.ent = calloc(SEG_MAX, sizeof *C.ent))) rt_fatal("vk: out of memory (MKW_SEG)");
        SegEnt *e = &C.ent[C.nent++];
        ent_init(e);
        return e;
    }
    for (int k = 0; k < 64; k++) {   // the clock hand over the live entries
        SegEnt *e = &C.ent[C.hand];
        C.hand = (C.hand + 1) % SEG_MAX;
        if (e->state == SE_LIVE && e->fence <= gpu && e->used + SEG_IDLE_FENCES < fence) {
            unlink_locked(e);
            C.free = e->next;   // (unlink_locked put it at the head of the free list)
            return e;
        }
    }
    return NULL;
}

static void release(SegEnt *e, uint64_t fence) {   // back to the free list; reusable once fence is done
    plat_mutex_lock(&C.lock);
    e->fence = fence;
    if (e->state == SE_LIVE) unlink_locked(e);
    else e->state = SE_FREE, e->next = C.free, C.free = e;
    plat_mutex_unlock(&C.lock);
}

static int ready(VkExec *ex) {   // what a segment may begin in
    return vk_exec_sec_active() && !vk_exec_oq_state() && ru_on <= 0 && !vk_dump_armed && !vk_pt_enabled() &&
           !vk_spec_census_on() && !vk_spec_mode() && !ph_following() && !(rt_trace & RT_TRACE_DRAW) && !xb_mode() &&
           !ex->pass_noatt;
}

// The hit's ops on the replay state and its draws' side effects; 0 when one would record otherwise (undone).
static int hit(VkExec *ex, SegEnt *e, const op_hdr *const *ops, int nops) {
    snap = *ex;
    uint64_t pos = vk_exec_op_pos();
    VkExecFrame f0 = vk_exec_frame;
    int di = 0, ok = vk_mem_pool_gen() == e->pgen;
    for (int i = 0; i < nops && ok; i++) {
        const op_hdr *h = ops[i];
        const void *a = nvn_op_payload(h);
        vk_exec_op_pos_set(vk_exec_op_pos() + 1);
        if (h->op == OP_DRAW || h->op == OP_DRAW_IDX) {
            vk_exec_frame.draws_seen++;
            ok = di < e->ndraws && vk_exec_seg_fill(ex, &e->d[di++], h->op, a, e->rmap);
        } else if (!vk_exec_seg_bind(ex, h->op, a)) {
            vk_exec_state_op(ex, h->op, a);
        }
    }
    for (int i = 0; i < e->nspan && ok; i++) {   // the mirror sync of each pool its buffers lie in (vk_mem_use)
        VkMemSpan sp = e->span[i].sp;
        VK_MEM_USE(sp);
        ok = sp.buffer == e->span[i].rec;
    }
    if (ok && di == e->ndraws) return 1;
    *ex = snap;
    vk_exec_op_pos_set(pos);
    vk_exec_frame = f0;
    return 0;
}

int vk_exec_seg_begin(VkExec *ex, const op_hdr *const *ops, int nops, int ndraws) {
    if (!ready(ex)) return SEG_NO;
    __atomic_add_fetch(&C.n_try, 1, __ATOMIC_RELAXED);
    uint64_t key = seg_key(ex, ops, nops), gpu = gpu_done();
    plat_mutex_lock(&C.lock);
    SegEnt *e = C.tab[key & (SEG_TAB - 1)];
    int busy = 0;
    while (e && (e->key != key || e->fence > gpu || e->ndraws != ndraws)) busy |= e->key == key, e = e->next;
    key_parts(ex, ops, nops);
    variant_census();
    field_census();
    if (!e) {
        if (busy) kd_busy++;
        else key_miss();
    }
    uint64_t was = 0;
    if (e) was = e->fence, e->fence = ex->fence;
    plat_mutex_unlock(&C.lock);
    if (e) {
        int ok = hit(ex, e, ops, nops);
        if (ok && seg_mode == 2) {   // check: the same draws recorded as usual, the two states compared at the end
            CHK.filled = *ex, CHK.on = 1;
            *ex = snap;
            vk_exec_op_pos_set(vk_exec_op_pos() - (uint64_t)nops);
            plat_mutex_lock(&C.lock);
            e->fence = was;
            plat_mutex_unlock(&C.lock);
            __atomic_add_fetch(&C.n_chk, 1, __ATOMIC_RELAXED);
            return SEG_CAP;
        }
        if (ok) {
            vk_exec_sec_switch(ex, VK_NULL_HANDLE);
            vkCmdExecuteCommands(ex->cb, 1, &e->cb);
            ex->recorded++;
            vk_exec_sec_resume(ex);
            e->used = ex->fence;
            __atomic_add_fetch(&C.n_hit, 1, __ATOMIC_RELAXED);
            __atomic_add_fetch(&C.draws_hit, (uint64_t)ndraws, __ATOMIC_RELAXED);
            return SEG_HIT;
        }
        __atomic_add_fetch(&C.n_undo, 1, __ATOMIC_RELAXED);
        int dead = vk_mem_pool_gen() != e->pgen;
        for (int i = 0; i < e->ndraws && !dead; i++)
            dead = __atomic_load_n(&e->d[i].info->pipeline, __ATOMIC_ACQUIRE) != e->d[i].pipe;
        if (dead) {
            __atomic_add_fetch(&C.n_kill, 1, __ATOMIC_RELAXED);
            release(e, was);
        } else {
            plat_mutex_lock(&C.lock);
            e->fence = was;
            plat_mutex_unlock(&C.lock);
            return SEG_NO;
        }
    }
    // Admission: a key is captured the second time it is seen; the first time the stretch records as usual and only the
    // key is noted (seg2, 28/11/2026: 11180 captures for 1781 hits, a capture costing more than a usual recording).
    {
        static uint64_t seen[SEEN_TAB];
        uint64_t *s = &seen[(key >> 7) & (SEEN_TAB - 1)];
        if (__atomic_exchange_n(s, key, __ATOMIC_RELAXED) != key) return __atomic_add_fetch(&C.n_first, 1, __ATOMIC_RELAXED), SEG_NO;
    }
    plat_mutex_lock(&C.lock);
    e = take_locked(ex->fence, gpu);
    if (e) e->state = SE_CAP, e->fence = ex->fence, e->key = key;
    plat_mutex_unlock(&C.lock);
    if (!e) return __atomic_add_fetch(&C.n_full, 1, __ATOMIC_RELAXED), SEG_NO;
    vkResetCommandPool(vk_ctx()->device, e->pool, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
                                   .pInheritanceInfo = vk_exec_sec_inherit()};
    vkBeginCommandBuffer(e->cb, &bi);
    vk_exec_sec_switch(ex, e->cb);
    B.sup_valid = 0;   // the support buffer comes from seg_support from here
    e->pgen = vk_mem_pool_gen();   // before any resolve
    e->nspan = 0;
    cap_tls = (SegCap){.e = e, .ndraws = ndraws};
    seg_cap = &cap_tls;
    __atomic_add_fetch(&C.n_cap, 1, __ATOMIC_RELAXED);
    return SEG_CAP;
}

// check: the fields a hit and the usual recording may set differently without recording otherwise are not compared
static void check_end(VkExec *ex) {
    CHK.on = 0;
    VkExec a = CHK.filled, b = *ex;
    VkExec *s[2] = {&a, &b};
    for (int i = 0; i < 2; i++) {
        VkExec *x = s[i];
        x->cb = VK_NULL_HANDLE, x->recorded = 0, x->dirty = 0, x->pass_crumb = 0;
        x->views = NULL, x->nviews = x->cap_views = 0, x->barrier_at = x->last_dispatch = x->clear_run_at = 0;
        x->fb_fs = NULL, x->fb_serial = x->fb_pass = x->fb_sampled = 0;
    }
    if (!memcmp(&a, &b, sizeof a)) return;
    __atomic_add_fetch(&C.n_chk_bad, 1, __ATOMIC_RELAXED);
    static int logs;
    if (logs >= 20) return;
    logs++;
    const uint8_t *p = (const uint8_t *)&a, *q = (const uint8_t *)&b;
    for (size_t o = 0, n = 0; o < sizeof a && n < 8; o++)
        if (p[o] != q[o]) {
            rt_log("vk: MKW_SEG check: the hit's state differs at byte %zu of VkExec (%02x, recorded %02x)", o, p[o], q[o]);
            n++;
            while (o + 1 < sizeof a && p[o + 1] != q[o + 1]) o++;
        }
}

void vk_exec_seg_end(VkExec *ex) {
    if (CHK.on) return check_end(ex);
    SegCap *c = seg_cap;
    seg_cap = NULL;
    SegEnt *e = c->e;
    if (!c->bad && c->nd != c->ndraws) c->bad = SEGB_SKIP;
    if (!c->bad && vk_mem_pool_gen() != e->pgen) c->bad = SEGB_SKIP;
    if (!c->bad && !vk_exec_sec_in(e->cb)) c->bad = SEGB_PASS;
    B.sup_valid = 0;
    if (c->bad) {   // the secondary goes on as a plain one of the pass (or was executed at its end)
        __atomic_add_fetch(&C.bad[c->bad], 1, __ATOMIC_RELAXED);
        release(e, ex->fence);
        return;
    }
    vk_exec_sec_resume(ex);   // executes it and opens the next one
    e->ndraws = c->nd, e->used = ex->fence;
    plat_mutex_lock(&C.lock);
    e->state = SE_LIVE;
    SegEnt **b = &C.tab[e->key & (SEG_TAB - 1)];
    e->next = *b, *b = e;
    plat_mutex_unlock(&C.lock);
    __atomic_add_fetch(&C.n_pub, 1, __ATOMIC_RELAXED);
}

void seg_bad(int why) {
    if (!seg_cap->bad) seg_cap->bad = why;
}

void seg_cap_draw(VkExec *ex, const VkPipeInfo *info, VkPipeline p, const GfxShader *vs, const GfxShader *fs,
                  uint32_t fbl, uint32_t wmask) {
    (void)ex;
    SegCap *c = seg_cap;
    if (c->nd == SEG_DRAWS) return seg_bad(SEGB_SKIP);
    c->e->d[c->nd++] = (SegDraw){info, p, vs, fs, fbl, wmask, {c->pend_row[0], c->pend_row[1]},
                                 {c->pend_span[0], c->pend_span[1]}};
    c->pend_row[0] = c->pend_row[1] = 0, c->pend_span[0] = c->pend_span[1] = 0;
}

void seg_cap_span(uint64_t addr, const VkMemSpan *sp, VkBuffer recorded) {
    SegEnt *e = seg_cap->e;
    for (int i = 0; i < e->nspan; i++)
        if (e->span[i].sp.pool == sp->pool && e->span[i].rec == recorded) return;   // one mirror sync per pool
    if (e->nspan == SEG_SPANS) return seg_bad(SEGB_DESC);
    e->span[e->nspan++] = (SegSpan){addr, *sp, recorded};
}

uint8_t *seg_cap_rows(int stage, uint32_t bytes, VkDeviceAddress *addr) {
    SegCap *c = seg_cap;
    uint32_t n = bytes / 16;
    if (c->rows + n > SEG_ROW_BYTES / 16 || n > 255) return seg_bad(SEGB_ROWS), NULL;
    c->pend_row[stage] = (uint16_t)c->rows, c->pend_span[stage] = (uint8_t)n;
    *addr = c->e->raddr + 16u * c->rows;
    uint8_t *at = c->e->rmap + 16u * c->rows;
    c->rows += (int)n;
    return at;
}

int seg_support(const uint8_t *b, uint32_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (n > SUP_SLOT) return seg_bad(SEGB_DESC), 0;
    uint64_t h = XXH3_64bits(b, n) | 1;
    plat_mutex_lock(&SUP.lock);
    if (!SUP.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)SUP_SLOTS * SUP_SLOT,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        VmaAllocationCreateInfo ac = {.usage = VMA_MEMORY_USAGE_AUTO, .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocation al;
        VmaAllocationInfo inf;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ac, &SUP.buf, &al, &inf) != VK_SUCCESS) rt_fatal("vk: MKW_SEG support buffers");
        SUP.map = inf.pMappedData;
    }
    int i = 0;
    while (i < SUP.n && (SUP.hash[i] != h || memcmp(SUP.map + (size_t)i * SUP_SLOT, b, n))) i++;
    if (i == SUP.n && SUP.n < SUP_SLOTS) memcpy(SUP.map + (size_t)i * SUP_SLOT, b, n), SUP.hash[SUP.n++] = h;
    plat_mutex_unlock(&SUP.lock);
    if (i == SUP_SLOTS) return seg_bad(SEGB_DESC), 0;
    *buf = SUP.buf, *off = (VkDeviceSize)i * SUP_SLOT;
    return 1;
}

void vk_exec_seg_log(void) {
    if (seg_mode <= 0) return;
    rt_log("vk: MKW_SEG: %llu tries, %llu hits (%llu draws), %llu undone (%llu entries dead), %llu captures (%llu kept; bad: "
           "skip %llu pass %llu feedback %llu index %llu desc %llu classic %llu rows %llu temporal %llu), %llu without an "
           "entry, %d entries; check %llu, %llu differing; %llu first sightings recorded as usual",
           (unsigned long long)C.n_try, (unsigned long long)C.n_hit, (unsigned long long)C.draws_hit,
           (unsigned long long)C.n_undo, (unsigned long long)C.n_kill, (unsigned long long)C.n_cap,
           (unsigned long long)C.n_pub, (unsigned long long)C.bad[SEGB_SKIP], (unsigned long long)C.bad[SEGB_PASS],
           (unsigned long long)C.bad[SEGB_FEEDBACK], (unsigned long long)C.bad[SEGB_INDEX],
           (unsigned long long)C.bad[SEGB_DESC], (unsigned long long)C.bad[SEGB_CLASSIC], (unsigned long long)C.bad[SEGB_ROWS],
           (unsigned long long)C.bad[SEGB_TEMPORAL], (unsigned long long)C.n_full, C.nent, (unsigned long long)C.n_chk,
           (unsigned long long)C.n_chk_bad, (unsigned long long)C.n_first);
    char b[256];
    int o = 0;
    for (int k = 0; k < KP_N; k++) o += snprintf(b + o, sizeof b - (size_t)o, " %s %llu", kp_name[k], (unsigned long long)kd_diff[k]);
    rt_log("vk: MKW_SEG misses: %llu with their key in use by the GPU, %llu with ops never seen, %llu with every part "
           "equal (evicted); parts that changed against the last segment of the same ops:%s",
           (unsigned long long)kd_busy, (unsigned long long)kd_new, (unsigned long long)kd_same, b);
    char f[1024];
    o = 0;
    for (size_t k = 0; k < NFIX; k++)
        if (fd_fixed_n[k]) o += snprintf(f + o, sizeof f - (size_t)o, " %s %llu", fd_fixed[k].name, (unsigned long long)fd_fixed_n[k]);
    o += snprintf(f + o, sizeof f - (size_t)o, ";");
    for (size_t k = 0; k < NPIPE; k++)
        if (fd_pipe_n[k]) o += snprintf(f + o, sizeof f - (size_t)o, " %s %llu", fd_pipe[k].name, (unsigned long long)fd_pipe_n[k]);
    rt_log("vk: MKW_SEG starting states, of %llu tries: %llu exact repeats of one of the last %d with the same ops, %llu "
           "alike; fields differing from the most alike (pass %llu, vbuf %llu; fixed; pipe):%s",
           (unsigned long long)fd_tries, (unsigned long long)fd_exact, KDB_WAYS, (unsigned long long)fd_alike,
           (unsigned long long)fd_pass_n, (unsigned long long)fd_vbuf_n, f);
    o = 0;
    for (int v = 0; v < KV_N; v++)
        o += snprintf(f + o, sizeof f - (size_t)o, "%s %s %llu", v ? "," : "", kv_name[v], (unsigned long long)kv_first[v]);
    rt_log("vk: MKW_SEG keys first seen, of %llu tries:%s", (unsigned long long)kv_tries, f);
}
