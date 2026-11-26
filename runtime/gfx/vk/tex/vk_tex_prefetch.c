// Texture prefetch (MKW_TEX_PREFETCH): a texture the game registers in a descriptor slot is made and uploaded on a
// thread of its own, before any draw samples it, so its first use finds the image ready.
//
// Why: the course flyover before a Grand Prix race stuttered on the replay thread, which made and uploaded each texture
// at the first draw that sampled it: two presents of ~78 ms (frame 1478/1479 of st2, 25/11/2026, RX 6700 XT, 1.8.0,
// 2560x1440), the replay 70 ms of which 45 in textures. MKW_TEX_TIMING split one such frame (lead1, frame 1482): 784
// first uploads, 174 MiB; creation 14.9 ms (the ASTC alpha scan 12.3), upload 32.0 ms (deswizzle 13.9, the GPU ASTC
// recording 13.9). 771 of those 784 had been registered more than a second before (and 718 of 747 at frame 1650): the
// game registers a course's textures as it loads, then the flyover reveals them.
//
// How, so that the replay thread never waits for this one:
// - The game thread queues the registered texture (vk_tex_prefetch_note: a copy, no lock but the queue's).
// - This thread takes the write-watch generation of its storage first (vk_mem_changed, which records every write seen
//   until then), then makes the image (create_tex) and uploads it (vk_tex_upload.c upload) on its own: its texture jobs
//   stay off the pool the replay uses (tex_par_solo_here), its staging is its own and the upload lock is taken only to
//   record the copy or the ASTC decode (tex_pf_thread). Nothing of it is visible to the replay until then.
// - Then, under the texture lock, the entry goes into the cache as entry_of would put it, with that generation as its
//   last check: a write to the storage after it is caught at the first use, which uploads again (make_fresh_snap). If
//   the replay made an entry at that address meanwhile, or anything else is there, this one is dropped (its image goes
//   once the batch that wrote it is done).
// - Queue order: every part the replay sends sends the open upload batch first (vk_sync.c), so a part that finds the
//   entry follows its upload on the queue.
// Not prefetched: storage the CPU cannot read or that is the GPU's (GPU pools, sparse), pitch-linear textures (render
// targets the CPU reads), buffers, textures over PF_MAX_BYTES, without a write watch (the hash check would differ),
// with MKW_ASTC=check or MKW_TEXPRE, and nothing while VRAM is past PF_VRAM_PCT of its budget: an image made here that
// no draw samples holds memory a first use would not have taken.
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_tex_internal.h"
#include "platform/platform.h"

_Thread_local int tex_pf_thread, tex_pf_failed;

#define PF_QUEUE 65536                // registrations waiting; past it new ones are dropped (counted). A capture's
                                      // state registers ~43 000 at once (fly, 25/11/2026)
#define PF_SEEN 16384                 // keys already queued, direct-mapped: the game registers a texture again and again
// A texture is taken PF_DELAY_NS after its registration: the game may still be writing its storage (a load registers
// first), and the write-watch query that starts a prefetch protects the pages again, so each later write faults.
// Taken at once, the load's frame before the flyover went 48-51 -> 66-72 ms (pf*a/b, 25/11/2026) and 647 of 8302
// textures used were uploaded again. The lead is over a second for nearly all (lead1).
#define PF_DELAY_NS 300000000ull
#define PF_MAX_BYTES (32ull << 20)    // storage larger than this is left to its first use (one staging buffer each)
#define PF_VRAM_PCT 80
// Pace: the uploads go on the queue the frames use, so a burst of them delays the frames behind it (pfl1, 25/11/2026:
// 1.26 GB made in the menus at once, then a 1265 ms frame). At most one batch of this thread's work is on the GPU: after
// PF_BATCH_TEXELS ASTC texels (decoded and re-encoded by compute) or PF_BATCH_BYTES of staging the batch is sent, and
// the next texture waits until it is done. A slower GPU gets the work slower; a fixed rate per wall time (2 M texels a
// frame, pfl2) left the course's textures waiting 32 s behind the menus'.
#define PF_BATCH_TEXELS (4ull << 20)
#define PF_BATCH_BYTES (32ull << 20)

static struct {
    PlatOnce once;
    PlatMutex lock;
    PlatCond work;
    struct { tex_t t; uint64_t at; } *q;   // PF_QUEUE of them, made by pf_init; at: the registration's time
    uint32_t head, n;
    uint64_t seen[PF_SEEN];
    int on;
    // statistics (this thread's, read by the log line)
    uint64_t queued, dropped, made, raced, skipped, failed, bytes, ns;
    uint64_t skip_there, skip_mem, skip_vram, skip_watch, fail_create, fail_upload;
    uint64_t used, used_bytes, stale;   // first lookups by a draw (tex_lock); of them uploaded again
    uint64_t batch_texels, batch_bytes, batch_seq, paced_ns, batches;
} PF = {PLAT_ONCE_INIT, .lock = PLAT_MUTEX_INIT, .work = PLAT_COND_INIT};

// A capture replay has no game heap: its guest memory is what the capture restored (cap_mem_restore), only ever added
// to while it runs, so the platform's test answers there
static int replay_mode;
void vk_tex_prefetch_replay(void) { replay_mode = 1; }

static int readable(uint64_t a, uint64_t n) { return replay_mode ? plat_mem_readable(a, n) : mem_guest_heap_readable(a, n); }

static uint64_t key_hash(const TexKey *k) {
    const uint8_t *p = (const uint8_t *)k;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < sizeof *k; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h ? h : 1;
}

// Worth making here: storage the CPU reads, a texture (no buffer, render target or sparse image).
static int eligible(const tex_t *t) {
    const nvn_texdesc *d = &t->d;
    int bw, bh;
    return t->mem && d->target != NVN_TEXTURE_TARGET_BUFFER && !(d->flags & NVN_TEXTURE_FLAGS_SPARSE) &&
           d->samples <= 1 && !nvn_tex_is_linear(d) && t->L.size && t->L.size <= PF_MAX_BYTES &&
           nvn_fmt_bpb(d->format, &bw, &bh) && !nvn_gpu_only((uintptr_t)t->mem) && tex_mirrored(t);
}

static int vram_room(void) {
    VkCtx *c = vk_ctx();
    VmaBudget b[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, b);
    uint32_t v = VK_MAX_MEMORY_HEAPS;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT &&
            (v == VK_MAX_MEMORY_HEAPS || c->mem_props.memoryHeaps[h].size > c->mem_props.memoryHeaps[v].size))
            v = h;
    return v == VK_MAX_MEMORY_HEAPS || !b[v].budget || b[v].usage < b[v].budget / 100 * PF_VRAM_PCT;
}

// An entry at k's address already (made by the replay, or by an earlier registration).
static int present_at(const TexKey *k) {
    return nvn_hm_get(&by_key, k) || nvn_hm_get(&by_addr, &k->addr);
}

// Before a texture: the batch sent last is done on the GPU. After it: its work counted, and the batch sent once full.
static void pace_wait(void) {
    if (!PF.batch_seq) return;
    uint64_t t0 = plat_now_ns();
    while (!vk_upload_done(PF.batch_seq)) plat_sleep_ns(500000);
    PF.batch_seq = 0;
    PF.paced_ns += plat_now_ns() - t0;
}

static void pace_count(const tex_t *t, const TexVkFormat *f) {
    if (f->astc_bw) PF.batch_texels += (uint64_t)t->d.width * t->d.height * (t->d.depth ? t->d.depth : 1) * 4 / 3;
    PF.batch_bytes += t->L.size;
    if (PF.batch_texels < PF_BATCH_TEXELS && PF.batch_bytes < PF_BATCH_BYTES) return;
    PF.batch_seq = vk_upload_flush(0);
    PF.batch_texels = PF.batch_bytes = 0;
    PF.batches++;
}

static void prefetch_one(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    lock();
    int there = present_at(&k);
    unlock();
    // the storage's pages may be gone by now (a registered descriptor outlives its pool) or trapped: only the game's
    // heap, read without a fault (st-pf1: a freed region's deswizzle faulted at 0x55da60000 in a capture replay)
    int why = there ? 1 : !readable(k.addr, t->L.size) ? 2 : !vram_room() ? 3 : 0;
    if (why) {
        PF.skipped++;
        *(why == 1 ? &PF.skip_there : why == 2 ? &PF.skip_mem : &PF.skip_vram) += 1;
        return;
    }
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f)) {
        PF.failed++, PF.fail_create++;
        return;
    }
    pace_wait();
    uint64_t t0 = plat_now_ns();
    uint32_t gen;
    if (vk_mem_changed(k.addr, t->L.size, 0, &gen) < 0) {   // no write watch: the first use decides by hash
        PF.skipped++, PF.skip_watch++;
        return;
    }
    tex_pf_failed = 0;
    Tex *e = create_tex(t, &k, 0);
    pace_count(t, &f);
    if (!e) {
        PF.failed++, PF.fail_create++;
        return;
    }
    e->t = *t;
    if (e->pub.mirrored && !e->pub.gpu_owned) upload_first(e);
    int keep = !tex_pf_failed && e->uploaded;
    lock();
    if (keep && !present_at(&k)) {
        e->wgen = gen;
        e->pf = 1;
        e->checked = e->seen = submit_epoch;   // not idle yet (evict_idle)
        e->next_at = NULL;
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
        e = NULL;
    }
    unlock();
    if (e) {   // the replay was first, or the upload failed: nothing named it but the upload batch
        if (keep) PF.raced++;
        else PF.failed++, PF.fail_upload++;
        static volatile int32_t n_fail_log;
        if (!keep)
            log_limited(&n_fail_log, "texture prefetch: %#llx %ux%u fmt %#x levels %u: not uploaded (staging %s, "
                        "mirrored %d, gpu_owned %d, uploaded %d)", (unsigned long long)k.addr, t->d.width, t->d.height,
                        t->d.format, e->pub.levels, tex_pf_failed ? "failed" : "made", e->pub.mirrored,
                        e->pub.gpu_owned, e->uploaded);
        tex_twin_destroy(e);
        vk_defer_image(e->pub.image, e->alloc);
        free(e->gpu_base);
        free(e);
        return;
    }
    PF.made++;
    PF.bytes += t->L.size;
    PF.ns += plat_now_ns() - t0;
}

// Under tex_lock: the replay's first lookup of a prefetched entry; uploaded again there when its storage changed since
void tex_pf_first_use(Tex *e, uint32_t uploads_before) {
    e->pf = 0;
    PF.used++, PF.used_bytes += e->t.L.size;
    if (e->n_up != uploads_before) PF.stale++;
}

static void pf_log(void) {
    rt_log("vk: texture prefetch: %llu made (%.1f MiB, %.0f ms on its thread), %llu made by a first use first, %llu failed "
           "(%llu image, %llu upload), %llu skipped (%llu made already, %llu storage not the heap's, %llu VRAM, %llu no "
           "write watch), %llu queued, %llu dropped (queue full); first used by a draw %llu (%.1f MiB), %llu of them "
           "uploaded again; %llu batches, %.0f ms waiting for them", (unsigned long long)PF.made, PF.bytes / 1048576.0, PF.ns / 1e6,
           (unsigned long long)PF.raced, (unsigned long long)PF.failed, (unsigned long long)PF.fail_create,
           (unsigned long long)PF.fail_upload, (unsigned long long)PF.skipped, (unsigned long long)PF.skip_there,
           (unsigned long long)PF.skip_mem, (unsigned long long)PF.skip_vram, (unsigned long long)PF.skip_watch,
           (unsigned long long)PF.queued, (unsigned long long)PF.dropped, (unsigned long long)PF.used,
           PF.used_bytes / 1048576.0, (unsigned long long)PF.stale, (unsigned long long)PF.batches,
           PF.paced_ns / 1e6);
}

static void pf_main(void *arg) {
    (void)arg;
    plat_thread_background();   // off the frame's fast cores, as the texture workers
    tex_pf_thread = 1;
    tex_par_solo_here();
    uint64_t logged = 0, logged_used = 0;
    for (;;) {
        plat_mutex_lock(&PF.lock);
        while (!PF.n) {
            // once a burst is done, and every 10 s while the draws use what was made
            uint64_t done = PF.made + PF.raced + PF.failed + PF.skipped, used = __atomic_load_n(&PF.used, __ATOMIC_RELAXED);
            if (done >= logged + 64 || (used != logged_used && done != 0)) {
                logged = done, logged_used = used;
                pf_log();
            }
            plat_cond_wait_ms(&PF.work, &PF.lock, 10000);
        }
        uint64_t now = plat_now_ns(), due = PF.q[PF.head].at + PF_DELAY_NS;
        if (now < due) {   // in registration order: the head is the oldest
            plat_mutex_unlock(&PF.lock);
            plat_sleep_ns(due - now);
            continue;
        }
        tex_t t = PF.q[PF.head].t;
        PF.head = (PF.head + 1) % PF_QUEUE, PF.n--;
        plat_mutex_unlock(&PF.lock);
        prefetch_one(&t);
    }
}

static void pf_init(void *arg) {
    (void)arg;
    const char *v = cfg_get("MKW_TEX_PREFETCH");
    PF.on = v && *v ? atoi(v) != 0 : 0;
    if (PF.on && (vk_astc_mode() == VK_ASTC_CHECK || vk_texpre_on())) {
        rt_log("vk: texture prefetch off: MKW_ASTC=check and MKW_TEXPRE upload on the replay thread only");
        PF.on = 0;
    }
    if (!PF.on) return;
    if (!(PF.q = malloc(sizeof *PF.q * PF_QUEUE)) || !plat_thread_new(pf_main, NULL, "vk tex prefetch")) {
        rt_log("vk: texture prefetch off: its queue or thread could not be made");
        PF.on = 0;
        return;
    }
    rt_log("vk: texture prefetch on (MKW_TEX_PREFETCH): registered textures are made and uploaded ahead of their first use");
}

// The game thread, at each texture registration (vk_backend.c): queued once per key.
void vk_tex_prefetch_note(const tex_t *t) {
    plat_once(&PF.once, pf_init, NULL);
    if (!PF.on || !t || !eligible(t)) return;
    TexKey k;
    make_key(t, &k);
    uint64_t h = key_hash(&k);
    uint64_t *slot = &PF.seen[h % PF_SEEN];
    if (__atomic_load_n(slot, __ATOMIC_RELAXED) == h) return;
    __atomic_store_n(slot, h, __ATOMIC_RELAXED);
    plat_mutex_lock(&PF.lock);
    if (PF.n == PF_QUEUE) {
        PF.dropped++;
    } else {
        PF.q[(PF.head + PF.n) % PF_QUEUE].t = *t;
        PF.q[(PF.head + PF.n) % PF_QUEUE].at = plat_now_ns();
        PF.n++, PF.queued++;
        plat_cond_signal(&PF.work);
    }
    plat_mutex_unlock(&PF.lock);
}
