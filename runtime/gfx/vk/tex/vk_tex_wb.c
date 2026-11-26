// CPU reads of GPU-written storage: mirrored textures the GPU rendered into, written back to their storage when the
// CPU reads it (an invalidate, a memcpy, ReadTexels) or waits for their fence (the small ones). From vk_tex_texel.c.
#include <stdlib.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_tex_internal.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- CPU reads of GPU-written storage
// A mirrored texture the GPU rendered into holds its texels in its image only; its storage keeps what the CPU last put
// there. The game may still read that storage with the CPU: TotK moves its generated grass maps between pools with
// memcpy once the GPU is done, and the copy then carried zeros. Such a read writes the image back first.
static void write_back(Tex *e) {
    const nvn_texdesc *d = &e->t.d;
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            uint32_t nl = lv + e->pub.drop;   // the NVN level of image level lv
            view_t v = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)nl, .min_layer = (uint16_t)ly};
            NVNcopyRegion r = {.width = (int)mip(d->width, nl),
                               .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, nl),
                               .depth = is3d ? (int)mip(d->depth, nl) : 1};
            Region g;
            if (!region_of(&e->t, &v, &r, 0, 0, &g)) continue;
            uint8_t *buf = malloc(g.img * (size_t)g.nz);
            if (!buf) rt_fatal("vk: out of memory for a texture write-back");
            if (image_texels(e, &g, &r, buf, 0)) storage_texels(&e->t, &g, &r, buf, 1);
            free(buf);
        }
    e->gpu_dirty = 0;
}

// Mirrored textures overlapping [addr, addr+size) that the GPU wrote since their last write-back.
static int owned_dirty(uint64_t addr, uint64_t size, Tex **hit) {
    int n = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        Tex *e = owned[i];
        if (__atomic_load_n(&e->gpu_dirty, __ATOMIC_RELAXED) && e->pub.addr + e->t.L.size > addr) {
            if (!hit) return 1;
            hit[n++] = e;
        }
    }
    return n;
}

size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img) {
    Region g;
    if (!region_of(t, v, r, row, img, &g)) return 0;
    return (size_t)(g.nz - 1) * g.img + (g.hb - 1) * g.row + g.wb;
}

int vk_texture_cpu_stale(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    plat_mutex_lock_shared(&owned_lock);
    int dirty = owned_dirty(addr, size, NULL);
    plat_mutex_unlock_shared(&owned_lock);
    return dirty;
}

// The newest submission recorded to write a GPU-written mirrored texture overlapping [addr, addr+size); 0: none.
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    lock();
    uint64_t f = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        const Tex *e = owned[i];
        if (!e->gpu_dirty || e->pub.addr + e->t.L.size <= addr) continue;
        if (e->gpu_fence > f) f = e->gpu_fence;
        if (e->wr_fence > f) f = e->wr_fence;
    }
    unlock();
    return f;
}

// The image is the newest content only while nothing wrote the storage since the GPU render: TotK finalizes grass maps,
// reuses their storage for new data and memcpys that data on, and writing the old image back first handed the copy
// a stale map or stale resource bytes (wrong textures in the distance). Every hit is checked before any is written
// back, since a write-back is itself a CPU write to storage another hit may share; the hits then go back oldest render
// first, so where two descriptors share storage the newer render wins.
void vk_texture_write_back(uint64_t addr, uint64_t size) {
    lock();
    Tex **hit = malloc(sizeof *hit * (size_t)(nowned + 1));
    if (!hit) rt_fatal("vk: out of memory for a texture write-back");
    int n = owned_dirty(addr, size, hit), k = 0;   // owned[] cannot change while tex_lock is held
    for (int i = 0; i < n; i++) {
        if (rewritten_since_gpu(hit[i])) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// TotK's grass system renders a 128x128 RGBA8 map of cut and trampled grass into a CPU-cached pool every frame and its
// CPU code reads the pool bytes without nvnMemoryPoolInvalidateMappedRange (cut reactions: blades, insects, the cut
// itself). On the Switch the bytes are there once the frame's fence has passed. So when the
// CPU waits for a fence, mirrored targets up to WB_SMALL bytes rendered by work up to that fence go back to their
// storage, unless the CPU rewrote it since. Larger targets still go back on an invalidate, memcpy or ReadTexels.
// Pitch textures go back at the fence whatever their size (wb_at_fence). MKW fills, on the GPU, three pitch RGBA16F
// textures for each animation of its instanced characters (scale, rotation, translation: 8x1218 texels, 0x80 bytes a
// row, 152 KiB each) and its CPU code reads the pool bytes with plain loads, every frame, for each bone (1.8.0:
// main+0x42bea0). Left on the GPU, the storage stayed as the heap had it: zeros gave the bones scale 0 and stale bytes
// gave huge matrices, the giant triangles over Shy Guy Bazaar (#72; free roam there, the same drive: 17 of 600 frames
// wrong, then 0).
int vk_texture_small_dirty(void) { return __atomic_load_n(&small_pending, __ATOMIC_ACQUIRE); }

void vk_texture_write_back_small(uint64_t fence) {
    lock();
    __atomic_store_n(&small_pending, 0, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&owned_lock);
    int n = 0, later = 0;
    for (int i = 0; i < nowned; i++) n += owned[i]->gpu_dirty && wb_at_fence(&owned[i]->t.d, owned[i]->t.L.size);
    Tex **hit = n ? malloc(sizeof *hit * (size_t)n) : NULL;
    if (n && !hit) rt_fatal("vk: out of memory for a texture write-back");
    n = 0;
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        if (!e->gpu_dirty || !wb_at_fence(&e->t.d, e->t.L.size)) continue;
        if (e->gpu_fence > fence) later = 1;
        else hit[n++] = e;
    }
    plat_mutex_unlock_shared(&owned_lock);
    if (later) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (rewritten_since_gpu(hit[i])) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    for (int i = 0; i < k; i++) write_back(hit[i]);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    static volatile int32_t n_pitch;
    for (int i = 0; i < k; i++)
        if (hit[i]->t.L.size > WB_SMALL)
            log_limited(&n_pitch, "pitch texture %#llx %ux%u format %#x (%llu bytes) written back to its storage at fence %llu",
                        (unsigned long long)hit[i]->pub.addr, hit[i]->t.d.width, hit[i]->t.d.height, hit[i]->t.d.format,
                        (unsigned long long)hit[i]->t.L.size, (unsigned long long)fence);
    free(hit);
    unlock();
}

