// MKW_BINDLESS=1 (default off): the per-draw rows of the programs translated with SM86_BINDLESS
// (sm86/translate/glsl/bindless.py), with the SPIR-V of that pass (MKW_SPIRV=<its folder>).
//
// Why: a recorded Vulkan command buffer cannot be patched, so the replay can only reuse what it recorded for an
// earlier frame (translating frame N+1 ahead of its submit, while the game computes it) if the per-draw values that
// change between frames are read by the GPU at execution. In a 2-player race 83 % of the big submission's 16-draw
// segments repeat two frames later once constant banks, storage buffers and textures are left out, 27 % with them
// recorded (sim2-dump2); the ceiling of the whole design took the frame from 21.9 to 17.5 ms (sim2-spec).
//
// What: bind_resources (vk_exec_bind.c) fills the descriptors as always; for a bindless pipeline this writes, per
// stage, a row of uvec4(address low, address high, bound bytes, 0) at each constant bank's and storage buffer's
// binding number (the stage's own numbering) into replay scratch, and pushes the rows' addresses and a scratch word
// for stores past a bound. The descriptors are still pushed (the program ignores them): this first step only moves
// where the program reads; the next ones drop them and reuse segments. Every buffer a bank or storage buffer can be
// bound to carries SHADER_DEVICE_ADDRESS usage while this is on (vk_bda_usage, vk_bda_chain).
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_draw_internal.h"
#include "vk_seg.h"

#define LOG_ONCE(...) do { static volatile int32_t once_; if (!plat_xchg32(&once_, 1)) rt_log(__VA_ARGS__); } while (0)
#define BL_ROW_MAX 64   // entries per stage: bindings go up to MAX_BINDINGS

static int bl_on = -1;

int vk_bindless_on(void) {
    if (bl_on < 0) {
        const char *e = cfg_get("MKW_BINDLESS");
        int want = e && atoi(e) > 0;
        if (want && !vk_ctx()->has_buffer_device_address)
            rt_fatal("vk: MKW_BINDLESS needs bufferDeviceAddress, which this device lacks");
        if (want && vk_pipeline_draw_cache()) rt_fatal("vk: MKW_BINDLESS and MKW_DRAW_CACHE do not combine");
        if (want) rt_log("vk: bindless rows (MKW_BINDLESS): banks and storage buffers by address");
        bl_on = want;
    }
    return bl_on;
}

VkBufferUsageFlags vk_bda_usage(void) {
    return vk_bindless_on() ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0;
}

const void *vk_bda_chain(const void *next, VkMemoryAllocateFlagsInfo *f) {
    if (!vk_bindless_on()) return next;
    *f = (VkMemoryAllocateFlagsInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = next,
                                     .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    return f;
}

// A buffer's device address. Not cached: a pool's handle can come back for another buffer once it is destroyed, and a
// stale address is a GPU page fault (to measure: ~10 calls a draw).
VkDeviceAddress vk_buf_addr(VkBuffer b) {
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
    VkDeviceAddress a = vkGetBufferDeviceAddress(vk_ctx()->device, &di);
    if (!a) rt_fatal("vk: MKW_BINDLESS: a bound buffer has no device address (created without its usage)");
    return a;
}

// The word that stores past a storage buffer's bound land on (bindless.py sbXp): never another address.
static VkDeviceAddress dummy_addr(void) {
    static VkDeviceAddress a;
    static PlatMutex lock = PLAT_MUTEX_INIT;
    if (__atomic_load_n(&a, __ATOMIC_ACQUIRE)) return a;
    plat_mutex_lock(&lock);
    if (!a) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO};
        VkBuffer b;
        VmaAllocation al;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &b, &al, NULL) != VK_SUCCESS) rt_fatal("vk: MKW_BINDLESS scratch word");
        VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
        __atomic_store_n(&a, vkGetBufferDeviceAddress(vk_ctx()->device, &di), __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&lock);
    return a;
}

void vk_bl_tex_entry(uint16_t t, const VkDescriptorImageInfo *ii, uint32_t e[4]);
VkDescriptorSet vk_bl_set(void);

// The rows of the descriptors d of a bindless pipeline: span[s] rows of 16 bytes for stage s (0 vertex, 1 fragment).
void vk_exec_bl_entries(const VkPipeInfo *info, const VkDescSlot *d, uint32_t e[2][BL_ROW_MAX][4], uint32_t span[2]) {
    span[0] = span[1] = 0;
    for (uint32_t i = 0, at = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        uint32_t cnt = b->count ? b->count : 1;
        if ((b->kind == GFX_RES_UBO || b->kind == GFX_RES_SSBO) && b->res) {
            int s = b->stage == GFX_STAGE_FS;
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t n = b->res->binding + k;
                if (n >= BL_ROW_MAX) rt_fatal("vk: MKW_BINDLESS: binding %u past the row", n);
                while (span[s] <= n) memset(e[s][span[s]++], 0, 16);
                const VkDescriptorBufferInfo *o = &d[at + k].buffer;
                VkDeviceAddress a = vk_buf_addr(o->buffer) + o->offset;
                uint64_t range = o->range;
                if (range == VK_WHOLE_SIZE) {
                    LOG_ONCE("vk: MKW_BINDLESS: a whole-size binding reads as empty");
                    range = 0;
                }
                e[s][n][0] = (uint32_t)a, e[s][n][1] = (uint32_t)(a >> 32);
                e[s][n][2] = range > 0xffffffffull ? 0xffffffffu : (uint32_t)range, e[s][n][3] = 0;
            }
        } else if (b->kind == GFX_RES_TEX && b->res && (b->res->flags & 0x80)) {   // SM86_BINDLESS=2: by index
            int s = b->stage == GFX_STAGE_FS;
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t n = b->res->binding + k;
                if (n >= BL_ROW_MAX) rt_fatal("vk: MKW_BINDLESS: binding %u past the row", n);
                while (span[s] <= n) memset(e[s][span[s]++], 0, 16);
                vk_bl_tex_entry(b->res->type, &d[at + k].image, e[s][n]);
            }
        }
        at += cnt;
    }
}

void vk_exec_bl_rows(VkExec *ex, const VkPipeInfo *info, const VkDescSlot *d) {
    if (!vk_bindless_on()) rt_fatal("vk: a bindless program (SM86_BINDLESS) without MKW_BINDLESS=1");
    uint32_t e[2][BL_ROW_MAX][4];
    uint32_t span[2];
    vk_exec_bl_entries(info, d, e, span);
    struct { VkDeviceAddress row[2], dummy; } pc = {{0, 0}, dummy_addr()};
    for (int s = 0; s < 2; s++) {
        if (!span[s]) continue;
        uint8_t *at = seg_cap ? seg_cap_rows(s, 16u * span[s], &pc.row[s]) : NULL;   // MKW_SEG: the segment's own rows
        if (at) {
            memcpy(at, e[s], 16ull * span[s]);
            continue;
        }
        VkBuffer buf;
        VkDeviceSize off;
        memcpy(scratch(ex, 16ull * span[s], &buf, &off), e[s], 16ull * span[s]);
        pc.row[s] = vk_buf_addr(buf) + off;
    }
    vkCmdPushConstants(ex->cb, info->layout, info->pc_stages, 0, sizeof pc, &pc);
    ex->recorded++;
    if (info->bl >= 2) {   // the global set: the same handle every frame, so a recorded segment stays valid
        VkDescriptorSet g = vk_bl_set();
        vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 2, 1, &g, 0, NULL);
        ex->recorded++;
    }
}

// ---- SM86_BINDLESS=2: textures by index into the global set (set 2 of every bindless pipeline's layout)
// Arrays of sampled views by type, at the bindings sm86/translate/glsl/bindless.py TEX_ARRAYS names, and the samplers.
// One index space for every view (its type picks the array it is written into); a sampler's index is its own (the
// replay never destroys samplers in play, vk_texture_shutdown). Written once per view or sampler, updated after bind.
// A destroyed view's index is reused only once the GPU is done with what could read it (vk_defer_call, as the view's
// own destruction in vk_exec_image_gone).
enum { BL_T2D, BL_T2DA, BL_T3D, BL_TCUBE, BL_TCUBEA, BL_SAMP, BL_BINDINGS };
#define BL_VIEWS 32768
#define BL_SAMPLERS 4096
#define BL_HASH (BL_VIEWS * 2)   // open addressing, power of two

static struct {
    PlatMutex lock;   // creation, inserts, frees and descriptor writes (a leaf lock: taken under view_lock)
    VkDescriptorSetLayout layout, empty;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    uint64_t key[BL_HASH];                 // VkImageView or VkSampler handle | 1 for samplers; 0 empty, 2 deleted
    uint32_t val[BL_HASH];
    uint32_t free_v[BL_VIEWS], nfree_v, next_v;
    uint32_t next_s;
} T = {.lock = PLAT_MUTEX_INIT};

static void bl_set_init_locked(void) {
    VkDevice dev = vk_ctx()->device;
    VkShaderStageFlags st = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding b[BL_BINDINGS];
    VkDescriptorBindingFlags f[BL_BINDINGS];
    for (int i = 0; i < BL_BINDINGS; i++) {
        int s = i == BL_SAMP;
        b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, s ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              s ? BL_SAMPLERS : BL_VIEWS, st, NULL};
        f[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
               VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = BL_BINDINGS, .pBindingFlags = f};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                          .bindingCount = BL_BINDINGS, .pBindings = b};
    VkDescriptorSetLayoutCreateInfo ei = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, BL_VIEWS * BL_SAMP}, {VK_DESCRIPTOR_TYPE_SAMPLER, BL_SAMPLERS}};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                     .poolSizeCount = 2, .pPoolSizes = ps};
    if (vkCreateDescriptorSetLayout(dev, &li, NULL, &T.layout) != VK_SUCCESS ||
        vkCreateDescriptorSetLayout(dev, &ei, NULL, &T.empty) != VK_SUCCESS ||
        vkCreateDescriptorPool(dev, &pi, NULL, &T.pool) != VK_SUCCESS)
        rt_fatal("vk: MKW_BINDLESS: the global texture set could not be created (descriptor indexing)");
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = T.pool,
                                      .descriptorSetCount = 1, .pSetLayouts = &T.layout};
    if (vkAllocateDescriptorSets(dev, &ai, &T.set) != VK_SUCCESS) rt_fatal("vk: MKW_BINDLESS: global texture set");
    rt_log("vk: bindless textures (MKW_BINDLESS, SM86_BINDLESS=2): %d views, %d samplers in set 2", BL_VIEWS, BL_SAMPLERS);
}

VkDescriptorSetLayout vk_bl_set_layout(VkDescriptorSetLayout *empty) {
    plat_mutex_lock(&T.lock);
    if (!T.layout) bl_set_init_locked();
    plat_mutex_unlock(&T.lock);
    *empty = T.empty;
    return T.layout;
}

static uint32_t slot_of(uint64_t k) { return (uint32_t)((k * 0x9e3779b97f4a7c15ull) >> 40) & (BL_HASH - 1); }

// The index of k (VkImageView, or VkSampler | 1), UINT32_MAX when it has none; lock-free (inserts publish val first).
static uint32_t bl_find(uint64_t k) {
    for (uint32_t i = slot_of(k), n = 0; n < BL_HASH; i = (i + 1) & (BL_HASH - 1), n++) {
        uint64_t e = __atomic_load_n(&T.key[i], __ATOMIC_ACQUIRE);
        if (e == k) return T.val[i];
        if (!e) return UINT32_MAX;
    }
    return UINT32_MAX;
}

static uint32_t bl_add_locked(uint64_t k, int binding, VkImageView view, VkSampler samp) {
    uint32_t v = bl_find(k);
    if (v != UINT32_MAX) return v;
    if (samp) {
        if (T.next_s == BL_SAMPLERS) rt_fatal("vk: MKW_BINDLESS: more than %d samplers", BL_SAMPLERS);
        v = T.next_s++;
    } else {
        if (T.nfree_v) v = T.free_v[--T.nfree_v];
        else if (T.next_v < BL_VIEWS) v = T.next_v++;
        else rt_fatal("vk: MKW_BINDLESS: more than %d live texture views", BL_VIEWS);
    }
    VkDescriptorImageInfo ii = {samp, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = T.set, .dstBinding = (uint32_t)binding,
                              .dstArrayElement = v, .descriptorCount = 1,
                              .descriptorType = samp ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                              .pImageInfo = &ii};
    vkUpdateDescriptorSets(vk_ctx()->device, 1, &w, 0, NULL);
    uint32_t i = slot_of(k);
    while (T.key[i] > 2) i = (i + 1) & (BL_HASH - 1);
    T.val[i] = v;
    __atomic_store_n(&T.key[i], k, __ATOMIC_RELEASE);
    return v;
}

static void free_index(void *arg) {
    plat_mutex_lock(&T.lock);
    T.free_v[T.nfree_v++] = (uint32_t)(uintptr_t)arg;
    plat_mutex_unlock(&T.lock);
}

// vk_exec_image_gone (vk_exec_view.c, under view_lock) as it defers a view's destruction: the view leaves the table now
// (no new draw takes it), its index once the GPU is done.
void vk_bl_view_gone(VkImageView view) {
    if (!T.layout) return;
    uint64_t k = (uint64_t)(uintptr_t)view;
    plat_mutex_lock(&T.lock);
    for (uint32_t i = slot_of(k), n = 0; n < BL_HASH && T.key[i]; i = (i + 1) & (BL_HASH - 1), n++)
        if (T.key[i] == k) {
            uint32_t v = T.val[i];
            __atomic_store_n(&T.key[i], 2, __ATOMIC_RELEASE);
            plat_mutex_unlock(&T.lock);
            vk_defer_call(free_index, (void *)(uintptr_t)v);
            return;
        }
    plat_mutex_unlock(&T.lock);
}

// The row entry of a texture binding of type t (gfx_shader.c GfxShaderRes.type): uvec4(view index, sampler index).
void vk_bl_tex_entry(uint16_t t, const VkDescriptorImageInfo *ii, uint32_t e[4]) {
    int arr = t & 0x100, base = t & 0xff;
    int b = base == 4 ? BL_T3D : base == 5 ? (arr ? BL_TCUBEA : BL_TCUBE) : base == 3 ? (arr ? BL_T2DA : BL_T2D) : -1;
    if (b < 0 || (t & 0x200)) rt_fatal("vk: MKW_BINDLESS: texture type %#x has no array", (unsigned)t);
    if (!ii->sampler) rt_fatal("vk: MKW_BINDLESS: a texture without a sampler (separate)");
    uint64_t kv = (uint64_t)(uintptr_t)ii->imageView, ks = (uint64_t)(uintptr_t)ii->sampler | 1;
    uint32_t v = bl_find(kv), s = bl_find(ks);
    if (v == UINT32_MAX || s == UINT32_MAX) {
        plat_mutex_lock(&T.lock);
        if (v == UINT32_MAX) v = bl_add_locked(kv, b, ii->imageView, VK_NULL_HANDLE);
        if (s == UINT32_MAX) s = bl_add_locked(ks, BL_SAMP, VK_NULL_HANDLE, ii->sampler);
        plat_mutex_unlock(&T.lock);
    }
    e[0] = v, e[1] = s, e[2] = e[3] = 0;
}

VkDescriptorSet vk_bl_set(void) { return T.set; }
