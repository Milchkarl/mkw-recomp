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

#define LOG_ONCE(...) do { static volatile int32_t once_; if (!plat_xchg32(&once_, 1)) rt_log(__VA_ARGS__); } while (0)
#define BL_ROW_MAX 64   // entries per stage: bindings go up to MAX_BINDINGS

static int bl_on = -1;

int vk_bindless_on(void) {
    if (bl_on < 0) {
        const char *e = cfg_get("MKW_BINDLESS");
        int want = e && atoi(e) > 0;
        if (want && !vk_ctx()->has_buffer_device_address)
            rt_fatal("vk: MKW_BINDLESS needs bufferDeviceAddress, which this device lacks");
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

void vk_exec_bl_rows(VkExec *ex, const VkPipeInfo *info, const VkDescSlot *d) {
    if (!vk_bindless_on()) rt_fatal("vk: a bindless program (SM86_BINDLESS) without MKW_BINDLESS=1");
    uint32_t e[2][BL_ROW_MAX][4];
    uint32_t span[2] = {0, 0};
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
        }
        at += cnt;
    }
    struct { VkDeviceAddress row[2], dummy; } pc = {{0, 0}, dummy_addr()};
    for (int s = 0; s < 2; s++) {
        if (!span[s]) continue;
        VkBuffer buf;
        VkDeviceSize off;
        memcpy(scratch(ex, 16ull * span[s], &buf, &off), e[s], 16ull * span[s]);
        pc.row[s] = vk_buf_addr(buf) + off;
    }
    vkCmdPushConstants(ex->cb, info->layout, info->pc_stages, 0, sizeof pc, &pc);
    ex->recorded++;
}
