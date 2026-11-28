// MKW_SEC_DRAWS=N (a measurement, default 0; MKW_SEC_AB=1: only in the B blocks of MKW_AB): inside the passes a chunk
// records (MKW_REPLAY_PAR), every N draws go to a secondary command buffer that the chunk's primary executes, each
// starting from nothing bound. What reusable stretches of draws (pantalla-partida, step b) would cost the GPU and the
// chunk threads before any of them is reused: the secondaries, their vkCmdExecuteCommands and the state each one binds
// again. A pass recorded so begins with VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT and holds only secondaries:
// one is open from the pass's beginning to its end, cut before a draw once it holds N and no occlusion query is open.
// The secondaries come from a pool of the thread and go back to it when the submission's fence passes.
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"

typedef struct { VkCommandBuffer cb; uint64_t fence; } SecSlot;

static int sec_n = -1, sec_ab;
static uint64_t n_sec, n_pass;   // counted by the chunk threads (relaxed), logged at shutdown

static _Thread_local struct {
    int chunk;                  // this thread records a chunk now
    int on;                     // the open pass is recorded in secondaries
    int draws;                  // draws in the open secondary
    VkCommandBuffer prim, sec;  // the chunk's primary and the open secondary
    VkCommandPool pool;
    SecSlot *slot;
    int nslot, cslot;
    VkCommandBufferInheritanceRenderingInfo ri;
    VkFormat color[8];
} S;

static int sec_mode(void) {
    if (sec_n < 0) {
        const char *e = cfg_get("MKW_SEC_DRAWS"), *ab = cfg_get("MKW_SEC_AB");
        int n = e && *e ? atoi(e) : 0;
        sec_ab = ab && *ab && strcmp(ab, "0");
        if (n > 0) rt_log("vk: MKW_SEC_DRAWS: chunk passes in secondaries of %d draws%s", n, sec_ab ? " in the B blocks" : "");
        __atomic_store_n(&sec_n, n > 0 ? n : 0, __ATOMIC_RELAXED);
    }
    return sec_n;
}

void vk_exec_sec_chunk(int on) { S.chunk = on && sec_mode() > 0; }

int vk_exec_sec_wanted(void) { return S.chunk && sec_mode() > 0 && (!sec_ab || rt_ab()); }

static VkCommandBuffer sec_begin(const VkExec *ex) {
    VkCtx *c = vk_ctx();
    if (!S.pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &S.pool) != VK_SUCCESS) rt_fatal("vk: MKW_SEC_DRAWS command pool");
    }
    uint64_t gpu = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &gpu);
    SecSlot *s = NULL;
    for (int i = 0; i < S.nslot && !s; i++)
        if (S.slot[i].fence <= gpu) s = &S.slot[i];
    if (!s) {
        if (S.nslot == S.cslot) {
            S.cslot = S.cslot ? S.cslot * 2 : 64;
            if (!(S.slot = realloc(S.slot, sizeof *S.slot * (size_t)S.cslot))) rt_fatal("vk: out of memory (MKW_SEC_DRAWS)");
        }
        s = &S.slot[S.nslot++];
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = S.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1};
        if (vkAllocateCommandBuffers(c->device, &ai, &s->cb) != VK_SUCCESS) rt_fatal("vk: MKW_SEC_DRAWS command buffer");
    }
    s->fence = ex->fence;
    VkCommandBufferInheritanceInfo ii = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, .pNext = &S.ri};
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                                            VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
                                   .pInheritanceInfo = &ii};
    vkBeginCommandBuffer(s->cb, &bi);
    __atomic_add_fetch(&n_sec, 1, __ATOMIC_RELAXED);
    return s->cb;
}

// After vkCmdBeginRendering with the secondary-contents flag on ex->cb: the pass's first secondary.
void vk_exec_sec_open(VkExec *ex) {
    memcpy(S.color, ex->pass_color, sizeof S.color);
    S.ri = (VkCommandBufferInheritanceRenderingInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, .colorAttachmentCount = ex->pass_ncolor,
        .pColorAttachmentFormats = S.color, .depthAttachmentFormat = ex->pass_depth,
        .stencilAttachmentFormat = ex->pass_stencil,
        .rasterizationSamples = ex->pass_samples ? (VkSampleCountFlagBits)ex->pass_samples : VK_SAMPLE_COUNT_1_BIT};
    S.on = 1, S.draws = 0, S.prim = ex->cb;
    ex->cb = S.sec = sec_begin(ex);
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_forget();
    __atomic_add_fetch(&n_pass, 1, __ATOMIC_RELAXED);
}

static void sec_flush(VkExec *ex) {
    vkEndCommandBuffer(S.sec);
    vkCmdExecuteCommands(S.prim, 1, &S.sec);
    ex->cb = S.prim, S.sec = VK_NULL_HANDLE;
}

// Before a draw of a chunk.
void vk_exec_sec_draw(VkExec *ex) {
    if (!S.on || !ex->in_pass) return;
    if (S.draws++ < sec_n || vk_exec_oq_state() & 2) return;
    sec_flush(ex);
    ex->cb = S.sec = sec_begin(ex);
    S.draws = 1;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_forget();
}

// Before vkCmdEndRendering: the primary records the end.
void vk_exec_sec_close(VkExec *ex) {
    if (!S.on) return;
    sec_flush(ex);
    S.on = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_forget();
}

void vk_exec_sec_log(void) {
    if (sec_n > 0) rt_log("vk: MKW_SEC_DRAWS: %llu passes in %llu secondaries", (unsigned long long)n_pass, (unsigned long long)n_sec);
}
