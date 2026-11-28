// MKW_SEG (vk_exec_seg.c): stretches of up to SEG_DRAWS draws of a chunk recorded once into a secondary command buffer
// and executed again in later submissions while their ops, minus the data the bindless rows carry, repeat. The hooks the
// rest of the replay calls: the chunk walk (vk_exec_par.c), what a segment being captured records, and a hit's draws.
#pragma once
#include <stdint.h>

#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"

#define SEG_DRAWS 16       // draws of a segment (sim2-keyed: fixed segments of 16 found 88.7 % two frames back)
#define SEG_MIN_DRAWS 4    // fewer: recorded as usual
#define SEG_OPS 384        // ops a segment holds at most
#define SEG_ROW_BYTES 16384   // a segment's bindless rows: 16 draws x 2 stages x 32 rows of 16 bytes

int vk_exec_seg_mode(void);   // MKW_SEG: 0 off, 1 on, 2 check (each hit compared with the draws recorded as usual)
int vk_exec_seg_op(uint32_t op);   // an op a segment may hold (state and bind ops, DRAW, DRAW_IDX)

// chunk_record, before a draw op: ops[0..nops) is the segment found there (ndraws draws, the last op a draw).
// SEG_HIT: replayed from a cached secondary, nothing left to do; SEG_CAP: the caller replays the ops with vk_exec_op and
// then calls vk_exec_seg_end; SEG_NO: the caller replays them as usual.
enum { SEG_NO, SEG_HIT, SEG_CAP };
int vk_exec_seg_begin(VkExec *ex, const op_hdr *const *ops, int nops, int ndraws);
void vk_exec_seg_end(VkExec *ex);
void vk_exec_seg_log(void);

// ---- what a segment being captured records (seg_cap set on the capturing thread only)
typedef struct SegCap SegCap;
extern _Thread_local SegCap *seg_cap;
enum { SEGB_SKIP = 1, SEGB_PASS, SEGB_FEEDBACK, SEGB_INDEX, SEGB_DESC, SEGB_CLASSIC, SEGB_ROWS, SEGB_TEMPORAL, SEGB_QUERY,
       SEGB_COUNT };
void seg_bad(int why);
// The end of prepare_draw_ex: the draw's pipeline and shaders, the attachments it samples in the feedback layout (fbl)
// and those it writes (feedback_wmask).
void seg_cap_draw(VkExec *ex, const VkPipeInfo *info, VkPipeline p, const GfxShader *vs, const GfxShader *fs,
                  uint32_t fbl, uint32_t wmask);
// A vertex or index buffer the segment binds: the span vk_mem_resolve gave for addr (buffer before VK_MEM_USE) and the
// buffer recorded.
void seg_cap_span(uint64_t addr, const VkMemSpan *sp, VkBuffer recorded);
uint8_t *seg_cap_rows(int stage, uint32_t bytes, VkDeviceAddress *addr);   // where a draw's rows of stage go; NULL: none
// The support buffer of contents b (SUPPORT_BYTES): a persistent copy, as a secondary outlives replay scratch.
int seg_support(const uint8_t *b, uint32_t n, VkBuffer *buf, VkDeviceSize *off);   // 0: none left (the segment is bad)

// ---- a hit's draw, its ops' state already applied (vk_exec_draw.c, vk_exec_bind.c)
typedef struct {
    const VkPipeInfo *info;
    VkPipeline pipe;           // the pipeline recorded: the entry dies when info's changes (fast link -> optimised)
    const GfxShader *vs, *fs;
    uint32_t fbl, wmask;
    uint16_t row[2];           // first row of each stage in the entry's rows (16-byte units)
    uint8_t span[2];           // rows of each stage
} SegDraw;
// 1: the draw's side effects done and its rows written to dst; 0: it would not record what was captured (skipped, a
// feedback barrier): the segment is undone and recorded as usual.
int vk_exec_seg_fill(VkExec *ex, const SegDraw *d, uint32_t op, const void *payload, uint8_t *rows);
int vk_exec_seg_rows(VkExec *ex, const VkPipeInfo *info, uint8_t *dst[2], const uint8_t span[2]);   // vk_exec_bind.c
void vk_exec_bl_entries(const VkPipeInfo *info, const VkDescSlot *d, uint32_t e[2][64][4], uint32_t span[2]);
uint32_t feedback_wmask(const VkExec *ex, const VkPipeState *s);   // vk_exec_cb.c
int vk_exec_seg_bind(VkExec *ex, uint32_t op, const void *payload);   // vk_exec_ops.c: a bind op's effect; 0: not one
int vk_tmp_hl_pending(void);   // vk_temporal.c: the draw sampled the picture the HUD-less redo follows

// ---- vk_exec_sec.c: the pass's secondaries around a segment
int vk_exec_sec_active(void);                                  // the open pass is recorded in secondaries
const VkCommandBufferInheritanceInfo *vk_exec_sec_inherit(void);   // a secondary's inheritance in the open pass
// Ends the open secondary (executed when it recorded anything) and records into next from here (a segment's
// secondary, begun); next NULL: into the primary, for vkCmdExecuteCommands.
void vk_exec_sec_switch(VkExec *ex, VkCommandBuffer next);
void vk_exec_sec_resume(VkExec *ex);                            // after a segment: a new secondary of the pass
int vk_exec_sec_in(VkCommandBuffer cb);                         // cb is still the open secondary
uint64_t vk_mem_pool_gen(void);                                 // vk_memory.c: bumped by every pool index change
