// MKW_GPU_SIM: a GPU of a chosen speed in place of this machine's (vk_gpu_sim.c), for vk_sync.c and vk_exec.c.
#ifndef VK_GPU_SIM_H
#define VK_GPU_SIM_H
#include <stdint.h>

int vk_sim_on(void);                                         // MKW_GPU_SIM is set
void vk_sim_draws(uint32_t draws);                           // vk_exec_record: the draws of the submission it recorded
void vk_sim_part(void);                                      // a part of the submission being recorded went to the queue
void vk_sim_issue(uint64_t value);                           // the submission is whole on the queue, before `issued` moves
uint64_t vk_sim_reached(uint64_t done, uint64_t reached);    // completion thread, unlocked: the newest value the
                                                             // simulated GPU has finished (sleeps until done + 1 is)
#endif
