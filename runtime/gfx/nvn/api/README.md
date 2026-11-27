# runtime/gfx/nvn/api

The `nvn*` functions the game calls: device, queues, pools, buffers, syncs and objects; command buffers are recorded as
operation logs for the back end (`../../vk/`).

| File | What it is |
|---|---|
| `nvn_api.h`, `nvn_api_core.c` | Device, queue, memory pools, buffers, syncs and events |
| `nvn_api_obj.c` | Descriptor pools, textures, samplers, programs, windows and state objects |
| `nvn_api_cmd.c`, `nvn_cmd_internal.h` | Command buffers: recording as operation logs and binds deferred to submit; the header, the shared recording helpers |
| `nvn_api_state.c` | The draw state commands (targets, binds, states), recorded as `nvn_obj.h` loads |
| `nvn_rec_census.c` | `MKW_REC_CENSUS` (diagnosis): how long before its submit the game finished recording each submit's draws, and the deferred binds that change by the submit |
| `nvn_subw.c` | `MKW_EARLY_SUBMASK`: the ranges a submit's GPU work writes in guest memory, from its ops and the storage buffers each program writes (told by the back end), for the early-read waits |
| `nvn_rec_fixed.c` | `MKW_REC_CENSUS=fixed` (diagnosis): what the recordings the game calls again and again set before their first draw and hold (barriers, queries, copies, events), weighted by their draws |
| `nvn_qorder.c` | The work of the game's two queues ordered by their event waits (`MKW_QUEUE_WAITS`, off: census; `auto`: only while more than one view is drawn); submits held in place in the replay queue |
| `nvn_dstk.c` | Diagnosis: the draws of the next frames counted by guest call stack and depth target (`ctl drawstacks`) |
| `nvn_ext.c` | Functions that only report saved state, debug metadata and hints |
| `nvn_obj.h`, `nvn_hmap.h`, `nvn_pool_index.h` | Object and record formats; the replay's hash table and pools sorted by address |
