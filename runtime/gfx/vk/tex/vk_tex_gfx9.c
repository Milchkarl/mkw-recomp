// The GFX9 RADV decision on block-view COLOR_ATTACHMENT, free of Vulkan so tests/runtime/test_gfx9_block_ca.py builds it
// alone (as vk_defaults.h for vk_defaults.c).
//
// Why: a texture that resolved as slot 337 died at ~frame 150 on a Radeon Vega 7 (RADV 26.2.6, Mesa, Bazzite) inside
// vkCreateImage: radv_CreateImage -> ac_compute_surface -> gfx9_get_preferred_swizzle_mode ->
// Addr::V2::Gfx9Lib::HwlGetPreferredSurfaceSetting, which raises SIGTRAP (ADDR_DBG_BREAK in the assertion-enabled
// builds Fedora/Bazzite ship) when the swizzle-mode set it filters comes out empty. The filters in
// HwlGetPreferredSurfaceSetting (Mesa, gfx9addrlib.cpp): a 3D texture with mips whose depth reaches the block counts
// loses the Display modes (SW_*_D only for X/Y-major mipmaps); a 3D-as-2D-array surface keeps only Thin/Linear; a
// block-compressed texture keeps only Standard|Display — together nothing is left. Sweeping that function's whole
// configuration space against Mesa's addrlib (70k inputs) puts every SIGTRAP in exactly this class: 3D,
// block-compressed, sampled, mips, depth >= ceil(width/block_w) and >= ceil(height/block_h), with the
// 3D-as-2D-array flag. The flag RADV sets from the image's COLOR_ATTACHMENT usage alone (radv_image.c:
// radv_get_surface_flags, 3D && COLOR_ATTACHMENT && !STORAGE), and RADV reports no COLOR_ATTACHMENT feature for
// block-compressed formats (ac_get_cb_format returns COLOR_INVALID off the plain layouts), so the only way such an
// image gets it here is the block-view candidate in create_tex (BLOCK_TEXEL_VIEW_COMPATIBLE + EXTENDED_USAGE +
// COLOR_ATTACHMENT). Its vkGetPhysicalDeviceImageFormatProperties query cannot see the assert: no layout is computed
// in the query, so the candidate passes and the image creation dies.
//
// Drawing into such a volume: nothing to lose, because nothing could be drawn. COLOR_ATTACHMENT at this shape does not
// return from vkCreateImage on this driver — the SIGTRAP above is at the image's creation, before any pass — so no
// frame on RADV GFX9 ever rendered into one. Without the usage there is no attachment to draw through, and
// vk_exec_rt.c's resolve_target names the draw once (format, size, levels) instead of taking it silently. Keeping the
// usage is not an option that trades rendering for stability: RADV takes its 3D-as-2D-array flag from that usage alone
// (radv_get_surface_flags), so the crash comes back with it and no Vulkan create flag avoids it. Every other shape
// keeps its block views, including the one agl's BC encoder uses (a BC texture's blocks drawn through an RGBA16UI
// view): the guard needs a volume whose depth reaches its width and height in blocks, which a 2D texture never is.
//
// The skip: make that one shape without COLOR_ATTACHMENT — sampled, transfer, pool aliasing, the cache and the
// lifetime logic all unchanged. Without COLOR_ATTACHMENT RADV does not request the 3D-as-2D-array swizzle, and the
// same sweep shows every remaining configuration legal. RADV on GFX10+ (RDNA), RADV on GFX6-8, AMD's proprietary
// driver and the other vendors keep their block views, and 2D images never take the depth test.
//
// Which generation (#708 #815): the guard first took RADV's maxImageDimension3D of 2048 as GFX9, but Mesa before 26
// reports 2048 on RDNA too (radv_init_image_properties: 8192 from GFX10 only since then), and GFX6-8 always do. Those
// lost the draws into such a volume, a 64x64x64 BC one with 7 levels, and the courses went dark but for the kart's
// lights: Steam Deck on Mesa 24.3, a Radeon 740M on Mesa 25.0 (fixed by updating to 26.1), an RX 550, and a 7900 XTX
// on Mesa 25.2-25.3 logged it. Neither has the crash: RDNA's addrlib is Gfx10Lib/Gfx11Lib, whose filter keeps
// SW_LINEAR for a 3D image viewed as a 2D array, BC formats included (gfx10addrlib.cpp HwlGetPreferredSurfaceSetting:
// no display-mode filter for 3D mipmaps), and GFX6-8's surface
// code (ac_surface.c, the mode loop of the GFX6 path) only skips the depth-aligned modes for that flag, with a 1D
// fallback. So GFX9 is told by what RADV reports, as radv_physical_device.c sets it: a minimum subgroup size of 64
// (32 from GFX10, wave32) and filterMinmaxImageComponentMapping (from GFX9).
#include "vk_tex_gfx9.h"

int vk_tex_radv_gfx9(int mesa_radv, uint32_t vendor_id, uint32_t min_subgroup_size, int filter_minmax_mapping) {
    return mesa_radv && vendor_id == 0x1002 && min_subgroup_size == 64 && filter_minmax_mapping;
}

int vk_tex_gfx9_skip_block_ca(int radv_gfx9, int image_3d, uint32_t mip_levels, uint32_t depth, uint32_t width,
                              uint32_t height, uint32_t block_w, uint32_t block_h) {
    if (!radv_gfx9 || !image_3d || mip_levels <= 1 || !block_w || !block_h) return 0;
    return depth >= (width + block_w - 1) / block_w && depth >= (height + block_h - 1) / block_h;
}
