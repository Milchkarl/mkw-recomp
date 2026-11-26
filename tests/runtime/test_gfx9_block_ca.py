"""The GFX9 RADV guard on block-view COLOR_ATTACHMENT (runtime/gfx/vk/tex/vk_tex_gfx9.c), built alone with the C
compiler at hand. RADV on GFX9 must make a 3D block-compressed image with mips and depth in blocks without
COLOR_ATTACHMENT — the combination addrlib's swizzle filter empties into a SIGTRAP inside vkCreateImage (the Vega 7
crash at slot 337) — and every other device must keep its block views: RDNA and GFX6-8 on any Mesa (#708 #815: Mesa
before 26 reports a 3D limit of 2048 on RDNA too, and the guard keyed on it left their light volume undrawn), other
vendors and drivers, 2D images, one mip, depth below the blocks. GFX9 is told by RADV's minimum subgroup size of 64
(32 from GFX10) and filterMinmaxImageComponentMapping (from GFX9).

Usage: uv run pytest tests/runtime/test_gfx9_block_ca.py
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runtime/gfx/vk/tex/vk_tex_gfx9.c"

MAIN = r"""
#include <stdio.h>
#include <stdlib.h>
#include "vk_tex_gfx9.h"
static unsigned u(const char *s) { return (unsigned)strtoul(s, 0, 0); }
int main(int argc, char **argv) {
    if (argc == 5) printf("%d\n", vk_tex_radv_gfx9(atoi(argv[1]), u(argv[2]), u(argv[3]), atoi(argv[4])));
    else printf("%d\n", vk_tex_gfx9_skip_block_ca(atoi(argv[1]), atoi(argv[2]), u(argv[3]), u(argv[4]), u(argv[5]),
                                                  u(argv[6]), u(argv[7]), u(argv[8])));
    return 0;
}
"""

# (name, expected, mesa_radv, vendor, minSubgroupSize, filterMinmaxImageComponentMapping): devices from players' logs
GENERATIONS = [
    ("Vega 7 (RENOIR), GFX9", 1, 1, 0x1002, 64, 1),
    ("Steam Deck (VANGOGH), Mesa 24.3, GFX10.3", 0, 1, 0x1002, 32, 1),
    ("Radeon 740M (PHOENIX2), Mesa 25.0, GFX11", 0, 1, 0x1002, 32, 1),
    ("RX 7900 XTX (NAVI31), Mesa 25.2, GFX11", 0, 1, 0x1002, 32, 1),
    ("RX 550 (POLARIS12), GFX8", 0, 1, 0x1002, 64, 0),
    ("AMD's own driver on Vega", 0, 0, 0x1002, 64, 1),
    ("RADV, non-AMD vendor", 0, 1, 0x10DE, 64, 1),
]

# (name, expected, radv_gfx9, 3D, mip_levels, depth, width, height, block_w, block_h)
CASES = [
    # the crash's shape: RADV on a Vega 7 (RENOIR), a 32^3 BC1 volume with mips, depth 32 >= ceil(32/4) = 8
    ("RADV GFX9 32x32x32 mipped BC1", 1, 1, 1, 6, 32, 32, 32, 4, 4),
    # the light volume of #708 #815, 64^3 with 7 levels: skipped on GFX9 only
    ("RADV GFX9 64x64x64 seven levels", 1, 1, 1, 7, 64, 64, 64, 4, 4),
    ("not GFX9 64x64x64 seven levels", 0, 0, 1, 7, 64, 64, 64, 4, 4),
    # depth exactly at the block counts is already the failing class (addrlib: 4x4x1 with two levels)
    ("RADV GFX9 4x4x1 two levels", 1, 1, 1, 2, 1, 4, 4, 4, 4),
    ("RADV GFX9 32x32x8 two levels (depth = blocks)", 1, 1, 1, 2, 8, 32, 32, 4, 4),
    # non-square: the depth must reach both block counts, 64x16 -> 16x4
    ("RADV GFX9 64x16x16 mipped", 1, 1, 1, 4, 16, 64, 16, 4, 4),
    ("RADV GFX9 64x16x15 mipped (depth 15 < 16)", 0, 1, 1, 4, 15, 64, 16, 4, 4),
    ("RADV GFX9 32x32x7 two levels (depth 7 < 8)", 0, 1, 1, 2, 7, 32, 32, 4, 4),
    # 2D images and single-level volumes never enter the failing class
    ("RADV GFX9 2D 32x32 mipped", 0, 1, 0, 6, 32, 32, 32, 4, 4),
    ("RADV GFX9 32x32x32 one level", 0, 1, 1, 1, 32, 32, 32, 4, 4),
    # no division by zero on a malformed block size (the call site only ever passes 4x4)
    ("RADV GFX9 zero block size", 0, 1, 1, 6, 32, 32, 32, 0, 0),
]


@pytest.fixture(scope="module")
def skip_exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("gfx9")
    (d / "main.c").write_text(MAIN)
    exe = d / "gfx9.exe"
    subprocess.run([cc, "-O1", f"-I{SRC.parent}", str(SRC), str(d / "main.c"), "-o", str(exe)], check=True)
    return exe


def run(exe: Path, case: tuple) -> int:
    args = [str(exe), *(str(v) for v in case[2:])]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def test_radv_gfx9(skip_exe: Path) -> None:
    for name, want, *rest in GENERATIONS:
        got = run(skip_exe, (name, want, *rest))
        assert got == want, f"{name}: GFX9 = {got}, want {want}"


def test_skip_block_ca(skip_exe: Path) -> None:
    for name, want, *rest in CASES:
        got = run(skip_exe, (name, want, *rest))
        assert got == want, f"{name}: skip = {got}, want {want}"
