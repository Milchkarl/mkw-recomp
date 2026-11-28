"""The bindless rewrite of a translated program (sm86/translate/glsl/bindless.py): bank and buffer reads become bounded
reads through the per-draw row, nested indices included, and the classic declarations go."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sm86/translate"), str(ROOT / "sm86/translate/frontend")]

from glsl import bindless  # noqa: E402

SRC = """#version 460
#define LOOP_CAP 0x1000u
layout(std140, binding = 1) uniform CB3 { uvec4 d[4096]; } cb3;
layout(std430, binding = 2) readonly buffer CB3W { uint w[]; } cb3w;
uint cb3w_at(uint i) { return i < uint(cb3w.w.length()) ? cb3w.w[i] : 0u; }
layout(std430, binding = 3) buffer SB10 { uint d[]; } sb10;
void main() {
    r0 = cb3.d[sb10.d[r1 >> 2] >> 4][1];
    sb10.d[(r2 >> 2) + 1u] = cb3w_at(r3) + uint(sb10.d.length());
}"""
REFL = {
    "cbufs": [{"binding": 1, "slot": 3}, {"binding": 2, "slot": 3, "flags": 2}],
    "sbufs": [{"binding": 3, "slot": 0, "cb_offset": 4, "flags": 1}],
}


def test_rewrite() -> None:
    refl = {k: list(v) for k, v in REFL.items()}
    out = bindless.apply(SRC, refl, "ps")
    assert refl["bindless"] == 1
    assert "uniform CB3" not in out and "buffer SB10" not in out and "cb3w.w" not in out
    assert "r0 = cbv3(BLSb(sb10p(r1 >> 2)).d[sb10i(r1 >> 2)] >> 4)[1];" in out
    assert "BLSb(sb10p((r2 >> 2) + 1u)).d[sb10i((r2 >> 2) + 1u)] = cb3w_at(r3) + uint(sb10n());" in out
    assert "BLRow(blpc.row[1]).e[2]" in out  # fragment row, the word view's binding
    assert out.index("#extension GL_EXT_buffer_reference") < out.index("layout(push_constant)")


def test_textures() -> None:
    src = SRC.replace(
        "void main() {",
        "layout(binding = 4) uniform sampler2DArrayShadow t2_8;\nvoid main() {\n    v = texture(t2_8, c);",
    )
    refl = {k: list(v) for k, v in REFL.items()}
    refl["textures"] = [{"binding": 4, "type": 0x500 | 3, "cbuf_slot": 2, "handle": 8, "flags": 0}]
    out = bindless.apply(src, refl, "vs", 2)
    assert refl["bindless"] == 2 and refl["textures"][0]["flags"] & 0x80
    assert "uniform sampler2DArrayShadow t2_8" not in out
    assert "layout(set = 2, binding = 1) uniform texture2DArray BLT2DArray[];" in out
    assert (
        "#define t2_8 sampler2DArrayShadow(BLT2DArray[BLRow(blpc.row[0]).e[4].x], BLS[BLRow(blpc.row[0]).e[4].y])"
        in out
    )
    assert out.index("#define t2_8") < out.index("v = texture(t2_8, c);")
