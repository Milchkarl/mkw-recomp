"""SM86_BINDLESS=1: the program reads its constant banks and storage buffers through buffer addresses taken from a
per-draw table the replay writes when the submission is sent, not through descriptors recorded in the command buffer.

Why: a recorded Vulkan command buffer cannot be patched, and the replay can only reuse what it recorded for an earlier
frame (translating frame N+1 ahead of its submit) if every per-draw value that changes between frames is read by the
GPU at execution. In a 2-player race 83 % of the big submission's 16-draw segments repeat two frames later once the
constant banks, storage buffers and textures are left out, 27 % with them recorded (MKW_REC_DUMP and
tools/run/recseq.py, sim2-dump2, 27/11/2026); the ceiling of the whole design took the frame from 21.9 to 17.5 ms
(MKW_SPEC_CEIL, sim2-spec).

How: the classic program (head.py) declares a binding per bank and buffer; here those declarations go and the same
binding numbers index a row of the table: row entry N is uvec4(address low, address high, bound bytes, 0) of what the
classic program would find at binding N. The push constants hold the row's address for the vertex or compute stage
(row[0]) and the fragment stage (row[1]), and a scratch word (dummy) for stores past a bound. The body is rewritten:
  cbN.d[E]       -> cbvN(E): 0 past the bound range, as the classic block under robustBufferAccess;
  cbNw_at(i)     -> the same bound over the row's size;
  sbX.d[E]       -> BLSb(sbpX(E)).d[sbiX(E)]: past the bound, element 0 of the scratch word, never another address
                    (a load or store through a pointer has no robustness: out of range is a GPU page fault);
  sbX.d.length() -> sbnX().
SM86_BINDLESS=2 also reads the textures by index: one global descriptor set (set 2, written once per view or sampler
and updated after bind) holds arrays of views by type and of samplers, and row entry N of a texture binding is
uvec4(view index, sampler index, 0, 0); `tB_I` becomes the combined sampler built from the two. The row index is the
same for every invocation of a draw, so no nonuniformEXT. Storage images keep their bindings.
Task and mesh programs keep the classic form (their push constants carry the mesh path's pointers). The reflection
says "bindless": 1 or 2; the runtime (MKW_BINDLESS) fills the rows.
"""

import re

ROW = "BLRow(blpc.row[{s}]).e[{n}]"


def _matching(text: str, i: int) -> int:
    """Index of the ']' closing the '[' at text[i - 1]."""
    depth = 1
    while depth:
        c = text[i]
        depth += c == "["
        depth -= c == "]"
        i += 1
    return i - 1


def _rewrite_body(text: str) -> str:
    out, at = [], 0
    pat = re.compile(r"\b(cb\d+|sb[0-9a-f]+)\.d(\.length\(\)|\[)")
    while m := pat.search(text, at):
        out.append(text[at : m.start()])
        name, tail = m.group(1), m.group(2)
        if tail != "[":
            out.append(f"{name}n()")
            at = m.end()
            continue
        end = _matching(text, m.end())
        e = _rewrite_body(text[m.end() : end])
        if name.startswith("cb"):
            out.append(f"cbv{name[2:]}({e})")
        else:
            out.append(f"BLSb({name}p({e})).d[{name}i({e})]")
        at = end + 1
    out.append(text[at:])
    return "".join(out)


DECL = re.compile(
    r"^(layout\(std140, binding = \d+\) uniform CB\d+ |layout\(std430, binding = \d+\) readonly buffer CB\d+W |"
    r"uint cb\d+w_at\(uint i\) |layout\(std430, binding = \d+\) buffer SB[0-9a-f]+ )"
)
TEX = re.compile(r"^layout\(binding = (\d+)\) uniform (sampler\w+) (t\d+_\d+);$")
# The global set's bindings, one array per view type (runtime/gfx/vk/exec/vk_exec_bl.c BL_T*), the samplers last
TEX_ARRAYS = {"2D": 0, "2DArray": 1, "3D": 2, "Cube": 3, "CubeArray": 4}
SAMPLERS = 5


def textures(head: list[str], s: int) -> tuple[list[str], list[str]]:
    """(head without the texture bindings, the global arrays and a define per texture)."""
    kept, defs, used = [], [], set()
    for ln in head:
        m = TEX.match(ln)
        if not m:
            kept.append(ln)
            continue
        nb, typ, name = m.groups()
        view = typ.removeprefix("sampler").removesuffix("Shadow")
        if view not in TEX_ARRAYS:
            raise ValueError(f"bindless: no array for {typ}")
        used.add(view)
        e = ROW.format(s=s, n=nb)
        defs.append(f"#define {name} {typ}(BLT{view}[{e}.x], BLS[{e}.y])")
    if not used:
        return kept, []
    arrays = [f"layout(set = 2, binding = {TEX_ARRAYS[v]}) uniform texture{v} BLT{v}[];" for v in sorted(used)]
    arrays.append(f"layout(set = 2, binding = {SAMPLERS}) uniform sampler BLS[];")
    return kept, arrays + defs


def apply(src: str, refl: dict, stage: str, level: int = 1) -> str:
    """The classic program src as its bindless form (level 2: textures too); marks refl."""
    s = 1 if stage == "ps" else 0
    helpers = [
        "#extension GL_EXT_buffer_reference : require",
        "#extension GL_EXT_buffer_reference_uvec2 : require",
        "layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer BLRow { uvec4 e[]; };",
        "layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer BLCb { uvec4 d[]; };",
        "layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer BLW { uint d[]; };",
        "layout(buffer_reference, std430, buffer_reference_align = 4) buffer BLSb { uint d[]; };",
        "layout(push_constant) uniform BLPC { uvec2 row[2]; uvec2 dummy; } blpc;",
    ]
    for c in refl["cbufs"]:
        e, n = ROW.format(s=s, n=c["binding"]), c["slot"]
        if c.get("flags", 0) & 2:
            helpers.append(f"uint cb{n}w_at(uint i) {{ uvec4 e = {e}; return i < (e.z >> 2) ? BLW(e.xy).d[i] : 0u; }}")
        else:
            helpers.append(
                f"uvec4 cbv{n}(uint i) {{ uvec4 e = {e}; return i < (e.z >> 4) ? BLCb(e.xy).d[i] : uvec4(0u); }}"
            )
    for b in refl["sbufs"]:
        e, x = ROW.format(s=s, n=b["binding"]), f"sb{b['cb_offset'] * 4:x}"
        helpers += [
            f"uvec2 {x}p(uint i) {{ uvec4 e = {e}; return i < (e.z >> 2) ? e.xy : blpc.dummy; }}",
            f"uint {x}i(uint i) {{ return i < ({e}.z >> 2) ? i : 0u; }}",
            f"uint {x}n() {{ return {e}.z >> 2; }}",
        ]
    lines = src.split("\n")
    head_end = next(k for k, ln in enumerate(lines) if ln.startswith("void main()"))
    kept = [ln for ln in lines[:head_end] if not DECL.match(ln)]
    if level >= 2:
        kept, tex = textures(kept, s)
        helpers += tex
        if tex:
            helpers.insert(2, "#extension GL_EXT_nonuniform_qualifier : require")  # unsized arrays of textures
    ext = 3 if level >= 2 and helpers[2].startswith("#extension") else 2
    at = next((k for k, ln in enumerate(kept) if not ln.startswith("#")), len(kept))
    # helpers after the #version/#extension/#define lines and before the first declaration that may use them
    head = kept[:at] + helpers[:ext] + kept[at:]
    first_use = next(
        (k for k, ln in enumerate(head) if re.search(r"\b(cb\d+w?_at|cb\d+|sb[0-9a-f]+|t\d+_\d+)\b", ln)), len(head)
    )
    head = head[:first_use] + helpers[ext:] + head[first_use:]
    refl["bindless"] = level
    if level >= 2:  # the runtime fills a texture's row entry instead of its descriptor
        for t in refl["textures"]:
            t["flags"] = t.get("flags", 0) | 8
    return "\n".join([*(_rewrite_body(ln) if k >= first_use else ln for k, ln in enumerate(head)),
                      _rewrite_body("\n".join(lines[head_end:]))])  # fmt: skip
