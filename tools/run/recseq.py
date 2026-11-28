"""How much of the frame's big submit repeats an earlier frame, from a MKW_REC_DUMP file, offline.

Why: whether the replay can translate frame N+1 ahead (from an earlier frame's stream, while the game still computes
N+1) and at the submit record again only what differs depends on how the big submits repeat each other. Asking that
with one game run per question took ten runs (sim2-seq to sim2-seq10); one run with MKW_REC_DUMP and this script
answer every variant in seconds.

The dump (runtime/gfx/nvn/api/nvn_rec_seq.c): each big submit's ops as the replay walks them. Per pair (submit, the one
`--base` frames before) the ops are aligned on the ops whose bytes are unique in both (patience diff: the longest run of
anchors in order, and between two anchors the ops compared one for one when both gaps are as long). An op is the same
exactly, or in shape when the fields named in `--data` (addresses and handles the GPU could read at the submit instead
of having them recorded) are left out. A segment of `--seg` draws is clean when all its ops are the same: a recorded
Vulkan command buffer cannot be patched, so a dirty segment is recorded again.

Usage: uv run tools/run/recseq.py DUMP [--base 1,2,3] [--seg 64,384] [--data ubo,ssbo,tex,vbuf,ibuf,rt]
"""

import argparse
import bisect
import struct
from collections import Counter
from dataclasses import dataclass

# Payload fields left out in shape, per op name: (offset, length) within the payload (runtime/gfx/nvn/api/nvn_obj.h).
DATA = {
    "ubo": {"BIND_UBO": [(8, 8)]},  # p_bind_buf {i32 stage, i32 index, u64 addr, u64 size}
    "ssbo": {"BIND_SSBO": [(8, 8)]},
    # p_bind_handles {i32 stage, first, n, pad; u64 src; u64 h[32]}: the handles
    "tex": {k: [(16, 8 * 33)] for k in ("BIND_TEX", "BIND_SAMP", "BIND_IMG")},
    "vbuf": {"BIND_VBUF": [(8 + 16 * i, 8) for i in range(16)]},  # p_bind_vbuf {i32 first, n; {u64 addr, size}[16]}
    "ibuf": {
        "DRAW_IDX": [(24, 8)]
    },  # p_draw_idx {i32 prim, itype, count, base_vertex, base_instance, instances; u64 ibuf}
    "rt": {"SET_RT": [(8, 64)]},  # p_set_rt {i32 n; tex_t *color[8]; ...}: the color targets' pointers
}
DRAWS = {"DRAW", "DRAW_IDX", "DRAW_IDX_IND", "DRAW_MESH", "DISPATCH", "DISPATCH_IND"}
# Ops that open or close a pass, order the GPU or move data: not recordable inside a secondary command buffer that is
# reused (exec/vk_exec.c pass_begin/pass_end, vk_exec_xfer.c).
STRUCT = {
    "SET_RT", "CLEAR_COLOR", "CLEAR_DS", "CLEAR_TEX", "CLEAR_BUF", "COPY_B2B", "COPY_B2T", "COPY_T2B", "COPY_T2T",
    "UPDATE_UB", "BARRIER", "SIGNAL_EVENT", "WAIT_EVENT", "REPORT_COUNTER", "RESET_COUNTER", "RENDER_COND",
    "RENDER_ENABLE", "QUEUE", "CALL", "JUMP", "BEGIN", "END", "DISPATCH", "DISPATCH_IND", "DRAW_MESH",
}  # fmt: skip


@dataclass
class Sub:
    number: int
    ops: list[int]
    exact: list[int]
    shape: list[int]


def read(path: str, data: list[str]) -> tuple[list[str], list[Sub]]:
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != b"MKWSEQ1\n":
        raise SystemExit(f"{path}: not a MKW_REC_DUMP file")
    (n,) = struct.unpack_from("<I", b, 8)
    at, names = 12, []
    for _ in range(n):
        ln = b[at]
        names.append(b[at + 1 : at + 1 + ln].decode())
        at += 1 + ln
    cut: dict[int, list[tuple[int, int]]] = {}
    for d in data:
        for name, spans in DATA[d].items():
            if name in names:
                cut.setdefault(names.index(name), []).extend(spans)
    subs = []
    while at + 16 <= len(b) and b[at : at + 4] == b"SUB1":
        number, size = struct.unpack_from("<QI", b, at + 4)
        at += 16
        end = at + size
        ops, exact, shape = [], [], []
        while at < end:
            op, sz = struct.unpack_from("<II", b, at)
            payload = b[at + 8 : at + sz]
            h = hash((op, payload))
            ops.append(op), exact.append(h)
            if op in cut:
                p = bytearray(payload)
                for o, ln in cut[op]:
                    p[o : o + ln] = bytes(min(ln, max(0, len(p) - o)))
                h = hash((op, bytes(p)))
            shape.append(h)
            at += sz
        subs.append(Sub(number, ops, exact, shape))
    return names, subs


def anchors(a: list[int], b: list[int]) -> list[tuple[int, int]]:
    """Pairs (i, j) of ops unique in both, the longest run in order (patience)."""
    ca, cb = Counter(a), Counter(b)
    jb = {h: j for j, h in enumerate(b) if cb[h] == 1}
    pairs = [(i, jb[h]) for i, h in enumerate(a) if ca[h] == 1 and h in jb]
    tails: list[int] = []  # index into pairs ending the best run of each length
    prev = [-1] * len(pairs)
    keys: list[int] = []
    for k, (_, j) in enumerate(pairs):
        x = bisect.bisect_left(keys, j)
        prev[k] = tails[x - 1] if x else -1
        if x == len(keys):
            keys.append(j), tails.append(k)
        else:
            keys[x], tails[x] = j, k
    out, k = [], tails[-1] if tails else -1
    while k >= 0:
        out.append(pairs[k])
        k = prev[k]
    return out[::-1]


def same(cur: Sub, base: Sub) -> list[int]:
    """Per op of cur: 1 the same exactly, 2 the same in shape, 0 not, -1 in a gap of another length than the base's
    (ops added or dropped between the two frames; aligned on exact anchors)."""
    s = [0] * len(cur.ops)
    pa = pb = -1
    for ea, eb in anchors(cur.exact, base.exact) + [(len(cur.ops), len(base.ops))]:
        if ea - pa == eb - pb:
            for g in range(1, ea - pa):
                i, j = pa + g, pb + g
                s[i] = 1 if cur.exact[i] == base.exact[j] else 2 if cur.shape[i] == base.shape[j] else 0
        else:
            for i in range(pa + 1, ea):
                s[i] = -1
        if ea < len(cur.ops):
            s[ea] = 1
        pa, pb = ea, eb
    return s


def segments(
    cur: Sub, s: list[int], seg: int, draw_ops: set[int], ok: set[int], block: frozenset[int] = frozenset()
) -> tuple[int, int, int]:
    """(segments, clean segments, draws in clean segments) for segments of seg draws. An op in block (a pass, barrier,
    copy or event: nothing a secondary command buffer inside a pass can hold) ends the segment and is in none."""
    n = clean = dclean = 0
    dirty = draws = 0
    for i in range(len(cur.ops) + 1):
        if i == len(cur.ops) or cur.ops[i] in block or (cur.ops[i] in draw_ops and draws == seg):
            if draws or i == len(cur.ops):
                n += 1
                if not dirty:
                    clean += 1
                    dclean += draws
            dirty = draws = 0
            if i == len(cur.ops):
                break
            if cur.ops[i] in block:
                continue
        dirty |= s[i] not in ok
        draws += cur.ops[i] in draw_ops
    return n, clean, dclean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--base", default="1,2,3", help="frames back to compare with")
    ap.add_argument("--seg", default="64,384", help="segment sizes in draws")
    ap.add_argument("--data", default="ubo,ssbo,tex", help=f"fields left out in shape, of {','.join(DATA)}")
    ap.add_argument("--struct", action="store_true", help=f"segments end at {','.join(sorted(STRUCT))} (reusable secondaries)")
    args = ap.parse_args()
    data = [d for d in args.data.split(",") if d]
    names, subs = read(args.dump, data)
    draw_ops = {i for i, n in enumerate(names) if n in DRAWS}
    block = frozenset(i for i, n in enumerate(names) if n in STRUCT) if args.struct else frozenset()
    print(
        f"{len(subs)} big submits, {sum(len(s.ops) for s in subs) / max(1, len(subs)):.0f} ops each; shape leaves out {data}"
    )
    for k in (int(x) for x in args.base.split(",")):
        pairs = [(subs[i], subs[i - k]) for i in range(k, len(subs))]
        if not pairs:
            continue
        ops = draws = eq = sh = deq = dsh = 0
        seg = {
            g: [0, 0, 0, 0, 0] for g in (int(x) for x in args.seg.split(","))
        }  # n, clean exact, dclean, clean shape, dclean
        diff: Counter[str] = Counter()
        gap = dgap = 0
        for cur, base in pairs:
            s = same(cur, base)
            for i, op in enumerate(cur.ops):
                d = op in draw_ops
                ops += 1
                draws += d
                eq += s[i] == 1
                sh += s[i] >= 1
                deq += d and s[i] == 1
                dsh += d and s[i] >= 1
                if s[i] <= 0:
                    diff[names[op]] += 1
                gap += s[i] < 0
                dgap += d and s[i] < 0
            for g, v in seg.items():
                n, c, dc = segments(cur, s, g, draw_ops, {1}, block)
                _, c2, dc2 = segments(cur, s, g, draw_ops, {1, 2}, block)
                v[0] += n
                v[1] += c
                v[2] += dc
                v[3] += c2
                v[4] += dc2
        print(f"\nagainst {k} frame(s) before, {len(pairs)} pairs:")
        print(
            f"  same in order: exact {100 * eq / ops:.1f} % of ops ({100 * deq / draws:.1f} % of draws), "
            f"in shape {100 * sh / ops:.1f} % ({100 * dsh / draws:.1f} %)"
        )
        for g, (_, _, dc, _, dc2) in seg.items():
            print(
                f"  clean segments of {g} draws: exact {100 * dc / draws:.1f} % of draws, in shape {100 * dc2 / draws:.1f} %"
            )
        print(
            f"  in gaps of another length (ops added or dropped): {100 * gap / ops:.1f} % of ops "
            f"({100 * dgap / draws:.1f} % of draws)"
        )
        nd = sum(diff.values())
        print(
            "  ops not the same in shape, by kind: "
            + ", ".join(f"{n} {100 * c / nd:.0f} %" for n, c in diff.most_common(10))
        )


if __name__ == "__main__":
    main()
