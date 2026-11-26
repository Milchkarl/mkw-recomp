#!/usr/bin/env python3
"""Per A/B side, what MKW_CPU_COUNTERS saw on each class of thread (rep: the replay thread recording submissions, rout:
the rest of its time, chk: its chunk threads recording, cout: their time between chunks, game: the game's submitting
threads): CPU ms, clock (GHz: cycles over CPU time), instructions per cycle, millions of instructions, cache misses per
thousand instructions, ms waiting for a core (run queue), and for the spans measured, their wall ms, the replay's ms
waiting for its chunks and what is left of the wall (off the CPU for anything else: locks), each per frame, with the
95 % interval of B - A over adjacent block pairs (tools/runner/abstat.py's pairing).

Why: with the game ahead of the replay (MKW_FENCE_AHEAD) the replay took 10-27 % longer per frame; a lower clock (a
laptop's power budget), shared caches or memory (fewer instructions per cycle, more misses), more work (more
instructions), waiting for a core, for its chunks or for a lock read apart here, from one run's frames.tsv.

Usage: uv run tools/run/cpuctr.py build/run/NAME.frames.tsv [--settle 5]
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))

import abstat  # noqa: E402

CLASSES = ("rep", "rout", "chk", "cout", "game")
COLS = ("cpu_us", "kcyc", "kins", "miss", "rq_us", "wall_us", "wait_us")
# name, per-frame value of a block from its column sums and frames
STATS = (
    ("cpu ms", lambda s, n: s["cpu_us"] / n / 1000),
    ("run queue ms", lambda s, n: s["rq_us"] / n / 1000),
    ("wall ms", lambda s, n: s["wall_us"] / n / 1000),
    ("chunk wait ms", lambda s, n: s["wait_us"] / n / 1000),
    ("rest ms", lambda s, n: (s["wall_us"] - s["cpu_us"] - s["rq_us"] - s["wait_us"]) / n / 1000 if s["wall_us"] else math.nan),
    ("GHz", lambda s, n: s["kcyc"] / s["cpu_us"] if s["cpu_us"] else math.nan),
    ("IPC", lambda s, n: s["kins"] / s["kcyc"] if s["kcyc"] else math.nan),
    ("M instr", lambda s, n: s["kins"] / n / 1000),
    ("miss/kinstr", lambda s, n: s["miss"] / s["kins"] if s["kins"] else math.nan),
)


def block_sums(rows: list[dict[str, float]], settle: int, cls: str) -> list[tuple[int, int, dict[str, float], int]]:
    """(index, side, column sums, frames) per A/B block, its first `settle` frames left out, as abstat.blocks."""
    period = abstat.period_ms(rows)
    if not period:
        return []
    groups: dict[int, list[dict[str, float]]] = {}
    for r in rows:
        if r["world_ms"] >= 0:
            groups.setdefault(int(r["world_ms"] // period), []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        kept = rs[settle:]
        if not kept or rs[-1]["world_ms"] - rs[0]["world_ms"] < 0.5 * period:
            continue
        side = statistics.mode(int(r["ab"]) for r in kept)
        sums = {c: sum(r.get(f"{cls}_{c}", 0.0) for r in kept) for c in COLS}
        out.append((k, side, sums, len(kept)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", type=Path, help="NAME.frames.tsv of a run with MKW_AB and MKW_CPU_COUNTERS=1")
    ap.add_argument("--settle", type=int, default=5, help="frames left out at the start of each block (5)")
    args = ap.parse_args()
    rows = abstat.read_rows(args.frames)
    if not rows or "rep_kcyc" not in rows[0]:
        sys.exit(f"{args.frames}: no MKW_CPU_COUNTERS columns")
    for cls in CLASSES:
        if f"{cls}_kcyc" not in rows[0]:
            continue
        bs = block_sums(rows, args.settle, cls)
        print(f"{cls}:")
        for name, fn in STATS:
            blocks = [abstat.Block(k, side, n, fn(s, n)) for k, side, s, n in bs]
            blocks = [b for b in blocks if not math.isnan(b.mean_ms)]
            if not blocks or not any(b.mean_ms for b in blocks):
                continue
            v = abstat.verdict(abstat.pair_diffs(blocks), blocks)
            half = f"±{v.half_ms:.3f}" if v.half_ms is not None else "(too few pairs)"
            print(f"  {name:12} A {v.a_ms:8.3f}  B {v.b_ms:8.3f}  B - A {v.diff_ms:+.3f} {half} ({v.pct:+.1f} %, "
                  f"{v.pairs} pairs)")


if __name__ == "__main__":
    main()
