#!/usr/bin/env python3
"""Per A/B side, where a run's threads spent their CPU: its perf profile (tools/runner records it with -k
CLOCK_MONOTONIC) split by the A/B side of the frame each sample falls in (the host_ms column of MKW_FRAMES), as ms per
frame per thread name and function, with B - A, largest differences first.

Why: a whole-run profile mixes the A and B blocks, and MKW_CPU_COUNTERS says how much CPU a class of thread took but
not in what (sim2-headab, 27/11/2026: the head's draws cost the chunk threads ~3 times what they cost the replay thread).

Usage: uv run tools/run/perf_ab.py NAME [--comm "vk chunks"] [--top 30]   (build/run/NAME.frames.tsv,
scratchpad/NAME.perf; runs before host_ms and -k CLOCK_MONOTONIC cannot be split)
"""

import argparse
import bisect
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def frames(name: str) -> tuple[list[float], list[int]]:
    with (ROOT / f"build/run/{name}.frames.tsv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if float(r["world_ms"]) >= 0]
    if not rows or "host_ms" not in rows[0]:
        sys.exit(f"{name}: no host_ms column in frames.tsv (a run before 27/11/2026)")
    return [float(r["host_ms"]) for r in rows], [int(r["ab"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--comm", action="append", help="thread names (default: all)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--hz", type=float, default=499.0, help="perf's sampling rate (tools/runner: -F 499)")
    a = ap.parse_args()
    t, side = frames(a.name)
    nframes = [side.count(0), side.count(1)]
    cmd = ["perf", "script", "-i", str(ROOT / f"scratchpad/{a.name}.perf"), "-F", "comm,time,ip,sym", "--ns"]
    if a.comm:
        cmd += ["--comms", ",".join(a.comm)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    ms = defaultdict(lambda: [0.0, 0.0])   # (comm, sym) -> samples per side
    for line in out.splitlines():
        # "<comm>  <secs>.<ns>:  <addr> <sym>": the comm may hold spaces
        head, _, rest = line.partition(":")
        parts = head.rsplit(None, 1)
        if len(parts) != 2:
            continue
        comm, ts = parts[0].strip(), parts[1]
        try:
            now = float(ts) * 1e3
        except ValueError:
            continue
        i = bisect.bisect_left(t, now)   # the frame this sample ends in
        if i >= len(t) or i == 0:
            continue
        sym = rest.split()[1] if len(rest.split()) > 1 else "?"
        ms[(comm, sym)][side[i]] += 1
    k = 1000.0 / a.hz
    rows = [(c, s, v[0] * k / max(nframes[0], 1), v[1] * k / max(nframes[1], 1)) for (c, s), v in ms.items()]
    tot = defaultdict(lambda: [0.0, 0.0])
    for c, _, x, y in rows:
        tot[c][0] += x
        tot[c][1] += y
    print(f"frames: A {nframes[0]}, B {nframes[1]}; ms of CPU per frame")
    for c, (x, y) in sorted(tot.items(), key=lambda kv: -kv[1][0] - kv[1][1])[:12]:
        print(f"  {c:20s} A {x:7.2f}  B {y:7.2f}  B-A {y - x:+7.2f}")
    print(f"-- functions by |B - A| (top {a.top})")
    for c, s, x, y in sorted(rows, key=lambda r: -abs(r[3] - r[2]))[: a.top]:
        print(f"  {x:6.2f} {y:6.2f} {y - x:+6.2f}  {c:14s} {s}")


if __name__ == "__main__":
    main()
