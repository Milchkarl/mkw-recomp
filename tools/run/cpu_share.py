"""Where the CPU of a run's perf profile went: per thread group, and inside the game's threads the translated code
(f_7...), the runtime, libc, the kernel and the clock calls.

Why: to compare two builds of the game code (another optimization level, PGO, a lifter change) without the frame time,
which the GPU and the waits blur. The replay threads do the same work per frame in both builds, so the translated
code's samples over theirs is its CPU per frame in units that do not depend on how many frames the profile covers.

Usage: uv run tools/run/cpu_share.py scratchpad/NAME.perf [OTHER.perf ...]
       (a race run records it; PERF_ARGS="-D 62000" starts the profile inside the race)
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

GAME = ("guest", "guest main")
REPLAY = ("vk replay", "vk chunks", "vk recorder")
LINE = re.compile(r"^\s+([0-9.]+)%\s+(\d+)\s+(.+?)\s{2,}(\S+)\s+\[(.)\]\s+(\S+)")


def kind(dso: str, symbol: str) -> str:
    if symbol.startswith("f_7"):
        return "translated"
    if dso == "[unknown]":
        return "kernel"
    if dso == "[vdso]" or symbol == "clock_gettime":
        return "clock"
    if dso.startswith("mkw"):
        return "runtime"
    return "libc" if dso.startswith("libc") else "other"


def shares(perf: Path) -> tuple[Counter[str], Counter[str]]:
    """Samples per thread group, and per kind inside the game's threads."""
    out = subprocess.run(
        ["perf", "report", "-i", str(perf), "--stdio", "--no-children", "-n", "--sort", "comm,dso,symbol"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    comms: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    for line in out.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        n, comm, dso, symbol = int(m[2]), m[3].strip(), m[4], m[6]
        comms[comm] += n
        if comm in GAME:
            kinds[kind(dso, symbol)] += n
    return comms, kinds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("perf", nargs="+", type=Path)
    a = ap.parse_args()
    for perf in a.perf:
        comms, kinds = shares(perf)
        total, game, replay = sum(comms.values()), sum(comms[c] for c in GAME), sum(comms[c] for c in REPLAY)
        if not total or not game or not replay:
            print(f"{perf.name}: no samples of the game's or the replay's threads")
            continue
        print(f"{perf.name}: {total} samples; game threads {100 * game / total:.1f} %, replay threads "
              f"{100 * replay / total:.1f} %")  # fmt: skip
        print("  inside the game's threads: "
              + ", ".join(f"{k} {100 * n / game:.1f} %" for k, n in kinds.most_common()))  # fmt: skip
        print(f"  translated code over the replay threads: {kinds['translated'] / replay:.3f}; "
              f"the game's threads over them: {game / replay:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
