"""Where the CPU of a run's perf profile went: per thread group, and inside the game's threads the translated code
(f_7...), the runtime, libc, the kernel and the clock calls.

Why: to compare two builds of the game code (another optimization level, PGO, a lifter change) without the frame time,
which the GPU and the waits blur. The replay threads do the same work per frame in both builds, so the translated
code's samples over theirs is its CPU per frame in units that do not depend on how many frames the profile covers.

Usage: uv run tools/run/cpu_share.py scratchpad/NAME.perf [OTHER.perf ...]
       (a race run records it; PERF_ARGS="-D 62000" starts the profile inside the race)
       --last SECS: only the profile's last SECS seconds (the race of a race run, which exits at its end: the part
       before the race differs in length between runs and has another mix; 26/11/2026, 1.12 against 1.19 for the
       same binary)
       A profile is read with the build/mkw that recorded it: perf buildid-cache -a build/mkw before building again.
       --kernel: also the kernel functions of the game's and the replay's threads, named with the running kernel's
       System.map (perf cannot name them where kptr_restrict hides /proc/kallsyms; the load offset is taken from the
       lowest sampled address, so a profile from another boot or kernel gives wrong names)
"""

import argparse
import bisect
import platform
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


def window(perf: Path, last: float) -> list[str]:
    """perf report's --time for the profile's last `last` seconds; nothing for the whole profile."""
    if not last:
        return []
    times = subprocess.run(["perf", "script", "-i", str(perf), "-F", "time"], capture_output=True, text=True,
                           check=True).stdout.split()  # fmt: skip
    end = float(times[-1].rstrip(":"))
    return ["--time", f"{end - last:.6f},{end:.6f}"]


def shares(perf: Path, time: list[str]) -> tuple[Counter[str], Counter[str]]:
    """Samples per thread group, and per kind inside the game's threads."""
    out = subprocess.run(
        ["perf", "report", "-i", str(perf), "--stdio", "--no-children", "-n", "--sort", "comm,dso,symbol", *time],
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


KLINE = re.compile(r"^\s+[0-9.]+%\s+(\d+)\s+(.+?)\s+\[k\] 0x([0-9a-f]+)")
KERNEL_ALIGN = 0x200000  # KASLR moves the kernel text in steps of 2 MiB


def kernel(perf: Path, top: int) -> None:
    """The kernel functions with most samples in the game's and in the replay's threads."""
    smap = Path("/usr/lib/modules") / platform.release() / "build/System.map"
    syms = sorted(
        (int(addr, 16), name)
        for addr, typ, name in (ln.split()[:3] for ln in smap.read_text().splitlines())
        if typ in "tTwW"
    )
    addrs = [addr for addr, _ in syms]
    text = next(addr for addr, name in syms if name == "_text")
    out = subprocess.run(
        ["perf", "report", "-i", str(perf), "--stdio", "--no-children", "-n", "--sort", "comm,symbol"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    rows = [(int(m[1]), m[2].strip(), int(m[3], 16)) for m in map(KLINE.match, out.splitlines()) if m]
    if not rows:
        print("  no kernel samples without a name")
        return
    offset = min(addr for _, _, addr in rows) // KERNEL_ALIGN * KERNEL_ALIGN - text
    for group, comms in (("game", GAME), ("replay", REPLAY)):
        names: Counter[str] = Counter()
        for n, comm, addr in rows:
            if comm in comms:
                names[syms[bisect.bisect_right(addrs, addr - offset) - 1][1]] += n
        total = sum(names.values())
        print(f"  kernel in the {group} threads: {total} samples")
        for name, n in names.most_common(top):
            print(f"    {n:6d} {100 * n / max(total, 1):5.1f} %  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("perf", nargs="+", type=Path)
    ap.add_argument("--kernel", type=int, nargs="?", const=15, default=0, metavar="N",
                    help="the N kernel functions with most samples (default 15)")  # fmt: skip
    ap.add_argument("--last", type=float, default=0, metavar="SECS", help="only the profile's last SECS seconds")
    a = ap.parse_args()
    for perf in a.perf:
        comms, kinds = shares(perf, window(perf, a.last))
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
        if a.kernel:
            kernel(perf, a.kernel)


if __name__ == "__main__":
    main()
