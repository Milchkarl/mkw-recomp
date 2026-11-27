"""Where the threads of a run wait, and for how long: the time between each switch out and the next switch in of a
thread, summed by the user stack it left the CPU with.

Why: a CPU profile shows who burns the CPU, not who waits. In a 2-player race no thread of the game or of the replay
uses more than a third of a core, so the frame is a chain of waits (26/11/2026); kstacks.py counts the sleeps but not
their length. Needs no root: the user stacks name the wait (build.py --frame-pointers gives the runtime's frames; the
translated code has none, so a stack through it stops at the first runtime function).

-c 1 is needed: without it perf samples one switch in three (77,055 stacks for 233,686 switches, 26/11/2026) and a
wait took the stack of an earlier switch of its thread; the lock and present counters did not show the waits that
profile named. A switch out with no stack of its own in the 50 us before now goes to "(no stack sample)".

Usage, once the race runs (10 s are ~200,000 switches):
  perf record -e cs -c 1 -g --switch-events -p "$(pgrep -x mkw)" -o scratchpad/NAME.cs -- sleep 10
  uv run tools/run/offcpu.py scratchpad/NAME.cs [--comm 'vk replay|guest main'] [--by-tid] [--depth 4] [--top 12]
"""

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HEAD = re.compile(r"^(.+?)\s+(\d+)\s+(\d+\.\d+):\s+(cs:|PERF_RECORD_SWITCH\s+(OUT|IN))")
FRAME = re.compile(r"^\s+([0-9a-f]+)\s+(.+?)\s+\((.+)\)$")
KERNEL = 0xFFFF800000000000
STALE = 50e-6  # s: a stack sampled longer before the switch out belongs to an earlier switch


def waits(perf: Path, depth: int, by_tid: bool) -> tuple[dict[tuple[str, str], list[float]], float]:
    """(comm, stack) -> the lengths of its waits in seconds, and the seconds the recording covers."""
    proc = subprocess.Popen(
        ["perf", "script", "-i", str(perf), "--show-switch-events", "-F", "comm,tid,time,event,ip,sym,dso"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )  # fmt: skip
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    stack: dict[int, tuple[float, str]] = {}  # the user frames of each thread's last cs sample, and when
    left: dict[int, tuple[float, str]] = {}  # thread -> when it left the CPU and with which stack
    frames: list[str] | None = None
    tid, first, last, at = 0, 0.0, 0.0, 0.0
    assert proc.stdout
    for line in proc.stdout:
        m = HEAD.match(line)
        if not m:
            if frames is None:
                continue
            f = FRAME.match(line)
            if f and int(f[1], 16) < KERNEL and len(frames) < depth:
                frames.append(f[2].split("+")[0] if f[2] != "[unknown]" else Path(f[3]).name)
            elif not line.strip():
                stack[tid] = (at, " < ".join(frames) or "(no user frame)")
                frames = None
            continue
        comm, tid, t = m[1].strip(), int(m[2]), float(m[3])
        if by_tid:
            comm = f"{comm} {tid}"
        first, last = first or t, t
        if m[4] == "cs:":
            frames, at = [], t
        elif m[5] == "OUT":
            when, why = stack.get(tid, (0.0, ""))
            if t - when > STALE:
                why = "(no stack sample)"
            left[tid] = (t, why + (" [preempted]" if "preempt" in line else ""))
        elif tid in left:
            t0, why = left.pop(tid)
            out[comm, why].append(t - t0)
    proc.wait()
    return out, last - first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("perf", type=Path)
    ap.add_argument("--comm", default="", help="only the threads whose name matches this regex")
    ap.add_argument("--depth", type=int, default=4, help="user frames per stack (default 4)")
    ap.add_argument("--top", type=int, default=12, help="stacks per thread name (default 12)")
    ap.add_argument("--by-tid", action="store_true",
                    help="each thread apart (the game's threads all carry the name guest)")  # fmt: skip
    a = ap.parse_args()
    out, secs = waits(a.perf, a.depth, a.by_tid)
    per_comm: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    for (comm, why), d in out.items():
        if not a.comm or re.search(a.comm, comm):
            per_comm[comm].append((sum(d), len(d), why))
    print(f"{a.perf.name}: {secs:.1f} s")
    for comm, rows in sorted(per_comm.items(), key=lambda kv: -sum(r[0] for r in kv[1])):
        total, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
        print(f"{comm}: off the CPU {1000 * total / secs:.0f} ms per second (all its threads), {n / secs:.0f} waits/s")
        for t, k, why in sorted(rows, reverse=True)[: a.top]:
            print(f"  {1000 * t / secs:7.1f} ms/s  {k / secs:7.0f}/s  avg {1e6 * t / k:8.0f} us  {why}")


if __name__ == "__main__":
    main()
