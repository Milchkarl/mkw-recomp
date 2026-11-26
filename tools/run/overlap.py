"""Frame-time model on a player's log: how much of each frame is work and how much is one stage waiting for another.

Reads the 60-frame `perf:` windows of a log taken with MKW_TRACE=perf (interval, what the presenting thread was
blocked on, replay time, GPU time, fence wait) and, for the windows that are not held by the frame pacing, compares the
measured interval with two predictions made from the same window's numbers:

  serial      replay + GPU: each frame's recording starts when the GPU has finished the previous frame
  overlapped  max(replay, GPU, the presenting thread's own work): the three stages run side by side

A measured interval that matches «serial» says the stages run one after another; «overlapped» is the interval the same
hardware would reach with no wait between them, never under the pacing target.

Why: our own GPU is the limit in split screen and hides the waits between the game, the replay and the GPU; a player's
log already holds each stage's time on a fast GPU, so the cost of the waits is computed from it in seconds instead of
asking the player for another run.

Limits: it is arithmetic on window averages, not a run. The game's own CPU time per frame is not in the log (only the
presenting thread's), so «overlapped» is a lower bound of the interval; and the replay may slow down when it shares
the CPU with the GPU driver. The simulated GPU (MKW_GPU_SIM) measures both on a real run.

Usage: uv run tools/run/overlap.py LOG [...] [--windows]
"""

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

INTERVAL = re.compile(
    r"perf: frame (\d+): interval avg ([\d.]+) ms.*?render thread blocked: pacing (\d+)% fence (\d+)%"
    r".*?replay avg ([\d.]+) ms"
)
GPU = re.compile(
    r"perf: frame (\d+): gpu avg ([\d.]+) ms.*?fence wait per frame: render thread ([\d.]+) ms"
    r"(?:, other threads ([\d.]+) ms)?"
)
PACED_PCT = 10  # a window with more pacing than this is held by the frame limiter, not by the stages


@dataclass(frozen=True)
class Window:
    frame: int
    interval: float
    pacing_pct: int
    fence_pct: int
    replay: float
    gpu: float
    fence_wait: float

    @property
    def own(self) -> float:
        """The presenting thread's time outside pacing and fence waits. The fence wait is the longer of its own and
        the other threads' (they wait at the same time): with only another thread waiting, the presenting thread is
        taken as held by that thread's submission (an assumption; the log does not say)."""
        return max(0.0, self.interval * (1 - self.pacing_pct / 100) - self.fence_wait)

    @property
    def serial(self) -> float:
        return self.replay + self.gpu

    def overlapped(self, target: float) -> float:
        return max(self.replay, self.gpu, self.own, target)


def windows(text: str) -> list[Window]:
    gpu = {int(m[1]): (float(m[2]), max(float(m[3]), float(m[4] or 0))) for m in GPU.finditer(text)}
    out = []
    for m in INTERVAL.finditer(text):
        frame = int(m[1])
        if frame in gpu:
            out.append(Window(frame, float(m[2]), int(m[3]), int(m[4]), float(m[5]), *gpu[frame]))
    return out


def target_interval(ws: list[Window]) -> float:
    """The pacing target: the median interval of the paced windows, 60 Hz when the log has none."""
    paced = [w.interval for w in ws if w.pacing_pct >= 50]
    return statistics.median(paced) if paced else 1000 / 60


def report(path: Path, show_windows: bool) -> None:
    ws = windows(path.read_text(errors="replace"))
    target = target_interval(ws)
    held = [w for w in ws if w.pacing_pct < PACED_PCT and w.gpu > 0]
    print(f"{path.name}: {len(ws)} windows, {len(held)} not held by the pacing (target {target:.2f} ms)")
    if not held:
        return
    if show_windows:
        print("  frame  interval  replay   gpu  fence  own  serial  overlapped")
        for w in held:
            print(
                f"  {w.frame:5d}  {w.interval:8.2f}  {w.replay:6.2f} {w.gpu:5.2f}  {w.fence_wait:5.2f} {w.own:4.1f}"
                f"  {w.serial:6.2f}  {w.overlapped(target):10.2f}"
            )

    def med(values: list[float]) -> float:
        return statistics.median(values)

    interval = med([w.interval for w in held])
    serial = med([w.serial for w in held])
    over = med([w.overlapped(target) for w in held])
    err = med([abs(w.interval - w.serial) for w in held])
    print(
        f"  measured   {interval:6.2f} ms ({1000 / interval:5.1f} fps): replay {med([w.replay for w in held]):.2f},"
        f" gpu {med([w.gpu for w in held]):.2f}, fence waits (the longest thread's)"
        f" {med([w.fence_wait for w in held]):.2f}, the presenting thread's own work {med([w.own for w in held]):.2f}"
    )
    print(f"  serial     {serial:6.2f} ms ({1000 / serial:5.1f} fps): off the measured by {err:.2f} ms (median)")
    print(f"  overlapped {over:6.2f} ms ({1000 / over:5.1f} fps): {interval - over:.2f} ms of each frame is waiting")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--windows", action="store_true", help="one row per window, not only the medians")
    args = ap.parse_args()
    for path in args.logs:
        report(path, args.windows)


if __name__ == "__main__":
    main()
