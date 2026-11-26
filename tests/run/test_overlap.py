"""The frame-time model of tools/run/overlap.py on synthetic perf windows with a known answer: a window where the
replay follows the GPU reads as serial and gains the wait back, a paced window is left out, and a window bound by the
presenting thread's own work gains nothing.

Usage: uv run pytest tests/run/test_overlap.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/run"))

import overlap  # noqa: E402


def window(frame: int, interval: float, pacing: int, fence: int, replay: float, gpu: float, wait: float) -> str:
    return (
        f"[host] perf: frame {frame}: interval avg {interval} ms max 20.00 ms (50.0 fps); render thread blocked: "
        f"pacing {pacing}% fence {fence}% drain 1%; render busy avg 1.00 ms; replay avg {replay} ms; ab 0/60\n"
        f"[host] perf: frame {frame}: gpu avg {gpu} ms (10 parts timed, 0 not); fence wait per frame: render thread "
        f"{wait} ms, other threads 0.00 ms; bound: cpu\n"
    )


def test_serial_window_gains_the_wait() -> None:
    text = window(60, 16.67, 93, 0, 5.0, 6.0, 0.0) + window(120, 18.4, 0, 80, 10.0, 8.4, 14.7)
    ws = overlap.windows(text)
    assert [w.frame for w in ws] == [60, 120]
    target = overlap.target_interval(ws)
    assert abs(target - 16.67) < 1e-9
    held = [w for w in ws if w.pacing_pct < overlap.PACED_PCT]
    assert [w.frame for w in held] == [120]
    assert abs(held[0].serial - 18.4) < 1e-9
    assert held[0].overlapped(target) == target


def test_window_bound_by_its_own_work_gains_nothing() -> None:
    (w,) = overlap.windows(window(60, 20.0, 0, 0, 9.5, 6.5, 0.0))
    assert w.own == 20.0
    assert w.overlapped(1000 / 60) == 20.0


def test_a_log_without_perf_windows_is_empty() -> None:
    assert overlap.windows("[host] nvn: nothing here\n") == []
    assert abs(overlap.target_interval([]) - 1000 / 60) < 1e-9
