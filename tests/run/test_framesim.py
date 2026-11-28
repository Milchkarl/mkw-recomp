"""tools/run/framesim.py on a synthetic trace with a known answer: a serial pipeline (the game waits for the GPU to finish
its frame before the next) is given back as measured with nothing changed, and making the replay free or letting the
GPU start with the replay shortens it by what the arithmetic says.
Why: the simulator stands in for game runs; it must reproduce its own run before a variant from it means anything.

Usage: uv run pytest tests/run/test_framesim.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/run"))

import framesim  # noqa: E402

MS = 1_000_000
GAME, REPLAY, GPU, DRAWS = 4 * MS, 6 * MS, 5 * MS, 1000   # GPU: 5 ms for 1000 draws


def serial_trace(path: Path, frames: int = 80) -> None:
    """Game: work, submit f, present f, wait for the GPU on f. Replay starts at the submit; the GPU after the replay."""
    rows = ["ns\tevent\ttid\ta\tb\tc"]
    t = 0
    for f in range(1, frames + 1):
        t += GAME
        rows.append(f"{t}\tsub\t1\t{f}\t1\t0")
        rows.append(f"{t}\trb\t2\t{f}\t1\t0")
        re = t + REPLAY
        rows.append(f"{re}\tre\t2\t{f}\t1\t0")
        ge = re + GPU
        rows.append(f"{re}\tgpu\t2\t{f}\t{re}\t{ge}")
        rows.append(f"{re}\tgpd\t2\t{f}\t{DRAWS}\t{GPU}")
        rows.append(f"{t + 1}\tpr\t1\t{f}\t0\t0")
        rows.append(f"{re + 1}\trp\t2\t{f}\t0\t0")
        rows.append(f"{t + 3}\twb\t1\t{f}\t0\t0")
        rows.append(f"{ge}\twe\t1\t{f}\t0\t0")
        t = ge
    rows.sort(key=lambda r: int(r.split("\t")[0]) if r[0].isdigit() else -1)
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def trace(tmp_path: Path) -> framesim.Trace:
    p = tmp_path / "st.tsv"
    serial_trace(p)
    return framesim.read(str(p))


def median(xs):
    return sorted(xs)[len(xs) // 2]


def test_reproduces_its_run(trace):
    meas = framesim.intervals(framesim.measured_presents(trace), 5)
    base = framesim.intervals(framesim.simulate(trace, framesim.Opts()), 5)
    assert median(meas) == pytest.approx(15.0, abs=0.01)
    assert median(base) == pytest.approx(median(meas), abs=0.01)


def test_free_replay_and_overlap(trace):
    free = framesim.intervals(framesim.simulate(trace, framesim.Opts(replay_scale=0)), 5)
    assert median(free) == pytest.approx(9.0, abs=0.01)   # game 4 + GPU 5
    half_gpu = framesim.intervals(framesim.simulate(trace, framesim.Opts(gpu_ms=2.5)), 5)
    assert median(half_gpu) == pytest.approx(12.5, abs=0.01)
    no_wait = framesim.intervals(framesim.simulate(trace, framesim.Opts(no_wait=frozenset({0}))), 5)
    assert median(no_wait) < 15.0


def test_guest_wait_carries_the_change(tmp_path: Path):
    """The presenting thread (4) only waits for the main thread (1) to wake it after the GPU wait: the free replay must
    reach its presents through that guest wait."""
    rows = ["ns\tevent\ttid\ta\tb\tc"]
    t = 0
    for f in range(1, 60):
        t += GAME
        re, ge = t + REPLAY, t + REPLAY + GPU
        rows += [f"{t}\tsub\t1\t{f}\t1\t0", f"{t}\trb\t2\t{f}\t1\t0", f"{re}\tre\t2\t{f}\t1\t0",
                 f"{re}\tgpu\t2\t{f}\t{re}\t{ge}", f"{re}\tgpd\t2\t{f}\t{DRAWS}\t{GPU}",
                 f"{t + 1}\twb\t1\t{f}\t0\t0", f"{ge}\twe\t1\t{f}\t0\t0",
                 f"{t + 2}\tgwb\t4\t40\t0\t0", f"{ge + 10}\tgwk\t1\t40\t10\t0", f"{ge + 20}\tgwe\t4\t40\t10\t0",
                 f"{ge + 30}\tpr\t4\t{f}\t0\t0", f"{ge + 40}\trp\t2\t{f}\t0\t0"]
        t = ge + 10
    rows.sort(key=lambda r: int(r.split("\t")[0]) if r[0].isdigit() else -1)
    p = tmp_path / "g.tsv"
    p.write_text("\n".join(rows) + "\n")
    tr = framesim.read(str(p))
    base = framesim.intervals(framesim.simulate(tr, framesim.Opts()), 5)
    free = framesim.intervals(framesim.simulate(tr, framesim.Opts(replay_scale=0)), 5)
    assert median(base) == pytest.approx(15.0, abs=0.01)
    assert median(free) == pytest.approx(9.0, abs=0.01)
