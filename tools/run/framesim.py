"""Frame simulator: replays one run's pipeline events (MKW_SIMTRACE) under other costs or dependencies, in seconds.

Why: from 25 to 28/11/2026 the split-screen work took 157 game runs; 7 were ceilings (one stage made free) and a dozen
asked «what if the game ran ahead». Coz's virtual speedups (Curtsinger and Berger, SOSP 2015) and trace-driven
simulation answer those from one recording: keep each stage's own work as measured, change one cost or one
dependency, and recompute when everything happens.

Model (runtime/core/diag/simtrace.c's events of one run with MKW_GPU_SIM):
  game threads  each a chain in time order of its own work, submits, presents and waits; the work between two events
                keeps its measured length, a wait ends when what it waits for is done: kind 0 the GPU reaching a fence,
                2 the replay issuing it, 3 the replay handling the previous present (the frame bound), 4 the replay
                emptying its queue; a wait on a guest object ends when the wake that ended it comes (that wake is an
                event of the waker's chain), one that timed out lasts as measured.
  replay        one thread taking submits and presents in queue order: each starts at max(queued, previous end) plus
                its measured lag and lasts its measured time (a submit's scaled by --replay-scale, 0 past --free-over
                draws).
  GPU           one job at a time in fence order, as vk_gpu_sim.c: starts at max(first part sent, previous end), costs
                draws x ms per 1000 (--gpu-ms, else the run's) and never ends before the replay issued it. The first
                part goes out at the replay's start plus its measured lag; --overlap sends it at the replay's start.
The pass with nothing changed must give the run's own present intervals back: its error is printed, and a variant whose
change is smaller than that error says nothing.

Limits: waits for a virtual core and for host locks count as work; the CPU the replay and the game share is not
modelled (the replay took +4.9 ms with the game ahead, sim2-ahead35dry); a change that makes the game take another
path (another wait, another submit) is not either: the chains are the measured ones.

Usage: uv run tools/run/framesim.py TRACE [--batch FILE] [--replay-scale K] [--free-over N] [--gpu-ms MS] [--overlap]
                                         [--no-wait 0,2,3,4] [--skip N]
"""

from __future__ import annotations

import argparse
import bisect
import statistics
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Sub:
    fence: int
    t_sub: int = 0
    rb: int = 0
    re: int = 0
    gs: int = 0
    ge: int = 0
    draws: int = 0


@dataclass
class Trace:
    subs: dict[int, Sub] = field(default_factory=dict)
    chains: dict[int, list[tuple[int, str, int, int]]] = field(default_factory=lambda: defaultdict(list))
    rp: dict[int, int] = field(default_factory=dict)   # present number -> when the replay handled it
    ms_per_k: float = 0.0


def read(path: str) -> Trace:
    tr = Trace()
    costs = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if len(p) < 6:
                continue   # a run cut mid-line
            ns, ev, tid, a, b, c = int(p[0]), p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5])
            if ev in ("sub", "rb", "re", "gpu", "gpd"):
                s = tr.subs.setdefault(a, Sub(a))
                if ev == "sub":
                    s.t_sub = ns
                    tr.chains[tid].append((ns, ev, a, b))
                elif ev == "rb":
                    s.rb = ns
                elif ev == "re":
                    s.re = ns
                elif ev == "gpu":
                    s.gs, s.ge = b, c
                else:
                    s.draws = b
                    if b:
                        costs.append(c / b)
            elif ev == "rp":
                tr.rp[a] = ns
            elif ev in ("wb", "we", "pr", "gwb", "gwk"):
                tr.chains[tid].append((ns, ev, a, b))
            elif ev == "gwe":
                tr.chains[tid].append((ns, ev, a, 0 if c else b))   # b: the waker, 0 when it timed out
    tr.ms_per_k = statistics.median(costs) / 1e3 if costs else 0.0
    return tr


@dataclass(frozen=True)
class Opts:
    replay_scale: float = 1.0
    free_over: int | None = None
    gpu_ms: float | None = None
    overlap: bool = False
    no_wait: frozenset[int] = frozenset()


@dataclass
class Item:
    kind: str        # "s" submit, "p" present
    key: int         # fence or present number
    queued: int      # measured
    start: int       # measured
    end: int         # measured


def replay_items(tr: Trace) -> list[Item]:
    pr = {a: ns for ch in tr.chains.values() for ns, ev, a, _ in ch if ev == "pr"}
    items = [Item("s", s.fence, s.t_sub, s.rb, s.re) for s in tr.subs.values() if s.t_sub and s.rb and s.re]
    items += [Item("p", k, pr[k], 0, t) for k, t in tr.rp.items() if k in pr]
    items.sort(key=lambda it: (it.queued, it.kind == "p"))   # queue order (submits and presents share one queue)
    prev = 0
    for it in items:   # a present's start is not traced: it starts when it is queued and the replay is free
        if it.kind == "p":
            it.start = min(max(it.queued, prev), it.end)
        prev = it.end
    return items


def simulate(tr: Trace, o: Opts) -> dict[int, int]:
    """Present times (frame -> ns) under o."""
    items = replay_items(tr)
    ns_per_draw = (o.gpu_ms if o.gpu_ms is not None else tr.ms_per_k) * 1e3
    subs = tr.subs
    # measured lags: replay start after max(queued, previous end); GPU first part after the replay's start
    lag, part = [], {}
    prev_end = prev_ge = 0
    for it in items:
        lag.append(max(0, it.start - max(it.queued, prev_end)))
        prev_end = it.end
    for f in sorted(subs):
        s = subs[f]
        if s.rb and s.re:
            part[f] = (s.gs - s.rb) if s.gs and s.gs > prev_ge + 1000 else (s.re - s.rb)
            prev_ge = s.ge or prev_ge
    queued_new: dict[tuple[str, int], int] = {}
    done_new: dict[tuple[str, int], int] = {}   # ("s", f) -> replay end, ("p", k) -> handled
    issued_new: dict[int, int] = {}
    gpu_new: dict[int, int] = {}
    state = {"next": 0, "r_prev": 0}
    gpu_pending: list[int] = []

    def gpu_run() -> None:   # the GPU in fence order, as far as fences are issued in order
        g_prev = state.get("g_prev", 0)
        for f in sorted(gpu_pending):
            s = subs[f]
            first = state["rb_" + str(f)] if o.overlap else min(state["rb_" + str(f)] + part.get(f, 0), issued_new[f])
            gs = max(first, g_prev)
            g_prev = gpu_new[f] = int(max(gs + s.draws * ns_per_draw, issued_new[f]))
        gpu_pending.clear()
        state["g_prev"] = g_prev

    def replay_until(pred) -> None:
        while state["next"] < len(items) and not pred():
            i = state["next"]
            it = items[i]
            q = queued_new.get((it.kind, it.key))
            if q is None:
                break   # not queued yet in the new schedule
            start = max(q, state["r_prev"]) + lag[i]
            length = it.end - it.start
            if it.kind == "s":
                s = subs[it.key]
                length = 0 if o.free_over is not None and s.draws > o.free_over else length * o.replay_scale
                state["rb_" + str(it.key)] = int(start)
                issued_new[it.key] = int(start + length)
                gpu_pending.append(it.key)
                gpu_run()
            end = int(start + length)
            done_new[(it.kind, it.key)] = end
            state["r_prev"] = end
            state["next"] = i + 1

    queued_sorted = [it.queued for it in items]   # items are in queue order
    events = sorted((ev[0], tid, i) for tid, ch in tr.chains.items() for i, ev in enumerate(ch))
    last_meas: dict[int, int] = {}
    last_new: dict[int, int] = {}
    pres: dict[int, int] = {}
    wait_meas: dict[int, int] = {}
    wakes: dict[int, list[tuple[int, int]]] = defaultdict(list)   # guest thread -> (measured, new) of its wakes
    for t, tid, i in events:
        _, ev, a, b = tr.chains[tid][i]
        if ev == "gwe":
            start_new = last_new[tid]
            w = wakes.get(a)
            if b and w:
                k = bisect.bisect_right(w, (t, float("inf"))) - 1
                # the wake latency as measured (the waker's signal to the waiter running again)
                new = max(start_new, w[k][1] + max(0, t - w[k][0])) if k >= 0 else start_new + (t - wait_meas.get(tid, t))
            else:
                new = start_new + (t - wait_meas.get(tid, t))
        elif ev == "we":
            start_new = last_new[tid]
            if b in o.no_wait:
                new = start_new
            elif b == 0 and a in subs and subs[a].rb:
                replay_until(lambda a=a: a in gpu_new)
                late = max(0, t - subs[a].ge) if subs[a].ge else 0   # the fence seen done after the GPU's end
                new = max(start_new, gpu_new.get(a, start_new) + late)
            elif b == 2 and a in subs:
                replay_until(lambda a=a: a in issued_new)
                new = max(start_new, issued_new.get(a, start_new) + max(0, t - subs[a].re))
            elif b == 3:
                replay_until(lambda a=a: ("p", a) in done_new)
                new = max(start_new, done_new.get(("p", a), start_new) + max(0, t - tr.rp.get(a, t)))
            elif b == 4:
                n_before = bisect.bisect_left(queued_sorted, wait_meas.get(tid, t))
                replay_until(lambda n=n_before: state["next"] >= n)
                new = max(start_new, state["r_prev"])
            else:   # a fence this trace does not place (a present's): as long as it was measured
                new = start_new + (t - wait_meas.get(tid, t))
        else:
            prev_m = last_meas.get(tid)
            new = last_new[tid] + (t - prev_m) if prev_m is not None else t
            if ev == "sub":
                queued_new[("s", a)] = new
            elif ev == "pr":
                pres[a] = new
                queued_new[("p", a)] = new
            elif ev in ("wb", "gwb"):
                wait_meas[tid] = t
            elif ev == "gwk":
                wakes[a].append((t, new))
        last_meas[tid], last_new[tid] = t, new
    return pres


def measured_presents(tr: Trace) -> dict[int, int]:
    return {a: ns for ch in tr.chains.values() for ns, ev, a, _ in ch if ev == "pr"}


def intervals(pres: dict[int, int], skip: int) -> list[float]:
    fr = sorted(pres)[skip:]
    return [(pres[b] - pres[a]) / 1e6 for a, b in zip(fr, fr[1:]) if b == a + 1]


def summary(xs: list[float]) -> str:
    if not xs:
        return "no frames"
    return f"median {statistics.median(xs):.2f} ms, mean {statistics.fmean(xs):.2f} ({len(xs)} frames)"


def variant_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--replay-scale", type=float, default=1.0)
    ap.add_argument("--free-over", type=int, help="the replay of submits with more draws than this costs nothing")
    ap.add_argument("--gpu-ms", type=float, help="simulated GPU ms per 1000 draws (default: the run's)")
    ap.add_argument("--overlap", action="store_true", help="the GPU may start when the replay starts a submit")
    ap.add_argument("--no-wait", default="", help="wait kinds that no longer wait (0, 2, 3, 4)")


def opts(a: argparse.Namespace) -> Opts:
    nw = frozenset(int(x) for x in a.no_wait.split(",") if x)
    return Opts(a.replay_scale, a.free_over, a.gpu_ms, a.overlap, nw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("trace")
    variant_args(ap)
    ap.add_argument("--skip", type=int, default=30, help="presents left out at the start")
    ap.add_argument("--batch", help="a file with one variant's options per line, all on one read of the trace")
    a = ap.parse_args(argv)
    tr = read(a.trace)
    meas = intervals(measured_presents(tr), a.skip)
    base = intervals(simulate(tr, Opts()), a.skip)
    o = opts(a)
    print(f"measured  {summary(meas)}")
    if not (base and meas):
        print("model: no frames")
        return 1
    print(f"model     {summary(base)}  (its error on its own run: "
          f"{statistics.median(base) - statistics.median(meas):+.2f} ms median)")
    todo = [("", o)] if o != Opts() else []
    if a.batch:
        vp = argparse.ArgumentParser()
        variant_args(vp)
        for line in open(a.batch).read().splitlines():
            if line.strip() and not line.startswith("#"):
                todo.append((line.strip(), opts(vp.parse_args(line.split()))))
    for label, v in todo:
        var = intervals(simulate(tr, v), a.skip)
        if var:
            print(f"variant   {summary(var)}  (variant - model {statistics.median(var) - statistics.median(base):+.2f} ms)"
                  + (f"  [{label}]" if label else ""))
    print(f"GPU {tr.ms_per_k:.2f} ms per 1000 draws in the run; {len(tr.subs)} submits, {len(tr.rp)} presents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
