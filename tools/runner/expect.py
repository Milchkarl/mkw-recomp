"""A run's expected result, written before it starts, and the A/B's decision against it.

Why: from 25 to 28/11/2026 there were 157 game runs; of the 114 split-screen measurements, 34 were A/Bs that showed
nothing and 10 did not count. A run that says beforehand what number it expects can be refused when it cannot tell
that number from zero, stopped as soon as its result is decided, and scored afterwards (hit or miss), so the next
prediction is made knowing how good the last ones were.

The expectation goes in MKW_RUN_WHY as `expect=LO..HI` (or `expect=X`), B - A in ms per frame on MKW_AB_METRIC, e.g.
`MKW_RUN_WHY="... expect=-4..-2 ..."`. A single number X stands for the range from X/2 to 3X/2.

Ledger: `python3 tools/runner/expect.py [build/run]` lists every run with an expectation, its judge and the hit rate.

The stop uses an always-valid 95 % interval (a normal-mixture confidence sequence, Howard et al. 2021, «Time-uniform,
nonparametric, nonasymptotic confidence sequences», eq. 14): unlike the t interval, it may be looked at after every
block without its error growing with the looks, which is what a run that stops itself does (Johari et al. 2017,
arxiv 1512.04922). The pairs' spread stands in for the true one, as in abstat.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

NUM = r"[-+−]?\d+(?:[.,]\d+)?"
EXPECT = re.compile(rf"expect\s*=\s*({NUM})(?:\s*\.\.\s*({NUM}))?")
RHO = 10.0     # the mixture's width, in pairs: the interval is tightest around this many (runs stop at 10-60 pairs)
ALPHA = 0.05


@dataclass
class Expect:
    lo: float
    hi: float

    def text(self) -> str:
        return f"{self.lo:+.2f}..{self.hi:+.2f} ms"


def num(s: str) -> float:
    return float(s.replace("−", "-").replace(",", "."))


def parse(why: str | None) -> Expect | None:
    m = EXPECT.search(why or "")
    if not m:
        return None
    a = num(m.group(1))
    if m.group(2) is None:
        lo, hi = a / 2, a * 3 / 2
    else:
        lo, hi = a, num(m.group(2))
    return Expect(min(lo, hi), max(lo, hi))


def cs_half(sd: float, n: int, rho: float = RHO, alpha: float = ALPHA) -> float:
    """Half-width of the always-valid interval for the mean of n pairs with spread sd."""
    if n <= 0 or sd <= 0:
        return math.inf
    return sd * math.sqrt(2 * (n + rho) / n**2 * math.log(math.sqrt((n + rho) / rho) / alpha))


def judge(e: Expect, mean: float, half: float) -> str:
    """hit: the interval lies inside the expected range; miss: it lies wholly outside; open: neither yet."""
    lo, hi = mean - half, mean + half
    if lo >= e.lo and hi <= e.hi:
        return "hit"
    if hi < e.lo or lo > e.hi:
        return "miss"
    return "open"


def decided(e: Expect, diffs: list[float]) -> tuple[str, float, float] | None:
    """(hit or miss, mean, half) once the always-valid interval settles it; None while open."""
    n = len(diffs)
    if n < 3:
        return None
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    half = cs_half(sd, n)
    j = judge(e, mean, half)
    return (j, mean, half) if j != "open" else None


def ledger(run_dir: str = "build/run") -> list[str]:
    """Every NAME.json with an expectation: started, name, range, judge; and the hit rate of the decided ones."""
    import json
    from pathlib import Path

    rows, n_hit, n_dec = [], 0, 0
    for p in sorted(Path(run_dir).glob("*.json"), key=lambda q: q.stat().st_mtime):
        try:
            ab = json.loads(p.read_text()).get("ab") or {}
        except (OSError, ValueError):
            continue
        ex = ab.get("expect")
        if not ex:
            continue
        rows.append(f"{p.stem}\t{ex['judge']}\t{ex['text']}")
        if ex["judge"] != "open":
            n_dec += 1
            n_hit += ex["judge"] == "hit"
    rows.append(f"hits {n_hit} of {n_dec} decided ({len(rows)} with an expectation)")
    return rows


def too_fine(e: Expect, target: float | None) -> str | None:
    """A refusal when the run cannot tell the expected effect from zero at its precision (MKW_AB_TARGET)."""
    if target is None:
        return None
    biggest = max(abs(e.lo), abs(e.hi))
    if biggest < 2 * target:
        return (f"expect={e.text()} is under twice the A/B's precision (MKW_AB_TARGET ±{target:g} ms): the run cannot "
                "tell it from zero; answer it off the game (capture, simulator, a player's log) or lower the target")
    return None


if __name__ == "__main__":
    import sys

    print("\n".join(ledger(*sys.argv[1:2])))
