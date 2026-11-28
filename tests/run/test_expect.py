"""tools/runner/expect.py: the expected range read from MKW_RUN_WHY, the always-valid interval, the refusal of an A/B
too coarse for its expectation and the decision that stops it.
Why: the A/B gate and stop act on every measured run; a wrong parse refuses good runs or lets blind ones through.

Usage: uv run pytest tests/run/test_expect.py
"""

import math
import random

import expect


def test_parse_range_and_single():
    assert expect.parse("x; expect=-4..-2 ms; y") == expect.Expect(-4, -2)
    assert expect.parse("expect=-2,5..−1") == expect.Expect(-2.5, -1)
    assert expect.parse("expect=-2") == expect.Expect(-3, -1)
    assert expect.parse("espero algo") is None
    assert expect.parse(None) is None


def test_cs_wider_than_t_and_shrinks():
    assert expect.cs_half(1.0, 10) > 1.96 / math.sqrt(10)
    assert expect.cs_half(1.0, 100) < expect.cs_half(1.0, 10)
    assert expect.cs_half(0, 10) == math.inf


def test_too_fine():
    assert expect.too_fine(expect.Expect(-0.6, -0.2), 1.0)
    assert expect.too_fine(expect.Expect(-4, -2), 1.0) is None
    assert expect.too_fine(expect.Expect(-0.6, -0.2), None) is None


def test_decided_hit_miss_open():
    rng = random.Random(1)
    e = expect.Expect(-4, -2)
    hit = [-3 + rng.gauss(0, 1) for _ in range(60)]
    assert expect.decided(e, hit)[0] == "hit"
    miss = [rng.gauss(0, 1) for _ in range(60)]
    assert expect.decided(e, miss)[0] == "miss"
    assert expect.decided(e, [-3 + rng.gauss(0, 8) for _ in range(10)]) is None
