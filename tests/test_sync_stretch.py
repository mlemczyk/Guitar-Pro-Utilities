"""Tests for the stretch measure and the refinement penalty built on it.

WHY THESE EXIST. The change these cover is subtle in the way that does not show up
in a batch average: `grid_align.refine_to_grid` used to penalise roughness in the
audio's raw tempo, which charges a tab for a tempo change it notates. Dividing by
the notated tempo first fixes that, and the SAFETY ARGUMENT for the change is an
algebraic identity -- on a tab of constant notated tempo the two penalties are the
same function, because scaling a, b and c by one constant leaves
abs(c - 2b + a)/b unchanged.

That identity is what `test_penalty_identical_when_tempo_constant` pins. It is
worth a test rather than a comment because it is the reason the change was safe to
ship against 271 files at once, and a future edit to `tempo_of` could break it
silently -- the batch A/B that proved it costs six minutes, this costs no time.

Run with `python -m pytest`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gp_utils.sync import grid_align
from gp_utils.sync.tempo_lock import (implied_tempo, notated_tempo, stretch_anomaly,
                        stretch_ratio)


class FakeBar:
    """The two fields the tempo arithmetic needs: where a bar starts, how long."""

    def __init__(self, playback_idx: int, start: float, quarters: float):
        self.playback_idx = playback_idx
        self.start = start
        self.quarters = quarters


class FakeScore:
    def __init__(self, bars):
        self.bars = bars


def make_score(bar_quarters, bar_seconds):
    """Bars of given length in quarters, laid out at given durations in seconds.

    Keeping the two independent is the whole point: `bar_seconds` encodes the
    NOTATED tempo, so a score whose bars are all 4 quarters but not all the same
    number of seconds is a score with a notated tempo change.
    """
    bars, t = [], 0.0
    for i, (q, sec) in enumerate(zip(bar_quarters, bar_seconds)):
        bars.append(FakeBar(i, t, q))
        t += sec
    bars.append(FakeBar(len(bars), t, 4.0))     # a terminator to span the last point
    return FakeScore(bars)


# --------------------------------------------------------------------------
# stretch_ratio
# --------------------------------------------------------------------------

def test_stretch_is_one_when_audio_matches_the_notation():
    score = make_score([4] * 8, [2.0] * 8)         # 4/4 at 120 BPM throughout
    pts = [(0, 0.0), (2, 4.0), (4, 8.0), (6, 12.0)]
    assert np.allclose(stretch_ratio(pts, score), 1.0)


def test_stretch_is_flat_across_a_notated_tempo_change():
    """The property the whole investigation turned on.

    Bars 0-3 are 2.0s (120 BPM), bars 4-7 are 1.0s (240 BPM). An audio track that
    plays this correctly has an implied tempo that DOUBLES and a stretch that does
    not move -- so a quality measure built on stretch exonerates the map that a
    measure built on tempo condemns.
    """
    score = make_score([4] * 8, [2.0] * 4 + [1.0] * 4)
    pts = [(0, 0.0), (2, 4.0), (4, 8.0), (6, 10.0)]
    imp = implied_tempo(pts, score)
    nom = notated_tempo(pts, score)
    assert imp.max() / imp.min() == pytest.approx(2.0, abs=0.01)
    assert nom.max() / nom.min() == pytest.approx(2.0, abs=0.01)
    assert np.allclose(stretch_ratio(pts, score), 1.0)


def test_stretch_catches_a_map_that_really_is_wrong():
    """Same notation, but the middle segment takes twice as long as it should."""
    score = make_score([4] * 8, [2.0] * 8)
    pts = [(0, 0.0), (2, 4.0), (4, 12.0), (6, 16.0)]      # middle span 8s, not 4s
    s = stretch_ratio(pts, score)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.5)
    assert s[2] == pytest.approx(1.0)


def test_stretch_and_notated_drop_the_same_segments():
    """They must stay element-wise aligned, or dividing them silently misaligns."""
    score = make_score([4] * 8, [2.0] * 8)
    pts = [(0, 0.0), (2, 4.0), (2, 4.0), (4, 8.0), (6, 12.0)]   # a repeated point
    assert len(implied_tempo(pts, score)) == len(notated_tempo(pts, score))
    assert len(stretch_ratio(pts, score)) == len(implied_tempo(pts, score))


# --------------------------------------------------------------------------
# The penalty change in refine_to_grid
# --------------------------------------------------------------------------

def _roughness(score, pts, use_stretch: bool) -> list[float]:
    """The per-segment quantity refine_to_grid differences, via its own code path.

    Reaching into `refine_to_grid` is deliberate: testing a reimplementation of
    `tempo_of` would pass while the real one was broken.
    """
    captured = {}
    real = grid_align._segment_energy

    def spy(score_, i0, t0, i1, t1, env, env_times, subdivision=1.0):
        return real(score_, i0, t0, i1, t1, env, env_times, subdivision)

    grid_align._segment_energy = spy
    try:
        # A flat envelope makes every nudge energy-neutral, so whatever the
        # refinement does is driven by the penalty alone.
        env = np.ones(4000)
        et = np.arange(4000) * 0.01
        out = grid_align.refine_to_grid(score, pts, env, et, use_stretch=use_stretch)
    finally:
        grid_align._segment_energy = real
    captured  # kept for clarity; the return value is what we compare
    return [t for _, t in out]


def test_penalty_identical_when_tempo_constant():
    """THE SAFETY ARGUMENT. Constant notated tempo => the two arms are one function.

    Measured over the real library this held exactly: 271 of 271 constant-tempo
    tabs moved by +0.00 gridline points. This is that claim in miniature.
    """
    score = make_score([4] * 24, [2.0] * 24)
    pts = [(0, 0.05), (4, 8.10), (8, 15.95), (12, 24.05), (16, 31.90), (20, 40.0)]
    old, new = _roughness(score, pts, False), _roughness(score, pts, True)
    # Non-vacuity first: two arms that both did nothing would agree for free, and
    # the test would keep passing after the mechanism it guards had been removed.
    assert new != [t for _, t in pts], "refinement moved nothing; test is vacuous"
    assert old == new


def test_penalty_differs_across_a_notated_tempo_change():
    """And where the tab DOES change tempo, the two arms must part company.

    If this passes trivially the change would be a no-op everywhere, which the
    batch A/B would have reported as "no effect" rather than as a fix.
    """
    score = make_score([4] * 24, [2.0] * 12 + [1.0] * 12)
    pts = [(0, 0.05), (4, 8.10), (8, 15.95), (12, 24.05), (16, 28.10), (20, 32.0)]
    assert _roughness(score, pts, False) != _roughness(score, pts, True)


# --------------------------------------------------------------------------
# stretch_anomaly
# --------------------------------------------------------------------------

def test_anomaly_clean_on_a_good_map():
    score = make_score([4] * 16, [2.0] * 16)
    pts = [(0, 0.0), (2, 4.0), (4, 8.0), (6, 12.0), (8, 16.0), (10, 20.0)]
    assert stretch_anomaly(score, pts)["verdict"] == "clean"


def test_anomaly_clean_across_a_notated_tempo_change():
    """The regression this replaces: the old detector called this 'structural'."""
    score = make_score([4] * 16, [2.0] * 8 + [1.0] * 8)
    pts = [(0, 0.0), (2, 4.0), (4, 8.0), (6, 12.0), (8, 16.0), (10, 18.0),
           (12, 20.0), (14, 22.0)]
    an = stretch_anomaly(score, pts)
    assert an["verdict"] == "clean"
    assert an["ref_stretch"] == pytest.approx(1.0, abs=0.01)


def test_anomaly_flags_a_structural_run():
    """Three or more consecutive interior segments out of band."""
    score = make_score([4] * 20, [2.0] * 20)
    t = [0.0, 4.0, 8.0]
    for _ in range(3):
        t.append(t[-1] + 8.0)          # three segments at half speed
    t.append(t[-1] + 4.0)
    t.append(t[-1] + 4.0)
    pts = [(2 * k, v) for k, v in enumerate(t)]
    an = stretch_anomaly(score, pts)
    assert an["verdict"] == "structural"
    assert an["longest_run"] >= 3


def test_anomaly_flags_edge_spread_only_at_the_ends():
    score = make_score([4] * 20, [2.0] * 20)
    t = [0.0, 7.0]                      # a slow lead-in
    for _ in range(6):
        t.append(t[-1] + 4.0)
    t.append(t[-1] + 7.5)               # and a slow fade
    pts = [(2 * k, v) for k, v in enumerate(t)]
    an = stretch_anomaly(score, pts)
    assert an["verdict"] == "edge_spread"
    assert an["head_bad"] >= 1 and an["tail_bad"] >= 1


def test_anomaly_too_few_points():
    score = make_score([4] * 8, [2.0] * 8)
    assert stretch_anomaly(score, [(0, 0.0), (2, 4.0)])["verdict"] == "too_few_points"
