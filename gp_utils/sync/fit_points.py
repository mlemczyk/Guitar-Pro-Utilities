"""Place sync points by FITTING the drift curve, not by sampling a bar grid.

THE AUTHOR'S OWN METHOD, WHICH IS A DIFFERENT ALGORITHM (2026-09-24): *"I aligned
the first drum hit, and then slightly adjusted the synch points whenever the
large transients drifted off the bar grid."* Anchor once, then add a breakpoint
only where the drift has grown enough to see. Their Sultans of Swing map spaces
points 2 to 44 bars apart -- dense where the tempo moved, sparse where it did
not.

WHY SAMPLING A FIXED GRID CANNOT MATCH THAT, measured over 115 ground-truth
tabs. Guitar Pro plays the audio at the tempo implied between consecutive sync
points, so the SPACING between neighbours matters as much as each position, and
tempo is a *difference* -- it amplifies whatever local noise the DTW map has:

    every  4 bars : position 82ms, implied-tempo SD 6.69
    every  8 bars : position 117ms, SD 4.09
    every 16 bars : position 166ms, SD 2.97
    the humans    :                 SD 0.70

Every density is 4-10x noisier than a human's, and buying tempo stability by
thinning the grid costs position linearly. The grid is the problem: each sampled
point inherits the map's error at that instant, and differencing two of them
doubles it.

A bounded-error piecewise-linear fit inverts that. Each segment is a straight
line through many hundreds of map samples, so its slope -- the implied tempo --
is an average rather than a difference, and a breakpoint appears only where one
line can no longer hold the whole span inside tolerance. Fewer points, smoother
tempo, and the error is bounded by construction rather than hoped for.

⚠⚠ THIS MODULE IS NOT WIRED INTO THE PIPELINE, AND MUST NOT BE UNTIL IT IS
FIXED. It does what it was built for on the song it was built from and fails
library-wide, which is the exact trap this project has now walked into three
times in one session. Measured on "Sultans of swing" against the author's own map
it beats the shipped grid on every axis at once -- 54 points to 13, error median
37ms to 20ms, worst 214ms to 104ms, tempo SD 3.48 to 2.03. Measured over 115
ground-truth tabs:

    variant           pts   err med   err max   implied-tempo SD
    grid 4/no snap     36      82ms     406ms       6.69
    fit tol 150ms      19      93ms     331ms      15.17    <- much WORSE
    fit tol 250ms      11     114ms     335ms      17.74

It buys a better worst case and pays more than double in tempo jitter. The
likely cause is its own variable segmentation: where drift is fast it emits
short segments, and a two-bar segment's implied tempo is computed over so few
quarters that it is wilder than anything the fixed grid produces. `min_segment_
bars` counts BARS, which is the wrong unit -- it should bound the span in
QUARTERS, or better in seconds of audio. The breakpoint times are also the mean
of two adjacent lines' predictions, so the tempo recomputed between emitted
points is not the fitted slope, which throws away much of the averaging.

Fix those two before measuring again, and measure on the full set, not on
Sultans.
"""

from __future__ import annotations

import numpy as np

from .align import Alignment
from .gp_score import Score

FIT_TOLERANCE_SEC = 0.050      # max |residual| a single segment may leave
MIN_SEGMENT_BARS = 2           # never emit two breakpoints closer than this


def first_sounding_bar(score: Score) -> int:
    """The bar to anchor on: the first with any notated note, skipping a pickup.

    The generated maps anchored on playback bar 0 unconditionally, and on Sultans
    of Swing that is a **1/4 pickup bar 0.40s long** -- a quarter-note stub at
    the very head of the file, which is the worst available anchor and not where
    the author put theirs (bar 2). An anchor wants a bar with real content in it.
    """
    if not score.notes:
        return 0
    first_note = min(n["start"] for n in score.notes)
    for bar in score.bars:
        if bar.end > first_note + 1e-6:
            # Prefer a full-length bar: a pickup is shorter than its neighbour.
            idx = bar.playback_idx
            if (idx + 1 < len(score.bars)
                    and (bar.end - bar.start) < 0.75 * (score.bars[idx + 1].end
                                                        - score.bars[idx + 1].start)):
                return min(idx + 1, len(score.bars) - 1)
            return idx
    return 0


def fit_sync_points(score: Score, alignment: Alignment,
                    tolerance_sec: float = FIT_TOLERANCE_SEC,
                    min_segment_bars: int = MIN_SEGMENT_BARS,
                    max_points: int | None = None) -> list[tuple[int, float]]:
    """Breakpoints of a piecewise-linear fit to the score->audio map.

    Greedy and bounded: from the current breakpoint, extend the segment to the
    furthest bar line whose straight line keeps every map sample in the span
    within `tolerance_sec`, then break there and continue. Breakpoints land on
    bar lines because that is the only place Guitar Pro can hold one.
    """
    bars = score.bars
    if len(bars) < 2:
        return []

    start_bar = first_sounding_bar(score)
    bar_starts = np.array([b.start for b in bars], dtype=float)

    # The map, densely sampled -- this is what each segment is fitted through.
    map_st = np.asarray(alignment.score_times, dtype=float)
    map_at = np.asarray(alignment.audio_times, dtype=float)
    if len(map_st) < 2:
        return []

    def audio_at(t):
        return np.interp(t, map_st, map_at)

    def residual(i0: int, i1: int) -> float:
        """Worst |error| if one straight line spans bars i0..i1."""
        t0, t1 = bar_starts[i0], bar_starts[i1]
        a0, a1 = audio_at(t0), audio_at(t1)
        sel = (map_st >= t0) & (map_st <= t1)
        if sel.sum() < 3 or t1 <= t0:
            return 0.0
        line = a0 + (map_st[sel] - t0) * (a1 - a0) / (t1 - t0)
        return float(np.max(np.abs(line - map_at[sel])))

    breaks = [start_bar]
    cur = start_bar
    last = len(bars) - 1
    while cur < last:
        # Exponential probe then binary search for the furthest acceptable bar.
        lo = min(cur + min_segment_bars, last)
        if residual(cur, lo) > tolerance_sec:
            cur = lo
            breaks.append(cur)
            continue
        step = min_segment_bars
        hi = lo
        while hi < last and residual(cur, min(hi + step, last)) <= tolerance_sec:
            lo = min(hi + step, last)
            hi = lo
            step *= 2
        left, right = lo, min(hi + step, last)
        while left + 1 < right:
            mid = (left + right) // 2
            if residual(cur, mid) <= tolerance_sec:
                left = mid
            else:
                right = mid
        cur = max(left, min(cur + min_segment_bars, last))
        breaks.append(cur)

    if breaks[-1] != last:
        breaks.append(last)
    breaks = sorted(set(breaks))

    # If a cap is asked for, drop the breakpoints whose removal costs least.
    while max_points and len(breaks) > max_points and len(breaks) > 2:
        costs = [residual(breaks[k - 1], breaks[k + 1]) for k in range(1, len(breaks) - 1)]
        breaks.pop(1 + int(np.argmin(costs)))

    times = _least_squares_breakpoints(breaks, bar_starts, map_st, map_at)
    return [(int(b), float(t)) for b, t in zip(breaks, times)]


def _least_squares_breakpoints(breaks, bar_starts, map_st, map_at) -> list[float]:
    """Breakpoint times from a least-squares line per segment, not endpoints.

    ⚠ THIS IS WHAT FIXES THE IMPLIED-TEMPO JITTER, AND THE CAUSE WAS NOT NOISE.
    `align()` pools frames to bound the DTW cost matrix, so the map's audio times
    land on a grid -- ~93ms for a 343s score. The map is otherwise perfectly
    smooth: measured against its own 4-second median filter its deviation is
    **0ms**, which is why smoothing it changed nothing at all.

    But a segment's implied tempo is the DIFFERENCE of its two endpoints, so
    reading them straight off that grid injects +-93ms into a span of a few
    seconds -- +-1.5% of tempo over 4 bars, which is the +-2 BPM wobble observed.
    Fitting each segment by least squares over the hundreds of map samples inside
    it averages the quantisation away instead of sampling it twice.

    Interior breakpoints take the mean of the two adjacent lines' predictions, so
    the result stays continuous to within the lines' own disagreement.
    """
    lines = []
    for i0, i1 in zip(breaks, breaks[1:]):
        t0, t1 = bar_starts[i0], bar_starts[i1]
        sel = (map_st >= t0) & (map_st <= t1)
        if sel.sum() >= 3 and t1 > t0:
            slope, intercept = np.polyfit(map_st[sel], map_at[sel], 1)
        else:
            a0 = float(np.interp(t0, map_st, map_at))
            a1 = float(np.interp(t1, map_st, map_at))
            slope = (a1 - a0) / (t1 - t0) if t1 > t0 else 1.0
            intercept = a0 - slope * t0
        lines.append((slope, intercept))

    out = []
    for k, b in enumerate(breaks):
        t = bar_starts[b]
        preds = []
        if k > 0:
            s, c = lines[k - 1]
            preds.append(s * t + c)
        if k < len(lines):
            s, c = lines[k]
            preds.append(s * t + c)
        out.append(float(np.mean(preds)) if preds else float(np.interp(t, map_st, map_at)))

    return list(np.maximum.accumulate(out))     # a sync map must not run backwards
