"""Make the backing track play at a steady tempo, which is what a player needs.

THE OBJECTIVE, STATED BY THE AUTHOR (2026-09-24): *"Work on the wobble, it's very
important for me as a bass player to be locked in with the drums."* They record
covers by playing along to Guitar Pro's playback, and Guitar Pro plays the
embedded audio at the tempo implied between consecutive sync points. So an
unsteady implied tempo moves the drums underneath the player. A map whose every
point sits within 40ms can still be unusable.

WHY SMOOTHING THE POSITIONS DOES NOT WORK, and what does. The implied tempo is a
*difference* of neighbouring point times, so it is the DERIVATIVE of the map that
has to be smooth, not the map itself. Two things were tried and refuted first:

  * the map is already smooth -- its deviation from its own 4-second median
    filter is **0ms**, and median-filtering it at 1/2/4/8s changes the output not
    at all;
  * frame quantisation is real but minor -- on a random sample the grid is
    already 46ms and raising `MAX_DTW_FRAMES` from 7000 to 24000 moved the
    implied-tempo SD 8.51 -> 8.42.

What remains is that the map's local accuracy (~111ms median) is differenced over
a few-second span, which turns 111ms into several BPM no matter how the points
are placed. Thinning the grid helps and costs position linearly: every 4 bars
gives SD 6.69 at 82ms, every 16 gives 2.97 at 166ms, and a human gets 0.41-0.70.

So the fix is to REGULARISE THE DRIFT CURVE'S SECOND DERIVATIVE. Writing the
drift as

    d(t) = audio_time(t) - score_time(t)

the implied tempo ratio over a span is Dscore/Daudio = 1/(1 + d'), so a smooth
d' is exactly a smooth tempo. Penalising the second difference of d gives that
directly, at a cost in position that is paid where -- and only where -- the raw
map was wiggling faster than a band can follow:

    minimise  ||d_s - d||^2  +  lambda * ||D2 d_s||^2

which is a banded symmetric positive-definite solve, so it is fast and exact.
`smooth_sec` is the interpretable form of lambda: features shorter than about
that many seconds are flattened out of the drift, and a real tempo change slower
than that still comes through.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from .align import Alignment
from .gp_score import Score

SMOOTH_SEC = 20.0         # drift features shorter than this are flattened; see align.py
# ⚠ THESE BOUND THE *STRETCH*, NOT A TEMPO. They were called MIN/MAX_TEMPO_RATIO
# until 2026-09-25, and that name cost real time: it reads as "the tempo may not
# move more than 2x", which invites the conclusion that a map showing 240 BPM
# against a notated 120 is this clamp saturating. It is not -- the clamp constrains
# `audio_span` to within [0.5x, 2x] of `span_score`, and since
# stretch = span_score / audio_span, that IS a stretch band. A tab that notates
# 120 -> 240 passes through it untouched, because its score span halves too.
#
# The band is deliberately wide: it is a guard against a map that failed at the
# very start or end, not a quality target. A good map holds stretch within +-5%,
# which is forty times tighter.
MIN_STRETCH = 0.5
MAX_STRETCH = 2.0


def smooth_drift(score_times: np.ndarray, audio_times: np.ndarray,
                 frame_sec: float, smooth_sec: float = SMOOTH_SEC) -> np.ndarray:
    """Return audio times whose DERIVATIVE is smooth over ~`smooth_sec`.

    Smooths the drift `audio - score` rather than the audio times themselves, so
    the strong prior -- that the two clocks run at nearly the same rate -- is
    built in and only the small departure from it is regularised. Smoothing the
    audio times directly would fight that prior and blur the alignment.
    """
    st = np.asarray(score_times, dtype=float)
    at = np.asarray(audio_times, dtype=float)
    n = len(st)
    if n < 5 or smooth_sec <= 0:
        return at

    drift = at - st

    # Second-difference penalty. On a uniform grid of spacing h, a penalty weight
    # of (T/h)^4 flattens features shorter than about T -- that is the standard
    # fourth-power relation between a roughness penalty and its cutoff, and it is
    # why the knob is expressed in seconds rather than as a bare lambda.
    lam = (smooth_sec / max(frame_sec, 1e-6)) ** 4
    d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    a_mat = (eye(n, format="csc") + lam * (d2.T @ d2)).tocsc()
    smoothed = spsolve(a_mat, drift)

    # Re-centre on the raw drift. The solve minimises a SUM of squares, so its
    # residual has zero mean but not zero median, and the penalty is unconstrained
    # at the two boundaries -- together those left a measured 35ms of extra bias
    # against hand-placed points at smooth_sec=20.
    #
    # Removing it is free in the only currency that matters here: a constant shift
    # changes no difference between neighbouring points, so it cannot affect the
    # implied tempo at all. And a bias is the inaudible half of position error --
    # it moves where the audio starts, not how it tracks -- so correcting it buys
    # accuracy against ground truth without touching the wobble.
    smoothed = smoothed + float(np.median(drift - smoothed))

    out = st + smoothed
    return np.maximum.accumulate(out)      # a sync map must never run backwards


def tempo_locked_points(score: Score, alignment: Alignment,
                        every_n_bars: int = 8,
                        smooth_sec: float = SMOOTH_SEC) -> list[tuple[int, float]]:
    """Sync points sampled from the tempo-regularised map.

    The anchor is the first bar with notated content rather than playback bar 0,
    because bar 0 is a partial pickup bar on real tabs -- on "Sultans of swing" a
    1/4 bar 0.40s long, which is the worst anchor available and not where the
    author put theirs.
    """
    from .fit_points import first_sounding_bar

    smoothed = smooth_drift(alignment.score_times, alignment.audio_times,
                            alignment.frame_sec, smooth_sec)

    start = first_sounding_bar(score)
    picks = [b for b in score.bars
             if b.playback_idx >= start and (b.playback_idx - start) % every_n_bars == 0]
    if not picks:
        return []
    if picks[-1].playback_idx != score.bars[-1].playback_idx:
        picks.append(score.bars[-1])

    st = np.asarray(alignment.score_times, dtype=float)
    times = np.interp([b.start for b in picks], st, smoothed)
    times = np.maximum.accumulate(times)

    # A guard, not a tuning knob: clamp any segment that would imply an absurd
    # tempo. Smoothing cannot normally produce one, but a map that failed badly
    # at the very start or end can, and a single absurd segment is exactly the
    # kind of thing a player would hear as the track lurching.
    out = [(picks[0].playback_idx, float(times[0]))]
    for k in range(1, len(picks)):
        i0, i1 = picks[k - 1].playback_idx, picks[k].playback_idx
        span_score = score.bars[i1].start - score.bars[i0].start
        lo = out[-1][1] + span_score / MAX_STRETCH
        hi = out[-1][1] + span_score / MIN_STRETCH
        out.append((i1, float(np.clip(times[k], lo, hi))))
    return out


def implied_tempo(points, score: Score) -> np.ndarray:
    """BPM between consecutive points -- what write_sync stores as ModifiedTempo.

    ⚠ DO NOT JUDGE A MAP BY THE STEADINESS OF THIS. A tab that notates a tempo
    change should produce one, so an unsteady implied tempo is not evidence of a
    bad map. Use `stretch_ratio`, which divides this by the notated tempo and is
    the quantity that actually has to be smooth. See the banner on
    `stretch_ratio` for the four songs this distinction exonerated.
    """
    out = []
    for (i0, t0), (i1, t1) in zip(points, points[1:]):
        quarters = sum(score.bars[b].quarters for b in range(i0, i1))
        if t1 > t0 and quarters > 0:
            out.append(quarters * 60.0 / (t1 - t0))
    return np.array(out)


def notated_tempo(points, score: Score) -> np.ndarray:
    """BPM the TAB asks for over each segment -- the same arithmetic on score time.

    Aligned with `implied_tempo`: same segments, same filtering, so the two divide
    element-wise. Both drop a segment with no length or no time, which is why they
    must be computed by the same rule rather than independently.
    """
    out = []
    for (i0, t0), (i1, t1) in zip(points, points[1:]):
        quarters = sum(score.bars[b].quarters for b in range(i0, i1))
        span = score.bars[i1].start - score.bars[i0].start
        if t1 > t0 and quarters > 0:
            out.append(quarters * 60.0 / span if span > 0 else 0.0)
    return np.array(out)


def stretch_ratio(points, score: Score) -> np.ndarray:
    """How fast the RECORDING plays the notated music: implied / notated tempo.

    ⚠ THIS, NOT THE IMPLIED TEMPO, IS THE QUANTITY THAT MUST BE SMOOTH -- and
    getting that wrong sent a whole investigation down the wrong road. The nine
    roughest files in a 422-file batch all showed an implied tempo spanning 2:1,
    which looked like the grid latching onto the wrong beat level. It was not.
    Their tabs simply notate a tempo change, and the maps follow it correctly:

        file                        implied tempo   notated tempo   STRETCH
        As I Lay Dying - Forsaken   119.2-240.6     120.0-240.0     0.99-1.02
        Sublime - Seed               99.8-208.5      98.5-205.0     0.97-1.05
        Sublime - Saw Red            92.9-222.9      98.0-216.0     0.95-1.03
        Avenged Sevenfold - M.i.a    66.5-165.6      70.0-165.0     0.95-1.01

    A stretch inside +-5% is a very good map. The "2:1 metrical slip" was an
    artefact of measuring the wrong thing: on Forsaken the implied-tempo zigzag is
    19.48 and the stretch zigzag is 0.0039.

    This is the same distinction the author drew when they corrected an earlier
    tempo penalty -- *"some songs have slow tempo drift. Some have dramatic tempo
    changes that aren't reflected in a simplified tab."* A change the tab DOES
    notate is the easy half of that, and dividing by the notated tempo removes it
    exactly, leaving only the half that has to be judged from the audio.

    AND THE FILE FORMAT AGREES, which is the strongest argument that this is the
    natural quantity rather than a convenient one. Every GPIF sync point stores
    BOTH numbers side by side -- `ModifiedTempo` (the performed tempo) and
    `OriginalTempo` (what the tab notates) -- so their ratio is exactly this, and
    Guitar Pro has been carrying it all along.

    ⚠ One subtlety in the denominator: `notated_tempo` averages over the whole
    segment (quarters / score span) rather than reading `bar.bpm` at the segment's
    first bar. That is deliberate, because the numerator is a segment average too.
    Using `bar.bpm` would be wrong for any segment containing a mid-segment tempo
    change -- the two would be measured over different spans.
    """
    imp = implied_tempo(points, score)
    nom = notated_tempo(points, score)
    if len(imp) == 0:
        return imp
    return np.where(nom > 0, imp / np.maximum(nom, 1e-9), np.nan)


# --------------------------------------------------------------------------
# RULED OUT (2026-09-25): repairing a map by its implied tempo.
#
# ⚠ EVERYTHING FROM HERE TO `stretch_anomaly` IS A CLOSED INVESTIGATION. It is
# kept as the evidence base, exactly as the project keeps `FundamentalDrive/`, and
# NOTHING IN THE SHIPPING PATH IMPORTS IT. Do not wire it in.
#
# The idea was that the roughest files in a batch had suffered a "metrical slip" --
# the grid latching onto a half- or double-time level -- because their implied
# tempo spanned 2:1. Two repairs were built and A/B'd over the whole 422-file
# batch with `slip_ab.py`, grading the gridline hit rate:
#
#     repair        fired on   hit rate change        worst case
#     interior         42      -0.02 pts mean         -1.4 pts
#     edges            26      -0.35 pts mean         -5.7 pts
#
# and, by verdict, no group gained more than +0.21 points -- inside the noise. The
# reason is that the premise was false. Those maps were not slipping; their TABS
# notate a tempo change and the maps follow it correctly, with a stretch ratio
# inside +-5%. See the banner on `stretch_ratio` for the four worked examples.
#
# So the real defect was the measurement, not the maps, and the fix that came out
# of this was one line in `grid_align.refine_to_grid` -- penalise roughness in the
# stretch rather than in the tempo. `stretch_anomaly` below is what survived, as a
# DETECTOR feeding triage rather than an automatic repair; the alignment literature
# agrees that unmarked repeats and jumps want a little human labelling rather than
# more cleverness ("Just Label the Repeats", ISMIR 2024).
# --------------------------------------------------------------------------

SLIP_RATIO = 1.5           # a segment this far from the song's median tempo is suspect


def _segment_quarters(score: Score, idxs: list[int]) -> np.ndarray:
    return np.array([sum(score.bars[b].quarters for b in range(idxs[k], idxs[k + 1]))
                     for k in range(len(idxs) - 1)], dtype=float)


def slip_cost(quarters: np.ndarray, times: np.ndarray,
              ratio: float = SLIP_RATIO) -> tuple[float, float]:
    """How far this map's implied tempo strays from its own median, and by how much.

    Returns (cost, reference_bpm). The cost is a sum of squared *log*-ratio excess
    beyond the allowed band, so it is scale-free -- 2x too fast and 2x too slow
    cost the same, which is the right symmetry for a metrical slip -- and it is
    exactly zero for any map whose every segment is inside the band. That last
    property is what makes it safe to use as an acceptance test: a map nobody
    should touch has nothing to gain.

    A segment spanning ZERO quarters is excluded rather than punished. Two sync
    points inside one bar carry no tempo information at all, so they are neither
    evidence of a slip nor repairable as one -- costing them as worst-case made
    "TOOL - Vicarious" score 92 on a map whose real tempo ratio is only 1.21x.
    A segment with real length but zero or negative *time*, on the other hand, is
    genuinely broken and is charged the maximum.
    """
    spans = np.diff(times)
    measurable = quarters > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tempo = np.where(spans > 1e-6, quarters * 60.0 / np.maximum(spans, 1e-9), 0.0)
    good = tempo[measurable & (tempo > 0)]
    if len(good) < 3:
        return 0.0, 0.0
    ref = float(np.median(good))
    if ref <= 0:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(tempo > 0, np.abs(np.log(np.maximum(tempo, 1e-9) / ref)), 10.0)
    excess = np.where(measurable, np.maximum(0.0, lr - np.log(ratio)), 0.0)
    return float(np.sum(excess ** 2)), ref


def fix_metrical_slips(score: Score, points: list[tuple[int, float]],
                       ratio: float = SLIP_RATIO, max_moves: int = 40
                       ) -> tuple[list[tuple[int, float]], int]:
    """Repair sync points whose implied tempo is a multiple of the real one.

    ⚠ THE WORST FILES IN A BATCH FAIL THIS WAY, AND THE GRID SCORE CANNOT SEE
    IT. Grading 422 written copies, the nine with the roughest tempo all showed
    the same signature -- an implied tempo spanning a **2:1 ratio**:

        Sublime - Seed              99.8-208.4 BPM   (2.09x)   hit rate 72.5%
        Sublime - Saw Red           92.9-222.9       (2.40x)   hit rate 73.5%
        As I Lay Dying - Forsaken  119.2-240.6       (2.02x)   hit rate 93.4%
        Bullet For My Valentine     79.5-206.8       (2.60x)   hit rate 89.3%

    ⚠ THE READING BELOW IS THE ONE THAT WAS WRONG. It is left in place because the
    reasoning was plausible and someone will reach for it again:

        "That is the grid locking onto the wrong METRICAL LEVEL -- a half-time feel
        or a breakdown read as double tempo -- not measurement jitter. Note the
        high hit rates: those grids really are sitting on transients, just the
        wrong ones. So the grid score cannot catch this, and a ratio test against
        the song's own median tempo is the only thing that sees it."

    Both sentences are false. The grids are on the RIGHT transients: every one of
    those tabs notates a tempo change, and the maps follow it to within 5%. The
    high hit rates were the evidence, read backwards -- a map that scores 93.4% on
    the grid is not one that latched onto the wrong beat.

    A second wrong step, recorded because it wasted a detector: the exactly-240.0
    segments were read as the `MIN/MAX_STRETCH` clamp saturating. That clamp bounds
    the stretch, not the tempo, so a notated 120 -> 240 passes through it untouched
    and it never fired once across 422 files.

    The repair is outlier re-interpolation under an explicit cost, never
    clamping. A segment with an absurd tempo means one of its two endpoints is
    displaced and which one is not knowable locally, so instead of guessing, every
    point is *tried*: replace it by the line through its neighbours (in SCORE
    time -- bars are not equal length, so interpolating on index would be wrong)
    and keep the move only if `slip_cost` falls. Clamping the segment instead
    would push the error into the next segment and drag the rest of the map with
    it.

    ⚠ THE ACCEPTANCE TEST IS THE SAFETY ARGUMENT, and it is here because the author
    warned about exactly the failure it prevents: *"some songs have dramatic tempo
    changes that aren't reflected in a simplified tab."* A map whose tempo really
    does move has cost 0 inside the band and gains nothing from any move, so it is
    returned untouched. Only a map that is already outside the band can change at
    all, and then only in the direction that reduces the violation.

    Returns (points, number of points moved).
    """
    if len(points) < 4:
        return points, 0

    idxs = [i for i, _ in points]
    times = np.array([t for _, t in points], dtype=float)
    starts = np.array([score.bars[i].start for i in idxs], dtype=float)
    quarters = _segment_quarters(score, idxs)

    cost, _ = slip_cost(quarters, times, ratio)
    if cost <= 0.0:
        return points, 0

    n = len(times)
    moved = 0
    for _ in range(max_moves):
        best_cost, best_k, best_t = cost, -1, 0.0
        for k in range(n):
            # Interior points come off the line through their neighbours; the two
            # ends are extrapolated from the nearest pair, since a displaced first
            # point is a real case (the map's start is where DTW is weakest) and
            # blaming its neighbour instead would only spread the error.
            if 0 < k < n - 1:
                lo, hi = k - 1, k + 1
            elif k == 0:
                lo, hi = 1, 2
            else:
                lo, hi = n - 3, n - 2
            span = starts[hi] - starts[lo]
            if span <= 0:
                continue
            frac = (starts[k] - starts[lo]) / span
            cand = times[lo] + frac * (times[hi] - times[lo])
            if not np.isfinite(cand):
                continue
            trial = times.copy()
            trial[k] = cand
            trial = np.maximum.accumulate(trial)
            c, _ = slip_cost(quarters, trial, ratio)
            if c < best_cost - 1e-12:
                best_cost, best_k, best_t = c, k, cand
        if best_k < 0:
            break
        times[best_k] = best_t
        times = np.maximum.accumulate(times)
        cost = best_cost
        moved += 1
        if cost <= 0.0:
            break

    return [(i, float(t)) for i, t in zip(idxs, times)], moved


def fix_edge_spread(score: Score, points: list[tuple[int, float]],
                    ratio: float = SLIP_RATIO, max_edge_frac: float = 0.25
                    ) -> tuple[list[tuple[int, float]], int]:
    """Re-derive the leading and trailing points from the song's stable interior.

    ⚠ THIS IS THE HALF OF THE SLIP PROBLEM THAT IS ACTUALLY FIXABLE, and it was
    found by decomposing a failed fix rather than by reasoning. "Sublime - Saw
    Red" runs at a rock-steady 216-222 BPM for nineteen consecutive segments and
    is slow only in its first two and last three:

        bar   0 ->   4    98.1 BPM
        bar   4 ->   8   137.1
        bar   8 ->  72   216-223   (nineteen segments, the song)
        bar  76 ->  80   133.2
        bar  80 ->  84    99.3
        bar  84 ->  85    92.9

    The middle is not in doubt, so neither is the tempo: the edges are wrong, not
    the song. The cause is that a chroma map has nothing to match in a lead-in or
    a fade -- no harmony, so no evidence -- and the DTW path spreads to fill the
    space. A human never has this problem because they put their first point on
    the first drum hit and stop.

    So: find the longest run of segments inside the band, take its tempo as the
    truth, and extrapolate the out-of-band head and tail from it at that rate.
    This is deliberately NOT applied to the middle of a song, where an out-of-band
    segment usually means a structural difference between tab and recording (see
    `tempo_anomaly`) and straightening it would be inventing data.

    `max_edge_frac` bounds how much of a song may be treated as an edge, so a map
    that is bad throughout cannot be rewritten wholesale under this name.

    Returns (points, number of points moved).
    """
    if len(points) < 6:
        return points, 0

    idxs = [i for i, _ in points]
    times = np.array([t for _, t in points], dtype=float)
    starts = np.array([score.bars[i].start for i in idxs], dtype=float)
    quarters = _segment_quarters(score, idxs)

    spans = np.diff(times)
    with np.errstate(divide="ignore", invalid="ignore"):
        tempo = np.where((spans > 1e-6) & (quarters > 0),
                         quarters * 60.0 / np.maximum(spans, 1e-9), 0.0)
    good = tempo[tempo > 0]
    if len(good) < 3:
        return points, 0
    ref = float(np.median(good))
    if ref <= 0:
        return points, 0

    in_band = (tempo > 0) & (tempo <= ref * ratio) & (tempo >= ref / ratio)
    if in_band.all() or not in_band.any():
        return points, 0

    # The reference rate comes from the longest in-band run, not from the median
    # of everything -- on a map whose edges are badly wrong the median is already
    # contaminated by them.
    best_len = best_start = 0
    run = 0
    for k, v in enumerate(in_band):
        run = run + 1 if v else 0
        if run > best_len:
            best_len, best_start = run, k - run + 1
    if best_len < 3:
        return points, 0
    run_lo, run_hi = best_start, best_start + best_len          # segment indices
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = float(np.median(tempo[run_lo:run_hi])) / 60.0     # quarters/sec
    if rate <= 0:
        return points, 0

    n = len(times)
    max_edge = max(1, int(round(n * max_edge_frac)))
    moved = 0

    # Head: points before the stable run's first point, if that is an edge at all.
    head_end = run_lo                     # point index the run starts at
    if 0 < head_end <= max_edge:
        anchor_t, anchor_s = times[head_end], starts[head_end]
        for k in range(head_end):
            times[k] = anchor_t - (anchor_s - starts[k]) * _sec_per_score_sec(
                score, idxs, k, head_end, rate)
            moved += 1

    tail_start = run_hi                   # point index the run ends at
    if 0 < n - 1 - tail_start <= max_edge:
        anchor_t, anchor_s = times[tail_start], starts[tail_start]
        for k in range(tail_start + 1, n):
            times[k] = anchor_t + (starts[k] - anchor_s) * _sec_per_score_sec(
                score, idxs, tail_start, k, rate)
            moved += 1

    times = np.maximum.accumulate(times)
    return [(i, float(t)) for i, t in zip(idxs, times)], moved


def _sec_per_score_sec(score: Score, idxs: list[int], k0: int, k1: int,
                       rate: float) -> float:
    """Audio seconds per score second between two points, at `rate` quarters/sec.

    Bars are not equal length and the notated tempo varies, so a score-time span
    is not proportional to a quarter count. This converts one to the other using
    the actual quarters between the two points, which is the same quantity
    `implied_tempo` and Guitar Pro's own playback use.
    """
    a, b = min(k0, k1), max(k0, k1)
    q = sum(score.bars[x].quarters for x in range(idxs[a], idxs[b]))
    span_score = score.bars[idxs[b]].start - score.bars[idxs[a]].start
    if q <= 0 or span_score <= 0 or rate <= 0:
        return 1.0
    return (q / rate) / span_score


# --------------------------------------------------------------------------
# What survived: a detector, on the right basis.
# --------------------------------------------------------------------------

STRETCH_BAND = 0.15       # a segment this far off the song's median stretch is suspect


def stretch_anomaly(score: Score, points: list[tuple[int, float]],
                    band: float = STRETCH_BAND) -> dict:
    """Describe, rather than repair, a map that disagrees with its own tab.

    ⚠ IT REPORTS AND DOES NOT REPAIR, AND THAT IS A MEASURED DECISION. Two
    automatic repairs were built for this and A/B'd over the full 422-file batch;
    both came out inside the noise (-0.02 and -0.35 gridline points) because a long
    out-of-band run is not a displaced point but a real difference between the tab
    and the recording. Straightening it would throw the rest of the map out to hide
    it. The alignment literature reaches the same place: handling unmarked repeats
    and jumps is open, and the current practical answer ("Just Label the Repeats",
    ISMIR 2024) is a little human labelling for a large quality gain. So this feeds
    the triage report and the author decides.

    The reference is the song's own MEDIAN STRETCH, not its median tempo. Against
    the median tempo this same test called four excellent maps broken, because
    their tabs notate a tempo change -- see the banner on `stretch_ratio`. Against
    the median stretch a notated change is invisible, which is the point.

    `band` is a fraction, not a multiple: 0.15 means a segment playing 15% faster
    or slower than the rest of the song relative to what the tab asks for. That is
    far tighter than the 1.5x the old test used, and it can afford to be, because
    a good map here holds +-5%.

    Keys: n_segments, ref_stretch, worst_dev, n_out_of_band, longest_run,
    head_bad, tail_bad, verdict. The verdicts are:

        clean        -- every segment inside the band
        edge_spread  -- only the leading/trailing segments are out. A chroma map
                        has nothing to match in a lead-in or a fade, so the DTW
                        path spreads to fill it. Cosmetic: it is before the first
                        note and after the last.
        isolated     -- one or two interior segments out, not adjacent. Usually a
                        single mis-placed point; worth a look.
        structural   -- three or more consecutive interior segments out. The tab
                        and the recording differ in CONTENT -- an unnotated repeat,
                        a skipped section, a solo the tab omits. No amount of
                        alignment fixes this; it needs a human ear.
    """
    out = {"n_segments": 0, "ref_stretch": 0.0, "worst_dev": 0.0,
           "n_out_of_band": 0, "longest_run": 0, "head_bad": 0, "tail_bad": 0,
           "verdict": "too_few_points"}
    if len(points) < 4:
        return out

    s = stretch_ratio(points, score)
    measurable = np.isfinite(s) & (s > 0)
    if measurable.sum() < 3:
        return out
    ref = float(np.median(s[measurable]))
    if ref <= 0:
        return out

    dev = np.where(measurable, np.abs(s / ref - 1.0), 0.0)
    bad = measurable & (dev > band)

    longest = run = 0
    for v in bad:
        run = run + 1 if v else 0
        longest = max(longest, run)

    head = 0
    while head < len(bad) and bad[head]:
        head += 1
    tail = 0
    while tail < len(bad) and bad[len(bad) - 1 - tail]:
        tail += 1

    # An all-bad map would otherwise be counted twice by head and tail.
    interior_bad = max(0, int(bad.sum()) - head - tail) if head + tail < len(bad) \
        else 0
    if not bad.any():
        verdict = "clean"
    elif interior_bad == 0:
        verdict = "edge_spread"
    elif longest >= 3:
        verdict = "structural"
    else:
        verdict = "isolated"

    out.update({
        "n_segments": int(measurable.sum()), "ref_stretch": round(ref, 4),
        "worst_dev": round(float(dev[measurable].max()), 4),
        "n_out_of_band": int(bad.sum()), "longest_run": int(longest),
        "head_bad": int(head), "tail_bad": int(tail), "verdict": verdict,
    })
    return out



