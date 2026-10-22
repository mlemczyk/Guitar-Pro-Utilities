"""Score and refine a sync map by how well the audio's transients sit on the grid.

THE AUTHOR'S IDEA (2026-09-24): *"render a grid in memory, and see what
offset/synch/tempo change would align the most transient spikes onto a 1/4 grid
line"*, and their working method: *"align one transient, jump over 4 or 5 bars,
and see if the next transient needs to be aligned to the bar, then keep going
until it drifts."*

WHY THIS IS A DIFFERENT INSTRUMENT FROM EVERYTHING ELSE HERE. `align.py` matches
the tab's HARMONY to the recording's, via chroma. That answers "which bar is
this", and it answers it well -- the maps agree with hand-placed points to a few
tens of milliseconds. It is blind to whether a bar LINE lands on a drum hit,
because chroma has no notion of a grid. The author's screenshot of "The Outfield -
Your Love" shows exactly that failure: the transients sit near the gridlines and
not on them, on a map whose harmonic alignment is fine.

So this module asks a different question -- *does the beat grid fall on the
audio's transients* -- and it has one property nothing else here has:

    ⚠ IT NEEDS NO GROUND TRUTH.

Every other quality measure in this package compares against hand-placed sync
points, so it only works on the 147 tabs that have them. This one compares a map
against the RECORDING, so it works on all 1410 that do not -- which is the set a
batch pass actually writes to.

`grid_score` is the mean onset strength sampled at quarter-note grid positions,
normalised by the mean onset strength everywhere. A value of 1.0 means the grid
lines fall on statistically average moments, i.e. the map is no better than
chance; above 1 means the grid is finding transients. It is a ratio rather than a
raw sum so that a loud song and a quiet one are comparable.
"""

from __future__ import annotations

import numpy as np

from .align import ONSET_HOP, SR
from .audio_io import load_mono
from .gp_score import Score

GRID_TOLERANCE_SEC = 0.030     # a transient this close to a gridline counts as on it


def onset_envelope(audio_path, sr: int = SR, hop: int = ONSET_HOP):
    """(envelope, times) of onset strength, at the fine hop onsets deserve."""
    import librosa
    y = load_mono(audio_path, sr)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    return env, times


def beat_positions(score: Score, points: list[tuple[int, float]],
                   subdivision: float = 1.0) -> np.ndarray:
    """Audio times of every grid line the map implies.

    `subdivision` is in quarter notes: 1.0 walks the quarter-note grid the author
    described, 0.5 would walk eighths. Positions are interpolated inside each
    sync-point segment at that segment's own rate, which is exactly what Guitar
    Pro does between two sync points.
    """
    if len(points) < 2:
        return np.array([])

    out = []
    for (i0, t0), (i1, t1) in zip(points, points[1:]):
        quarters = sum(score.bars[b].quarters for b in range(i0, i1))
        if quarters <= 0 or t1 <= t0:
            continue
        sec_per_quarter = (t1 - t0) / quarters
        n = int(np.floor(quarters / subdivision))
        out.extend(t0 + np.arange(n) * subdivision * sec_per_quarter)
    return np.array(out)


def grid_score(score: Score, points: list[tuple[int, float]],
               env: np.ndarray, env_times: np.ndarray,
               subdivision: float = 1.0) -> dict:
    """How well does this map put the beat grid on the audio's transients?

    Returns `ratio` (mean onset strength on the grid, over the mean everywhere),
    plus the fraction of gridlines that land within GRID_TOLERANCE_SEC of a local
    onset peak. Both are scale-free, so two different songs can be compared.
    """
    grid = beat_positions(score, points, subdivision)
    if len(grid) < 4 or len(env) < 4:
        return {"ratio": 0.0, "hit_rate": 0.0, "n_grid": len(grid)}

    idx = np.clip(np.searchsorted(env_times, grid), 0, len(env) - 1)
    on_grid = env[idx]
    baseline = float(np.mean(env))
    ratio = float(np.mean(on_grid) / baseline) if baseline > 0 else 0.0

    # Hit rate: is there a LOCAL PEAK of the envelope within tolerance? A high
    # mean can come from sitting on a broad loud region, which is not the same
    # as sitting on an attack.
    import librosa
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=0.1 * baseline, wait=2)
    if len(peaks) == 0:
        return {"ratio": ratio, "hit_rate": 0.0, "n_grid": len(grid)}
    peak_times = env_times[peaks]
    nearest = np.abs(peak_times[np.clip(np.searchsorted(peak_times, grid), 0,
                                        len(peak_times) - 1)] - grid)
    left = np.abs(peak_times[np.clip(np.searchsorted(peak_times, grid) - 1, 0,
                                     len(peak_times) - 1)] - grid)
    nearest = np.minimum(nearest, left)
    hit_rate = float(np.mean(nearest <= GRID_TOLERANCE_SEC))

    return {"ratio": ratio, "hit_rate": hit_rate, "n_grid": len(grid),
            "median_dist_ms": float(np.median(nearest) * 1000)}


# --------------------------------------------------------------------------
# Refinement: nudge each sync point onto the transients, without reintroducing
# the wobble that regularising the drift curve just removed.
# --------------------------------------------------------------------------

MAX_NUDGE_SEC = 0.120      # how far one point may move; beyond this it is a
                           # different bar, not a correction
TEMPO_PENALTY = 2.5        # weight on tempo ACCELERATION; see refine_to_grid


def _segment_energy(score: Score, i0: int, t0: float, i1: int, t1: float,
                    env: np.ndarray, env_times: np.ndarray,
                    subdivision: float = 1.0) -> float:
    """Total onset strength on the grid between two sync points."""
    quarters = sum(score.bars[b].quarters for b in range(i0, i1))
    if quarters <= 0 or t1 <= t0:
        return 0.0
    sec_per_quarter = (t1 - t0) / quarters
    n = int(np.floor(quarters / subdivision))
    if n <= 0:
        return 0.0
    grid = t0 + np.arange(n) * subdivision * sec_per_quarter
    idx = np.clip(np.searchsorted(env_times, grid), 0, len(env) - 1)
    return float(np.sum(env[idx]) / n)


def refine_to_grid(score: Score, points: list[tuple[int, float]],
                   env: np.ndarray, env_times: np.ndarray,
                   max_nudge: float = MAX_NUDGE_SEC,
                   tempo_penalty: float = TEMPO_PENALTY,
                   passes: int = 3,
                   use_stretch: bool = True) -> list[tuple[int, float]]:
    """Coordinate ascent on the sync points, maximising grid energy.

    This is the author's own method made mechanical: *"align one transient, jump
    over 4 or 5 bars, and see if the next transient needs to be aligned to the
    bar, then keep going until it drifts."* Each point is nudged, in turn, to
    wherever it puts the most onset energy on the grid of the two segments it
    borders -- which is what "align this bar line to that drum hit" is, scored
    over a whole segment instead of one visible spike.

    ⚠ IT CARRIES A TEMPO PENALTY, AND MUST. Maximising grid energy alone would
    happily move neighbouring points in opposite directions to catch two loud
    hits, which is precisely the wobble that regularising the drift curve was
    built to remove -- a map can sit on every transient and still lurch between
    them. The penalty prices a change in implied tempo against the energy gained,
    so a nudge has to be worth more than the unsteadiness it introduces.

    `max_nudge` is deliberately small. A point that wants to move further than
    this is not being corrected, it is being matched to a different bar, and that
    is the alignment's job rather than the refinement's.
    """
    if len(points) < 3 or len(env) < 4:
        return points

    idxs = [i for i, _ in points]
    times = np.array([t for _, t in points], dtype=float)

    # Notated rate of each segment, in quarters per SCORE second. Dividing the
    # audio rate by this is what makes the penalty below blind to a tempo change
    # the tab actually asks for; see the banner on the roughness term.
    seg_quarters = [sum(score.bars[b].quarters for b in range(idxs[k], idxs[k + 1]))
                    for k in range(len(idxs) - 1)]
    seg_score_span = [score.bars[idxs[k + 1]].start - score.bars[idxs[k]].start
                      for k in range(len(idxs) - 1)]
    notated_rate = [(q / s) if q > 0 and s > 0 else 0.0
                    for q, s in zip(seg_quarters, seg_score_span)]

    def tempo_of(k: int, ts: np.ndarray) -> float:
        """Segment k's STRETCH: how fast the audio plays what the tab notates.

        1.0 means the recording is at the notated tempo. It is deliberately a
        ratio and not a rate -- the penalty below differences this, and a rate
        would carry the tab's own tempo changes into that difference.
        """
        quarters = seg_quarters[k]
        dt = ts[k + 1] - ts[k]
        if dt <= 0 or quarters <= 0:
            return 0.0
        rate = quarters / dt
        if not use_stretch:
            return rate                # the pre-2026-09-25 behaviour, for A/B only
        return rate / notated_rate[k] if notated_rate[k] > 0 else 0.0

    def local_objective(k: int, ts: np.ndarray) -> float:
        """Grid energy of the segments touching point k, less tempo roughness."""
        energy = 0.0
        if k > 0:
            energy += _segment_energy(score, idxs[k - 1], ts[k - 1], idxs[k], ts[k],
                                      env, env_times)
        if k + 1 < len(idxs):
            energy += _segment_energy(score, idxs[k], ts[k], idxs[k + 1], ts[k + 1],
                                      env, env_times)
        # ⚠ THE PENALTY IS ON TEMPO *ACCELERATION*, NOT ON TEMPO CHANGE.
        #
        # An earlier version penalised |tempo[j+1] - tempo[j]|, which charges a
        # song for changing tempo at all -- and the author's correction is that
        # changing tempo is normal: "some songs have slow tempo drift. Some have
        # dramatic tempo changes that aren't reflected in a simplified tab." A
        # map that follows a real accelerando is right, and that penalty fought
        # it.
        #
        # What is NOT normal is tempo ALTERNATING between neighbouring segments,
        # which is measurement noise rather than music. "Nirvana - Smells Like
        # Teen Spirit" came out at 111.99, 118.37, 114.15, 120.61, 115.83 BPM
        # over its first five segments -- swinging +-4 BPM around 116 on a song
        # whose tempo is steady. A second difference is exactly the quantity that
        # separates the two: a sustained drift has a constant first difference
        # and so costs nothing here, while an up-down-up wobble is charged twice.
        #
        # ⚠ AND IT IS THE SECOND DIFFERENCE OF THE *STRETCH*, NOT OF THE TEMPO.
        # `tempo_of` divides by the notated rate, so a tempo change the tab writes
        # down costs nothing at all. Without that division this penalty fought the
        # tab: "As I Lay Dying - Forsaken" notates 120 BPM stepping to 240 and
        # back, and the raw second difference charged ~0.5 roughness at each step
        # -- pinning the points exactly where the music changes and the map most
        # needs freedom to find the transients.
        #
        # The change is surgical rather than a retuning. Scaling a, b and c by the
        # same constant leaves abs(c - 2b + a)/b unchanged, so on a tab of constant
        # notated tempo this is bit-for-bit the old penalty and TEMPO_PENALTY
        # keeps its meaning. Only the segments that straddle a notated tempo change
        # behave differently, which is exactly the set that was wrong.
        rough = 0.0
        for j in (k - 3, k - 2, k - 1, k):
            if j >= 0 and j + 2 <= len(idxs) - 2:
                a, b, c = (tempo_of(j, ts), tempo_of(j + 1, ts), tempo_of(j + 2, ts))
                if a > 0 and b > 0 and c > 0:
                    rough += abs(c - 2.0 * b + a) / b
        return energy - tempo_penalty * rough * float(np.mean(env))

    steps = np.linspace(-max_nudge, max_nudge, 25)
    for _ in range(passes):
        moved = 0.0
        for k in range(1, len(times) - 1):          # ends stay put: they anchor
            base = times[k]
            best_t, best_v = base, local_objective(k, times)
            for d in steps:
                cand = base + d
                if not (times[k - 1] < cand < times[k + 1]):
                    continue
                times[k] = cand
                v = local_objective(k, times)
                if v > best_v:
                    best_t, best_v = cand, v
            times[k] = best_t
            moved = max(moved, abs(best_t - base))
        if moved < 0.002:
            break

    return [(i, float(t)) for i, t in zip(idxs, np.maximum.accumulate(times))]

