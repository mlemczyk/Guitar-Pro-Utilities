"""Align a Guitar Pro score to its backing track with chroma DTW.

The problem the sync points solve is a monotonic map from *score time* (the tab
played on its own authored tempo map) to *audio time* (the real recording). A
linear pad/offset and BPM drift are not two problems -- they are the intercept
and the slope of that one map, plus whatever the drummer did on the night.

So rather than detect notes in the audio and match them to notated notes, we
estimate the map directly:

  1. turn the score into a chromagram (12 pitch classes x time) by laying every
     pitched track's notes onto a decaying piano roll,
  2. compute a chromagram of the mp3 on the same frame grid,
  3. dynamic-time-warp one onto the other,
  4. read the warping path at each bar line to get a sync point.

Chroma is used rather than note events on purpose. It is aggregate harmonic
content, so it degrades gracefully where note detection fails outright -- and
note detection is the component this project has spent the most time debugging
(octave ghosts, over-split let-rings, string noise), all of that on isolated DI
bass, which is a far easier signal than a full mix.

Nothing here writes to a .gp file. This module only produces a mapping and a
confidence; deciding whether that mapping is good enough to write is a separate
question, deliberately kept separate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import librosa
import numpy as np

from .audio_io import load_mono
from .cymbals import cymbal_onsets, score_cymbal_bars, snap_to_cymbals
from .gp_score import Score

SR = 22050
HOP = 1024                 # 46.4ms per frame at SR
MAX_DTW_FRAMES = 7000      # pool coarser than this; keeps the cost matrix ~200MB
NOTE_DECAY_TAU = 1.2       # seconds; how fast a struck note fades in the score roll


ONSET_HOP = 256            # 11.6ms at SR -- onsets want far finer time resolution

# CENS smoothing and onset snapping, both measured on the FULL 138-file clean
# ground-truth set with one variable changed at a time:
#
#   variant                                  median   <50ms  <100ms  <250ms   <1s
#   A  smooth 41, no snap, no cymbals         173ms      11      47      91   126
#   B  smooth 41, snap 0.2s, no cymbals       175ms      33      50      86   127
#   C  smooth 41, snap 0.2s, cymbals          180ms      35      52      87   127
#   D  smooth 21, snap 0.2s, cymbals          134ms      40      61     103   129  <- chosen
#
# Read B and C carefully: snapping alone TRIPLES the files landing inside 50ms
# (11 -> 33 -> 35) while leaving the median flat and costing a little in the
# middle band. It sharpens what is already nearly right and occasionally grabs
# a wrong transient elsewhere. Only combined with less smoothing (D) does it
# win everywhere -- because a 1.9s-smoothed chromagram cannot put a point close
# enough for a 0.2s snap window to contain the right transient in the first
# place. The two changes are not independent and must not be judged separately.
#
# ⚠ TWO MEASUREMENT MISTAKES ON THE WAY HERE, BOTH WORTH NOT REPEATING:
#
# 1. **A subset chosen by a quality proxy flatters a precision/robustness
#    trade.** The first sweep of these parameters used 45 files taken off the
#    top of a list sorted by sync-point count -- the most carefully synced,
#    easiest files, i.e. exactly the ones that do not need the robustness the
#    smoothing buys. Sweep on the full set or a random sample, never on the
#    head of a sorted list.
# 2. **The follow-up full-set run changed two things at once** and appeared to
#    show smoothing 21 was a regression (173 -> 269ms). The baseline had been
#    run with `--step classic_p20` and the new run took the then-default
#    `--step classic`. That was a step-set difference misread as a smoothing
#    one, and it nearly buried a real improvement. `--step` now defaults to the
#    measured winner so the two cannot drift apart again.
CENS_SMOOTH = 21           # frames; None would mean librosa's default of 41

# ⚠ SNAPPING IS OFF, AND THIS REVERSES AN EARLIER DECISION IN THIS FILE.
#
# It was adopted on position error alone, where it looked worth +2 files inside
# 50ms out of 138. Re-measured over 115 ground-truth tabs while ALSO scoring the
# tempo the points imply between them -- the thing Guitar Pro actually plays the
# audio at, and which nothing had ever measured:
#
#   variant             pts   err med   err max   implied-tempo SD
#   grid 4, snap 0.20    36      87ms     407ms       6.88
#   grid 4, snap 0.00    36      82ms     406ms       6.69   <- chosen
#
# Snapping is neutral-to-slightly-worse on both, so the +2 files were noise and
# the earlier claim was overstated. It is off because it can only add noise to
# the SPACING between neighbours: each point is moved independently to whichever
# transient is loudest within its window, and tempo is a difference of two such
# positions. On "Dire Straits - Sultans of swing", where the author hand-synced a
# file and found the generated one unsatisfying, turning it off took the error
# against their own points from 37ms to 13ms and the tempo SD from 3.48 to 1.70.
#
# ⚠ THE REAL DEFECT IS STILL OPEN. Human maps on the same files run a tempo SD
# of 0.41-0.70; the best generated variant is 6.69, and no density reaches them
# (grid 16 gets SD 2.97 by paying 166ms of position). The cause is NOT frame
# quantisation -- tested and refuted on a random sample, where the grid is
# already 46ms and raising the cap moved SD 8.51 -> 8.42. It is the map's own
# local accuracy, ~111ms median, which differencing over a few-second span
# amplifies. Closing it needs the drift curve REGULARISED to be smooth by
# construction, not sampled. See `fit_points.py`, which is a first attempt that
# is NOT wired in because it wins on one song and loses library-wide.
ONSET_SNAP_WINDOW = 0.0    # seconds; 0 disables. See above before changing.


@dataclass
class Alignment:
    score_times: np.ndarray    # strictly increasing, seconds in score time
    audio_times: np.ndarray    # the matched audio time for each, seconds
    frame_sec: float
    path_cost: float           # mean cosine distance along the warping path
    diagonality: float         # fraction of path steps that advance both axes
    coverage: float            # fraction of the audio the score was mapped onto
    score_duration: float
    audio_duration: float
    onset_times: np.ndarray | None = None      # detected transients, seconds
    onset_strength: np.ndarray | None = None   # peak strength at each onset
    cymbal_times: np.ndarray | None = None     # high-frequency (crash-like) transients
    cymbal_strength: np.ndarray | None = None
    onset_env: np.ndarray | None = None        # the full envelope, for grid refinement
    onset_env_times: np.ndarray | None = None

    def to_audio(self, t: float | np.ndarray) -> np.ndarray:
        """Map score time -> audio time. Extrapolates flat past either end."""
        return np.interp(t, self.score_times, self.audio_times)

    @property
    def summary(self) -> str:
        return (f"cost={self.path_cost:.3f} diag={self.diagonality:.2f} "
                f"cov={self.coverage:.2f} "
                f"score={self.score_duration:.0f}s audio={self.audio_duration:.0f}s")


def score_chromagram(score: Score, frame_sec: float, n_frames: int,
                     tau: float = NOTE_DECAY_TAU) -> np.ndarray:
    """A 12 x n_frames chromagram of the tab, at authored timing.

    Each note contributes to its pitch class from its onset until its notated
    end, decaying exponentially so that a struck note counts for more than a
    long-held one -- which is roughly what a chromagram of real audio sees.
    """
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    decay_frames = max(1.0, tau / frame_sec)

    for note in score.pitched_notes:
        start = int(note["start"] / frame_sec)
        end = int(np.ceil(note["end"] / frame_sec))
        if end <= start:
            end = start + 1
        start = max(0, min(start, n_frames))
        end = max(0, min(end, n_frames))
        if end <= start:
            continue
        envelope = np.exp(-np.arange(end - start) / decay_frames)
        chroma[note["midi"] % 12, start:end] += envelope

    return chroma


def _normalize(chroma: np.ndarray) -> np.ndarray:
    """L2-normalize each frame, replacing silent frames with a uniform vector.

    A silent frame is an all-zero column, and cosine distance to an all-zero
    vector is 0/0 -- which reaches librosa as "cost matrix has NaN values" and
    kills the DTW outright. Silent frames are common and legitimate: every rest
    in the tab and every moment of quiet in the recording produces one. A
    uniform column is the honest substitute, being equidistant from everything
    rather than spuriously close to anything.
    """
    chroma = np.asarray(chroma, dtype=np.float64)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    silent = norms[0] < 1e-8
    out = chroma / np.maximum(norms, 1e-8)
    if silent.any():
        out[:, silent] = 1.0 / np.sqrt(12.0)
    return out


def _pool(chroma: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return chroma
    n = (chroma.shape[1] // factor) * factor
    if n == 0:
        return chroma
    return chroma[:, :n].reshape(chroma.shape[0], n // factor, factor).mean(axis=2)


def audio_chromagram(y: np.ndarray, sr: int, kind: str = "cens",
                     win_len_smooth: int | None = None) -> np.ndarray:
    """Chromagram of the recording. `kind` is cens, cqt or stft.

    CENS is the default on measurement, not by reputation. Over the first four
    ground-truth files it cut the residual bias to +-60ms where plain CQT ran
    to +338ms, and on "Pennywise - Fuck Authority" it took the worst-case error
    from 13.9s to 2.4s. Its smoothing and quantisation are what make it robust
    to the timbral gulf between a synthetic piano roll and a real mix. `stft`
    is much worse than either and is kept only so the comparison is repeatable.

    ⚠ THE SMOOTHING IS ALSO THE COST, AND IT IS LARGER THAN IT LOOKS.
    `chroma_cens` defaults to `win_len_smooth=41` frames, which at this hop is
    **1.90 seconds**. That is what buys the global robustness and it is also
    why the result is locally mushy: the author listened to a written file and
    found bar lines 200-300ms out, in a stretch where onsets sit 250ms apart.
    A smoothed feature cannot distinguish them. `win_len_smooth` is exposed so
    that trade can be measured rather than inherited, and `refine_to_onsets()`
    attacks the same problem from the other end.
    """
    if kind == "cqt":
        return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    if kind == "stft":
        return librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    if kind == "cens":
        kwargs = {} if win_len_smooth is None else {"win_len_smooth": win_len_smooth}
        return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP, **kwargs)
    raise ValueError(f"unknown chroma kind: {kind}")


# DTW local step sets, as (steps, weights_mul, weights_add).
#
# "classic" is librosa's default: a diagonal step plus pure horizontal and
# vertical ones. Those last two let the path stall indefinitely on one axis,
# which is how a 208-second error gets produced -- the path parks on one score
# frame while the audio runs away underneath it. The `_pN` variants add a
# constant penalty to stalling without forbidding it.
#
# "slope" forbids stalling outright: every step advances both axes, so the local
# slope is confined to [1/2, 2]. Real drift here is a few percent, so that bound
# cannot exclude a true path.
#
# THE WEIGHTS ARE NOT DECORATION. With uniform weights a (1,2) step costs the
# same single cell as a (1,1) step while covering more ground, so DTW treats
# long steps as free progress: measured on Zombie, the slope set produced 2227
# (1,2) steps, 2160 (2,1) steps and **zero** diagonals -- a pure zigzag that
# tripled the spread from 53ms to 139ms. Weighting each step by its own length
# (di+dj) removes that incentive.
#
# MEASURED over the 138 clean ground-truth files (see eval_*.csv):
#
#   step set      median |err|  <100ms  <250ms   <1s   >10s     worst
#   classic             271ms       28      63    98     17   208.7s
#   classic_p05         178ms       38      85   120      3   102.5s
#   classic_p10         171ms       42      89   125      2   116.6s
#   classic_p20         173ms       47      91   126      1    11.9s   <-- default
#   slope               376ms       11      44   114      3    14.4s
#
# `classic_p20` is the default on that evidence. The penalty's real work is in
# the last two columns: it takes catastrophic failures from 17 files to 1, and
# the worst case from 209 seconds to 12, while also improving the median. The
# `slope` set fixes stalling by construction but pays for it everywhere else.
STEP_SETS = {
    "classic":     (np.array([[1, 1], [0, 1], [1, 0]]), [1, 1, 1], [0.0, 0.00, 0.00]),
    "classic_p05": (np.array([[1, 1], [0, 1], [1, 0]]), [1, 1, 1], [0.0, 0.05, 0.05]),
    "classic_p10": (np.array([[1, 1], [0, 1], [1, 0]]), [1, 1, 1], [0.0, 0.10, 0.10]),
    "classic_p20": (np.array([[1, 1], [0, 1], [1, 0]]), [1, 1, 1], [0.0, 0.20, 0.20]),
    "slope":       (np.array([[1, 1], [1, 2], [2, 1]]), [2, 3, 3], [0.0, 0.00, 0.00]),
}


# When the recording is much longer than the score, corner-to-corner DTW is
# being asked to stretch the score across material that is not in it, and it
# fails badly. That is the case subsequence DTW exists for, so it is enabled
# automatically -- but ONLY there, because on a song where score ~= audio it is
# a disaster (see the `subseq` docstring below).
#
# Validated on Insomnium's "Winter's Gate", a 40-minute piece split into parts
# where every part's tab carries the WHOLE recording. Corner-to-corner scores
# diagonality 0.11-0.14; subsequence scores 0.92-0.99. The check that makes this
# more than a better-looking number is that the parts then map onto SUCCESSIVE,
# ASCENDING windows of the recording -- Pt.1 at 32-372s, Pt.4 at 1123-1440s,
# Pt.5 at 1355-1626s, with a 751s gap exactly where the library's missing Pts
# 2-3 belong. Nothing but a correct alignment tiles a recording in order.
AUTO_SUBSEQ_MIN_RATIO = 1.20   # audio at least 20% longer than the score
AUTO_SUBSEQ_MIN_SEC = 30.0     # ...and at least 30s longer, so short songs are safe


def align(score: Score, audio_path, subseq: bool | None = None,
          band_sec: float | None = None, tau: float = NOTE_DECAY_TAU,
          chroma_kind: str = "cens", step: str = "classic_p20",
          win_len_smooth: int | None = CENS_SMOOTH,
          verbose: bool = False) -> Alignment:
    """Estimate the score-time -> audio-time map.

    `subseq` lets the score match a *sub*sequence of the audio, so the path's
    endpoints in the audio float freely. That sounds like the right way to
    absorb an intro or outro the tab omits, and on repetitive music it is a
    trap: measured on "The Cranberries - Zombie", whose score (304s) spans
    essentially the whole recording (307s), it squeezed the entire score into
    audio [27.9s, 250.4s] -- mapping bar 0 to 27.9s -- because skipping audio
    at both ends costs nothing and one verse of that song looks much like the
    next. It is off by default for that reason.

    `band_sec` restricts the path to a Sakoe-Chiba band of that many seconds
    around the diagonal. Real drift on this library is seconds, so a band of a
    few tens of seconds cannot exclude a true path but does forbid the
    catastrophic excursions repetitive material invites.
    """
    y = load_mono(audio_path, SR)
    sr = SR
    audio_duration = len(y) / sr

    if subseq is None:
        subseq = (audio_duration > score.duration * AUTO_SUBSEQ_MIN_RATIO
                  and audio_duration - score.duration > AUTO_SUBSEQ_MIN_SEC)
        if subseq and verbose:
            print(f"    auto-subseq ON: audio {audio_duration:.0f}s vs score "
                  f"{score.duration:.0f}s -- the recording contains material the "
                  f"tab does not")

    audio_chroma = audio_chromagram(y, sr, chroma_kind, win_len_smooth)
    frame_sec = HOP / sr

    # Onsets are computed at a much finer hop than the chroma, because their
    # whole purpose is the precision the chromagram does not have.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ONSET_HOP)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                              hop_length=ONSET_HOP, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=ONSET_HOP)
    onset_strength = onset_env[np.clip(onset_frames, 0, len(onset_env) - 1)]
    cymbal_times, cymbal_strength = cymbal_onsets(y, sr, hop=ONSET_HOP)

    n_score_frames = int(np.ceil(score.duration / frame_sec)) + 1
    tab_chroma = score_chromagram(score, frame_sec, n_score_frames, tau=tau)

    # Pool both axes equally so the frame grids stay comparable.
    factor = 1
    while max(tab_chroma.shape[1], audio_chroma.shape[1]) // factor > MAX_DTW_FRAMES:
        factor += 1
    X = _normalize(_pool(tab_chroma, factor))
    Y = _normalize(_pool(audio_chroma, factor))
    pooled_sec = frame_sec * factor

    dtw_kwargs = {}
    if band_sec is not None:
        longest = max(X.shape[1], Y.shape[1])
        dtw_kwargs = {"global_constraints": True,
                      "band_rad": max(band_sec / pooled_sec, 1.0) / longest}

    steps, w_mul, w_add = STEP_SETS[step]
    dtw_kwargs["step_sizes_sigma"] = steps
    dtw_kwargs["weights_mul"] = np.array(w_mul, dtype=float)
    dtw_kwargs["weights_add"] = np.array(w_add, dtype=float)

    if verbose:
        print(f"    frames: score {X.shape[1]}, audio {Y.shape[1]} "
              f"(hop {pooled_sec*1000:.0f}ms, pool x{factor}, "
              f"chroma={chroma_kind}, step={step}, subseq={subseq})")

    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine", subseq=subseq,
                                 **dtw_kwargs)
    wp = wp[::-1]  # librosa returns the path end-first

    si, ai = wp[:, 0], wp[:, 1]

    # Collapse to one audio frame per score frame. Where the path runs
    # horizontally (one score frame matching many audio frames, i.e. the tab is
    # running ahead) the mean is the sensible representative.
    uniq, inv = np.unique(si, return_inverse=True)
    summed = np.bincount(inv, weights=ai)
    counts = np.bincount(inv)
    mapped = summed / counts

    # Fraction of steps that are exactly (1,1). Under the "slope" step set every
    # step already advances both axes, so the old "advances both" test would
    # read 1.00 on every file and be worthless as a confidence signal; this
    # measures how much the path zig-zags between (1,2) and (2,1) instead, and
    # stays comparable across both step sets.
    path_steps = np.diff(wp, axis=0)
    diagonality = (float(np.mean((path_steps[:, 0] == 1) & (path_steps[:, 1] == 1)))
                   if len(path_steps) else 0.0)
    cost_per_step = [1.0 - float(np.dot(X[:, i], Y[:, j])) for i, j in wp]
    path_cost = float(np.mean(cost_per_step)) if cost_per_step else 1.0
    coverage = float((ai.max() - ai.min() + 1) / Y.shape[1]) if len(ai) else 0.0

    return Alignment(
        score_times=uniq * pooled_sec,
        audio_times=mapped * pooled_sec,
        frame_sec=pooled_sec,
        path_cost=path_cost,
        diagonality=diagonality,
        coverage=coverage,
        score_duration=score.duration,
        audio_duration=audio_duration,
        onset_times=onset_times,
        onset_strength=onset_strength,
        cymbal_times=cymbal_times,
        cymbal_strength=cymbal_strength,
        onset_env=onset_env,
        onset_env_times=librosa.frames_to_time(np.arange(len(onset_env)), sr=sr,
                                               hop_length=ONSET_HOP),
    )


def refine_to_onsets(times: np.ndarray, alignment: Alignment,
                     window: float = ONSET_SNAP_WINDOW) -> np.ndarray:
    """Snap each time to the strongest nearby transient.

    DTW and onset detection fail in opposite directions, which is why combining
    them works. DTW knows WHICH bar goes where but reads it off a chromagram
    that has been smoothed over ~1.9s, so it is globally right and locally
    vague. Onset detection knows exactly WHEN a transient happened -- to about
    12ms here -- but has no idea which bar it belongs to.

    So: let DTW choose the neighbourhood, and let the onsets choose the instant
    inside it. In this library a bar line almost always lands on an attack, and
    the author's own correction was phrased in exactly those terms ("the first
    transient of the bass is 3/10ths of a second later").

    The STRONGEST onset in the window is chosen rather than the nearest,
    because a bar line falls on a downbeat: the nearest transient is often a
    quiet ghost note or a ring-out, while the downbeat is the loud one. Points
    with no onset in range are left exactly where DTW put them.
    """
    if alignment.onset_times is None or len(alignment.onset_times) == 0:
        return times

    onsets, strength = alignment.onset_times, alignment.onset_strength
    out = np.asarray(times, dtype=float).copy()
    for k, t in enumerate(out):
        lo, hi = np.searchsorted(onsets, [t - window, t + window])
        if hi <= lo:
            continue
        out[k] = onsets[lo + int(np.argmax(strength[lo:hi]))]
    return out


def refine(score: Score, alignment: Alignment, playback_indices,
           times: np.ndarray, snap_window: float = ONSET_SNAP_WINDOW,
           cymbal_window: float = 0.25, use_cymbals: bool = True
           ) -> tuple[np.ndarray, dict]:
    """Two-tier refinement: cymbal crashes where notated, general onsets elsewhere.

    The tiers are in that order because a crash is a much better landmark than a
    generic onset when the score agrees there should be one. Measured on "Rise
    Against - Savior" bar 11 -- the full-band entry -- the high-frequency
    detector found a single crash at 22.872s with strength 2.2, where the
    general detector offered three candidates of strength 0.5, 0.3 and 0.3
    spread over half a second. The author's own hand-placed point for that bar is
    22.87s.

    Bars the score does not mark with a downbeat cymbal fall through to the
    general onset snap, and bars with no transient in range keep the raw DTW
    position.
    """
    times = np.asarray(times, dtype=float)
    stats = {"cymbal_snapped": 0, "onset_snapped": 0, "unchanged": 0}
    before = times.copy()

    if use_cymbals and alignment.cymbal_times is not None:
        cym_bars = score_cymbal_bars(score)
        times, n = snap_to_cymbals(playback_indices, times, cym_bars,
                                   alignment.cymbal_times, alignment.cymbal_strength,
                                   cymbal_window)
        stats["cymbal_snapped"] = n
        cymbal_handled = np.array([i in cym_bars for i in playback_indices])
    else:
        cymbal_handled = np.zeros(len(times), dtype=bool)

    if snap_window > 0:
        general = refine_to_onsets(times, alignment, snap_window)
        times = np.where(cymbal_handled, times, general)
        stats["onset_snapped"] = int(np.sum((~cymbal_handled) & (times != before)))

    stats["unchanged"] = int(np.sum(times == before))
    return np.maximum.accumulate(times), stats   # snapping must not reorder bars


def sync_points_from_alignment(score: Score, alignment: Alignment,
                               every_n_bars: int = 8,
                               snap_window: float = ONSET_SNAP_WINDOW,
                               tempo_lock: bool = True) -> list[tuple[int, float]]:
    """Sync points for `score` -> [(playback_idx, audio_sec), ...].

    ⚠ BY DEFAULT THIS NO LONGER SAMPLES THE RAW MAP. It samples the
    tempo-regularised one (`tempo_lock.tempo_locked_points`), because the raw map
    read at bar lines produces an audibly unsteady backing track: Guitar Pro plays
    the audio at the tempo implied between consecutive points, and that tempo is a
    DIFFERENCE of two point times, so it amplifies the map's local error.

    Measured over 22 random ground-truth tabs, splitting position error into the
    inaudible constant offset and the `spread` a listener actually hears:

        variant              spread   implied-tempo SD
        raw map, every 4      71ms         8.14
        tempo-locked 20s      90ms         2.35   <- default
        (human maps)                       0.70

    So it trades 19ms of audible spread for a 3.5x steadier track. That is the
    right trade for this project specifically: the author plays bass along to this
    playback, and said so -- *"it's very important for me as a bass player to be
    locked in with the drums."*

    `tempo_lock=False` restores the raw sampling for comparison; `snap_window` is
    then still honoured, but snapping is off by default (see ONSET_SNAP_WINDOW).
    """
    if tempo_lock:
        from .tempo_lock import tempo_locked_points   # local: avoids an import cycle
        pts = tempo_locked_points(score, alignment, every_n_bars=every_n_bars)

        # Then pull each bar line onto the drum transients. Chroma matching says
        # WHICH bar this is and is blind to whether the bar LINE lands on a hit;
        # this closes that, and it is the author's own method made mechanical.
        # Measured over 16 random ground-truth tabs: gridlines within 30ms of an
        # onset peak go from 51.9% to 78.4%, and the median gridline-to-peak
        # distance from 29ms to 11ms.
        if alignment.onset_env is not None and len(pts) >= 3:
            from .grid_align import refine_to_grid
            pts = refine_to_grid(score, pts, alignment.onset_env,
                                 alignment.onset_env_times)
        return pts

    picks = [b for b in score.bars if b.playback_idx % every_n_bars == 0]
    times = np.array([alignment.to_audio(b.start) for b in picks], dtype=float)
    times, _ = refine(score, alignment, [b.playback_idx for b in picks], times,
                      snap_window=snap_window)
    return [(b.playback_idx, float(t)) for b, t in zip(picks, times)]


def cymbal_sync_bars(score: Score, every_n_bars: int = 4) -> list[int]:
    """Bars to place sync points on, preferring downbeat cymbal crashes.

    Rather than sampling the map blindly every N bars, anchor it where the
    evidence is strongest. A bar whose downbeat carries a notated crash is a bar
    whose position can be located in the audio to a few milliseconds; an
    arbitrary bar in the middle of a sustained passage cannot. Bars are still
    taken every `every_n_bars` as a floor, so a stretch with no cymbals is not
    left unanchored.
    """
    cym_bars = set(score_cymbal_bars(score))
    picks = {b.playback_idx for b in score.bars if b.playback_idx % every_n_bars == 0}
    return sorted(picks | cym_bars)
