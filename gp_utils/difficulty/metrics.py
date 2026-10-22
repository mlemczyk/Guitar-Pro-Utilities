"""Measure how hard a Guitar Pro bass tab is to play, from its notes alone.

    from gp_utils.difficulty import measure
    row = measure(Path("Artist - Title.gp"))
    row["predicted_accuracy_pct"], row["notes_per_minute"], row["shifts_per_min"]

Two kinds of number come out of `measure()`:

DESCRIPTIONS of the part, argued from first principles:
  * notes_per_minute over the bars where the bass plays (not BPM, which is
    blind to note values, and not over the whole song, which a long tacet
    intro would dilute);
  * burst density: the 90th-percentile and peak note rate over a sliding
    4-second window, and their ratio to the average -- a part is as hard as
    its hardest passage;
  * fretboard movement: fret travel, string crossings and position shifts
    (moves of SHIFT_FRETS or more) per minute, from string/fret geometry
    rather than pitch intervals, since 5 semitones across strings is the same
    fret and 5 along one string is a real shift. Simultaneous notes are one
    hand position, an open string does not move the fretting hand.
    By default the part is first re-fingered onto a 5-string B-E-A-D-G bass
    (`refinger`), because many tabs are notated in tunings that do not
    transfer; `fingering="notated"` trusts the transcriber's frets instead.

A PREDICTION, fitted to real performances: `predicted_accuracy_pct`, the
note accuracy (correct / (reference + extra), a Jaccard share) one player is
expected to score. It is a per-note logistic in `note_features()`'s features
-- plucking speed capped at 8 notes/sec, sustained 2-second density with a
knee, distinct pitches nearby, interval from the previous note -- plus a
flat rate of spurious notes per playing second:

    recall(note) = 1 / (1 + exp(-(b0 + sum b_k * x_k)))
    predicted    = sum(recall) / (notes + r * playing seconds)

⚠ It is calibrated to ONE player (the author: fingerstyle, 5-string bass),
scored by one note-matching scorer, on 82 takes of 42 songs. Use it to ORDER
tabs by difficulty; its absolute percentages are that player's, not yours.
Because of the extras term it is not monotonic in density: a very sparse part
predicts lower than a moderate one (harder to score cleanly, not to play).

THE FITTED NUMBERS LIVE IN weights.json, not in this file. A refit rewrites
that file and bumps its `measure_version`; anything caching `measure()`
output should key on MEASURE_VERSION. Bump it for any code change here that
moves a number, too.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

from ..parser import parse_song
from .note_features import note_features
from .refinger import STANDARD_5, refinger

WEIGHTS_PATH = Path(__file__).with_name("weights.json")
_W = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))

MEASURE_VERSION = _W["measure_version"]

# v10 bar-density curve, kept for bar_accuracy() and predicted_accuracy(),
# which diagnostics still call; v11 below is what measure() reports.
BAR_CURVE_PLATEAU = _W["v10_bar_curve"]["plateau"]   # recall where density does not bite
BAR_CURVE_KNEE = _W["v10_bar_curve"]["knee"]         # notes/sec at which it starts to
BAR_CURVE_DECAY = _W["v10_bar_curve"]["decay"]       # notes/sec per factor of e above the knee

# v11 per-note model.
V11_INTERCEPT = _W["v11"]["intercept"]
V11_COEF = dict(_W["v11"]["coef"])
BAR_EXTRAS_PER_SEC = _W["v11"]["extras_per_sec"]     # spurious notes per playing second

# |delta fret| at or above this counts as a real position shift rather than a
# move the hand covers with its fingers. Four is the standard
# one-finger-per-fret span; low positions are really a three-fret Simandl
# span, so this is mildly conservative down there by design -- it under-
# counts shifts rather than inventing them.
SHIFT_FRETS = 4

# Sliding window for the burst columns, in seconds of wall clock, stepped
# finely enough that a burst can't hide between two window positions.
#
# 4 seconds is roughly two bars at 120bpm and three at 190, i.e. a phrase
# rather than a beat or a section. Deliberately fixed in wall-clock time
# rather than scaled to the tab's tempo: what makes a passage hard is how
# many notes the hands have to place per second, and notes_per_minute is
# wall-clock too, so the two compose.
BURST_WINDOW_SEC = 4.0
BURST_STEP_SEC = 0.5
# Which percentile of the windowed density counts as "the hard part". The
# maximum alone is one window wide and rides on a single dense bar, so it is
# reported beside this but not used for the ratio.
BURST_PEAK_Q = 0.90


def _percentile_sorted(vals: list, q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def burst_density(notes: list, notes_per_minute: float) -> dict:
    """How much denser the part's hardest passage is than its own average.

    Added 2026-09-10. `notes_per_minute` is a mean over every playing bar, so
    it cannot tell a part that is evenly moderate from one that rests through
    most of its length and then plays a fast run -- and the second is harder,
    because a part is as hard as its hardest passage rather than its average.
    The two songs that exposed this, `Linkin Park - With You` and
    `Jordan Davis - Bar None`, sit within 2 npm of each other (both at the
    library's p3) while their windowed densities differ by a third.

    A part is as hard as its hardest passage, not its average.

    The ratio's denominator is `notes_per_minute`, i.e. the playing-bars-only
    rate, not the notes/total-span rate. Using the raw span would let a long
    tacet intro inflate burstiness on a part that is perfectly even whenever
    it is actually playing, which is not what this is meant to catch.
    """
    starts = sorted({round(float(n["start"]), 4) for n in notes})
    out = {"peak_density_per_min": 0.0, "max_density_per_min": 0.0, "burst_ratio": 0.0}
    if len(starts) < 2 or notes_per_minute <= 0:
        return out

    span = starts[-1] - starts[0]
    per_min = 60.0 / BURST_WINDOW_SEC
    if span <= BURST_WINDOW_SEC:
        # Shorter than one window: the whole part is the window.
        dens = [len(starts) * per_min]
    else:
        dens = []
        t = starts[0]
        end = starts[-1] - BURST_WINDOW_SEC
        while t <= end:
            lo = bisect.bisect_left(starts, t)
            hi = bisect.bisect_left(starts, t + BURST_WINDOW_SEC)
            dens.append((hi - lo) * per_min)
            t += BURST_STEP_SEC
    dens.sort()

    peak = _percentile_sorted(dens, BURST_PEAK_Q)
    out["peak_density_per_min"] = peak
    out["max_density_per_min"] = dens[-1]
    out["burst_ratio"] = peak / notes_per_minute
    return out


def bar_accuracy(density: float) -> float:
    """Expected RECALL in a bar of this local density -- the share of the
    bar's reference notes that get played correctly. This is the numerator
    half of the measure; see bar_extras() for the other."""
    return BAR_CURVE_PLATEAU * math.exp(-max(0.0, density - BAR_CURVE_KNEE) / BAR_CURVE_DECAY)


def bar_extras(seconds: float) -> float:
    """Expected spurious notes in a bar lasting this many seconds.

    Extras -- string noise, ring-out, over-split sustains, re-plucks the tab
    does not call for -- arrive at a roughly fixed rate per second, so their
    cost per reference note explodes as a bar gets sparser. This is what
    `note_accuracy_pct` charges for and what the pre-2026-09-11 curve could
    not see. Since MEASURE_VERSION "8" it is priced as exactly that, a rate
    times time, rather than a per-note tax falling with density.
    """
    return BAR_EXTRAS_PER_SEC * seconds


def note_recall(f: dict) -> float:
    """v11: the chance one note is played, from its note_features() row."""
    eta = V11_INTERCEPT + sum(c * f[k] for k, c in V11_COEF.items())
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))


def predicted_accuracy_v11(features: list, playing_sec: float) -> dict:
    """{"predicted_accuracy_pct", "hard_note_fraction"} from the per-note model.

    `hard_note_fraction` keeps its v10 meaning: the share of notes in
    passages past the sustained-density knee, i.e. "how much of this is too
    fast for me". It ignores the extras half and the walk terms."""
    if not features or playing_sec <= 0:
        return {"predicted_accuracy_pct": None, "hard_note_fraction": None}
    num = sum(note_recall(f) for f in features)
    den = len(features) + bar_extras(playing_sec)
    hard = sum(1 for f in features if f["sus_over"] > 0)
    return {"predicted_accuracy_pct": 100.0 * num / den,
            "hard_note_fraction": hard / len(features)}


def predicted_accuracy(playing_bars: list) -> dict:
    """v10, kept for comparison: no longer what fills the CSV (see
    predicted_accuracy_v11()).

    How well this part is expected to be played, from its own bars.

    {"predicted_accuracy_pct", "hard_note_fraction"}.

    The headline measure, added 2026-09-11. Every other difficulty number in
    this file is a description of the part -- how fast, how far the hand
    moves, how bursty. This one is a *prediction of an outcome*, in the units
    the scorer already reports, and it is the only one calibrated against
    measured performance rather than argued from first principles.

    Each bar's own note density is run through the fitted curve and the
    results combined as a ratio of expected totals, so a dense bar counts
    for every note it puts at risk rather than counting once:

        predicted = sum(n_b * recall(d_b)) / (sum(n_b) + r * sum(seconds_b))

    Note-weighting is what makes this different from taking a percentile of
    the density distribution. A part with one ferocious bar and fifty easy
    ones should not score as hard overall -- but that one bar contains many
    notes, so it pulls its real weight and no more.

    The denominator is the 2026-09-11 change and it is what removed the
    plateau: it charges each bar for the spurious notes its sparseness
    invites, matching score_notes()' own denominator of reference + extra.
    A consequence worth knowing before reading a number: **the measure is no
    longer monotonic in density.** It peaks around 4 notes/sec, and a very
    sparse part now predicts LOWER than a moderate one -- `Linkin Park -
    With You` at 1.6 notes/sec predicts ~85% where it used to predict 95.9%.
    That is not the tab being harder to play; it is the tab being harder to
    score cleanly, which is what note_accuracy_pct measures.

    `hard_note_fraction` is still purely a recall statement: the share of a
    part's notes that live in bars past the knee. It deliberately ignores
    the extras half, so it remains the "how much of this is too fast for me"
    number and is easier to sanity-check by ear.

    Density comes from each bar's real duration rather than a fixed
    bar-length from the tempo, so a tempo change or an odd time signature is
    handled correctly. The curve was calibrated against fixed-length bars,
    which is the same quantity measured slightly more carefully here.
    """
    num = den = hard = 0.0
    for b in playing_bars:
        dur = b["end"] - b["start"]
        n = b["note_count"]
        if dur <= 0 or n <= 0:
            continue
        d = n / dur
        num += n * bar_accuracy(d)
        den += n + bar_extras(dur)
        if d > BAR_CURVE_KNEE:
            hard += n
    # hard_note_fraction is a share of NOTES, so it keeps the plain note
    # count as its denominator rather than the extras-inflated one.
    note_total = sum(b["note_count"] for b in playing_bars
                     if b["note_count"] > 0 and b["end"] - b["start"] > 0)
    if den <= 0 or note_total <= 0:
        return {"predicted_accuracy_pct": None, "hard_note_fraction": None}
    return {"predicted_accuracy_pct": 100.0 * num / den,
            "hard_note_fraction": hard / note_total}


def movement(notes: list) -> dict:
    """Fretting-hand travel and plucking-hand string crossings for one
    tab's note list, as raw totals (the caller divides by playing time).

    Returns zeros for a tab whose notes carry no string/fret data at all,
    rather than raising -- gp_parser emits both as Optional[int] and a tab
    authored without tab notation legitimately has neither."""
    groups = {}
    for n in notes:
        if n.get("string") is None:
            continue
        groups.setdefault(round(n["start"], 4), []).append(n)

    anchors = []  # (string_anchor, fret_anchor or None) in time order
    for start in sorted(groups):
        chord = groups[start]
        strings = [n["string"] for n in chord]
        frets = [n["fret"] for n in chord if n.get("fret")]  # fret 0 == open
        anchors.append((sum(strings) / len(strings),
                        (sum(frets) / len(frets)) if frets else None))

    fret_travel = 0.0
    string_cross = 0.0
    shifts = 0
    fret_pairs = 0
    last_fret = None
    for i, (string_a, fret_a) in enumerate(anchors):
        if i > 0:
            string_cross += abs(string_a - anchors[i - 1][0])
        if fret_a is None:
            # Open-string group: no fretting-hand position, hand stays put.
            continue
        if last_fret is not None:
            delta = abs(fret_a - last_fret)
            fret_travel += delta
            fret_pairs += 1
            if delta >= SHIFT_FRETS:
                shifts += 1
        last_fret = fret_a

    semitone_travel = sum(abs(b["midi"] - a["midi"]) for a, b in zip(notes, notes[1:]))

    return {
        "fret_travel": fret_travel,
        "string_cross": string_cross,
        "shifts": shifts,
        "fret_pairs": fret_pairs,
        "semitone_travel": semitone_travel,
        "onset_groups": len(anchors),
        "string_pairs": max(0, len(anchors) - 1),
        "notes_missing_fret": sum(1 for n in notes if n.get("fret") is None),
    }


def measure(gp_path: Path, fingering: str = "standard5") -> dict:
    """{"notes_per_minute", ...} for one tab, or {"error": "..."} if the
    file can't be parsed or has no playing bars at all.

    `fingering` picks what the movement columns are measured off:
    "standard5" re-fingers the part onto the author's bass (5-string,
    standard B-E-A-D-G) via refinger.py, "notated" trusts the string/fret
    the transcriber wrote. standard5 is the default because only 44% of
    this library's tabs are notated in a tuning that transfers to this
    rig -- see refinger.py's docstring for the survey."""
    song = parse_song(gp_path)
    bars = song["bars"]
    if not bars:
        return {"error": "no bars"}

    playing = [b for b in bars if b["note_count"] > 0 and b["end"] > b["start"]]
    if not playing:
        return {"error": "no bars with notes"}

    playing_sec = sum(b["end"] - b["start"] for b in playing)
    total_sec = sum(max(0.0, b["end"] - b["start"]) for b in bars)
    note_count = sum(b["note_count"] for b in playing)
    if playing_sec <= 0:
        return {"error": "playing bars have zero duration"}

    tuning = song.get("tuning") or ()
    fingered = refinger(song["notes"])
    if fingering == "standard5":
        notes_for_movement = fingered["notes"]
        octave_shifted = fingered["octave_shifted"]
    else:
        notes_for_movement = song["notes"]
        octave_shifted = 0

    mv = movement(notes_for_movement)
    per_min = 60.0 / playing_sec
    npm = note_count / playing_sec * 60.0
    # Timing is unaffected by re-fingering, so this reads the parsed note
    # list rather than the re-fingered one.
    burst = burst_density(song["notes"], npm)
    # v11 is always priced off the B-E-A-D-G re-fingering it was fit on,
    # whatever `fingering` the movement columns use.
    predicted = predicted_accuracy_v11(note_features(fingered["notes"]), playing_sec)

    return {
        # The headline measure -- see predicted_accuracy().
        "predicted_accuracy_pct": predicted["predicted_accuracy_pct"],
        "hard_note_fraction": predicted["hard_note_fraction"],
        "fingering_source": fingering,
        "notated_strings": len(tuning),
        # Whether the transcriber's own fret numbers would have been
        # playable as written on this rig, i.e. their strings are a subset
        # of B-E-A-D-G. Informational: with fingering=standard5 the
        # movement columns no longer depend on it, but it says how far the
        # re-fingering had to move from what the tab shows.
        "notated_transfers": bool(tuning) and set(tuning) <= set(STANDARD_5),
        "octave_shifted": octave_shifted,
        "notes_per_minute": npm,
        # See burst_density() -- how dense the hardest passage is, and how
        # far above the part's own playing-time average that sits.
        "peak_density_per_min": burst["peak_density_per_min"],
        "max_density_per_min": burst["max_density_per_min"],
        "burst_ratio": burst["burst_ratio"],
        "fret_travel_per_min": mv["fret_travel"] * per_min,
        "string_cross_per_min": mv["string_cross"] * per_min,
        "shifts_per_min": mv["shifts"] * per_min,
        # Per-note averages alongside the per-minute rates: the rates are
        # what compose with notes_per_minute (both are costs per second of
        # playing), while these say whether the line itself moves much
        # independent of how fast it is.
        "mean_abs_fret_delta": (mv["fret_travel"] / mv["fret_pairs"]) if mv["fret_pairs"] else 0.0,
        "mean_abs_string_delta": (mv["string_cross"] / mv["string_pairs"]) if mv["string_pairs"] else 0.0,
        "mean_abs_semitone_delta": (mv["semitone_travel"] / (len(song["notes"]) - 1)) if len(song["notes"]) > 1 else 0.0,
        "onset_groups": mv["onset_groups"],
        "notes_missing_fret": mv["notes_missing_fret"],
        "note_count": note_count,
        "playing_bars": len(playing),
        "total_bars": len(bars),
        "playing_sec": playing_sec,
        "total_sec": total_sec,
        # How much of the song the bass is actually in. Not part of the
        # speed metric, but it is what separates "fast part, plays
        # constantly" from "fast part, plays for 30 seconds" -- worth
        # having in the CSV before deciding the metric is the whole story.
        "density_ratio": (playing_sec / total_sec) if total_sec > 0 else 0.0,
        "initial_tempo_bpm": song["initial_tempo_bpm"],
        "synced_to_audio": song["synced_to_audio"],
        "error": "",
    }
