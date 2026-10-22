"""Use drum cymbal accents as alignment landmarks.

The author's own manual method, and the reason it is worth scripting: *"I use the
drum cymbal clashes to synchronize loud transients in the audio."* A crash is
the single most locatable event in a rock mix -- broadband, loud, and with a
near-vertical attack -- and the tab says exactly which bar it lands on.

This module exists because the drum track is otherwise **thrown away entirely**.
`gp_score.pitched_notes` excludes percussion, correctly, since drum-kit MIDI
numbers are instrument slots rather than pitches and would poison a chromagram.
But that leaves the rhythmically most informative track unused.

TWO FACTS FROM THE LIBRARY SHAPE WHAT IS USEFUL HERE:

1. **A notated crash is NOT rare, so it is not by itself a landmark.**
   "Rise Against - Savior" has **392** cymbal accents, of which only **56** land
   on a bar line -- that transcription crashes on continuous eighths. Snapping to
   "the nearest crash" would therefore be snapping to noise.
   **What is informative is a crash ON A DOWNBEAT**, which is what this module
   extracts.

2. **Coverage is good but not universal.** Of 40 ground-truth tabs checked, 38
   have a drum track and 35 notate cymbal accents, at a median of 144 per tab.
   So this is a strong signal on most files and absent on some, and it has to
   degrade to the general onset snap rather than being required.

On the audio side a crash is found with a **high-frequency** onset detector.
Above ~5kHz there is essentially no bass or guitar fundamental, so an onset
envelope restricted to that band responds to cymbals and little else -- which is
what separates a crash from the kick and bass notes surrounding it.
"""

from __future__ import annotations

import librosa
import numpy as np

from .gp_score import Score

# General MIDI percussion. The loud accents a player would sync to, as distinct
# from ride/hi-hat timekeeping, which is continuous and therefore useless as a
# landmark.
CRASH_MIDI = {49, 57, 52, 55}       # crash 1, crash 2, china, splash
CYMBAL_MIDI = CRASH_MIDI | {51, 59, 53}   # + ride 1, ride 2, ride bell

CYMBAL_MIN_HZ = 5000.0
ONSET_HOP = 256


def score_cymbal_bars(score: Score, tolerance: float = 0.02,
                      midi_set: set[int] = CRASH_MIDI) -> dict[int, int]:
    """Playback bars whose DOWNBEAT carries a cymbal accent -> how many.

    `tolerance` is how close to the bar line a hit must be to count as on it.
    The count is returned because a bar with several cymbals struck together
    (crash plus china, say) is a louder and more reliable landmark than one.
    """
    drum_tracks = {t.track_idx for t in score.tracks if t.is_percussion}
    if not drum_tracks:
        return {}

    hits = sorted(n["start"] for n in score.notes
                  if n["track_idx"] in drum_tracks and n["midi"] in midi_set)
    if not hits:
        return {}

    hits_arr = np.array(hits)
    out: dict[int, int] = {}
    for bar in score.bars:
        lo, hi = np.searchsorted(hits_arr, [bar.start - tolerance, bar.start + tolerance])
        if hi > lo:
            out[bar.playback_idx] = int(hi - lo)
    return out


def cymbal_onsets(y: np.ndarray, sr: int, min_hz: float = CYMBAL_MIN_HZ,
                  hop: int = ONSET_HOP) -> tuple[np.ndarray, np.ndarray]:
    """Detect cymbal-like transients: onsets in a high-frequency band only.

    Returns (times, strengths). Restricting the onset envelope to bins above
    `min_hz` is what makes this a cymbal detector rather than a general one --
    bass fundamentals, guitar fundamentals and the kick have essentially no
    energy up there, while a crash has most of its.
    """
    spec = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    band = freqs >= min_hz
    if not band.any():
        return np.array([]), np.array([])

    env = librosa.onset.onset_strength(S=librosa.power_to_db(spec[band] ** 2),
                                       sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                        hop_length=hop, backtrack=True)
    if len(frames) == 0:
        return np.array([]), np.array([])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    return times, env[np.clip(frames, 0, len(env) - 1)]


def snap_to_cymbals(playback_indices, times: np.ndarray, cymbal_bars: dict[int, int],
                    cym_times: np.ndarray, cym_strength: np.ndarray,
                    window: float = 0.25) -> tuple[np.ndarray, int]:
    """Move sync points onto a cymbal crash, but ONLY where the score has one.

    That condition is the whole point. A crash is a superb landmark when the
    notation agrees there should be one on this downbeat, and a trap otherwise:
    in a transcription that crashes on every eighth note, the nearest crash to
    an arbitrary bar line is meaningless.

    Returns the adjusted times and how many points were moved.
    """
    out = np.asarray(times, dtype=float).copy()
    if len(cym_times) == 0 or not cymbal_bars:
        return out, 0

    moved = 0
    for k, (pb_idx, t) in enumerate(zip(playback_indices, out)):
        if pb_idx not in cymbal_bars:
            continue
        lo, hi = np.searchsorted(cym_times, [t - window, t + window])
        if hi <= lo:
            continue
        out[k] = cym_times[lo + int(np.argmax(cym_strength[lo:hi]))]
        moved += 1
    return out, moved
