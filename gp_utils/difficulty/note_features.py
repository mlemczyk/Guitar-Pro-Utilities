"""Per-note difficulty features for a tab's bass part.

Shared by measure() (which prices every note through the fitted v11 model)
and the fit that produces weights.json. One definition, so the fit and the
measurement can never measure a note differently.

Input is the RE-FINGERED note list from refinger.py (string 0 = low B), so
the positions are the easiest B-E-A-D-G fingering, not the transcriber's.

The features follow the author's own definition of a hard tab (2026-10-20):
a passage past the right hand's top plucking speed, or one where the left
hand has to walk awkwardly and/or the right hand jump strings. Each was
kept only if it predicted missed notes in recorded takes.

    speed       1 / gap to the previous onset, notes/sec, capped at 16
    rh_over     max(0, speed - RH_LIMIT): the plucking-speed ceiling
    distinct    distinct pitches among the note and its WINDOW neighbours
                either side (how much the line walks rather than repeats)
    interval    |semitones| from the previous note
    lh_travel   frets the fretting hand moves per second over that window
    rh_cross    strings the plucking hand crosses per second over that window
    sustained   notes per second over the 2 seconds around the note -- v10's
                bar density, kept because a long run at a moderate speed is
                hard in a way one quick pair of notes is not (endurance)
    sus_over    max(0, sustained - SUSTAINED_KNEE)
"""

import bisect

RH_LIMIT = 8.0      # notes/sec; the probe's best hinge (6, 10 and 12 lost)
SUSTAINED_KNEE = 4.24   # notes/sec; v10's fitted knee
WINDOW = 4          # notes either side
SPEED_CAP = 16.0


def note_features(fingered: list) -> list:
    """One feature dict per note, in (start, midi) order, each also carrying
    the note's own "start" and "midi" so a caller can join on them."""
    notes = sorted(fingered, key=lambda n: (n["start"], n["midi"]))
    starts = [n["start"] for n in notes]
    out = []
    for i, n in enumerate(notes):
        # gap to the previous ONSET: a chord tone is measured from the onset
        # before its chord, not from its own chord-mate at zero distance
        j = bisect.bisect_left(starts, n["start"] - 1e-4) - 1
        gap = (n["start"] - starts[j]) if j >= 0 else 9.0
        gap = max(gap, 1e-3)
        speed = min(1.0 / gap, SPEED_CAP)
        prev = notes[i - 1] if i else None
        win = notes[max(0, i - WINDOW):i + WINDOW + 1]
        win_sec = max(win[-1]["start"] - win[0]["start"], 0.25)
        lh_moves, last_f = 0, None
        for w in win:
            if w["fret"] > 0:              # an open string does not move the hand
                if last_f is not None:
                    lh_moves += abs(w["fret"] - last_f)
                last_f = w["fret"]
        rh = sum(abs(b["string"] - a["string"]) for a, b in zip(win, win[1:]))
        sustained = (bisect.bisect_left(starts, n["start"] + 1.0)
                     - bisect.bisect_left(starts, n["start"] - 1.0)) / 2.0
        out.append({
            "speed_cap": min(speed, RH_LIMIT),
            "sustained": sustained,
            "sus_over": max(0.0, sustained - SUSTAINED_KNEE),
            "start": n["start"], "midi": n["midi"],
            "speed": speed,
            "rh_over": max(0.0, speed - RH_LIMIT),
            "distinct": len({w["midi"] for w in win}),
            "interval": abs(n["midi"] - prev["midi"]) if prev else 0,
            "lh_travel": lh_moves / win_sec,
            "rh_cross": rh / win_sec,
        })
    return out
