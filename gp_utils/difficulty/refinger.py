#!/usr/bin/env python3
"""
refinger.py

Re-fingers a tab's notes onto the bass this project is actually played on
-- a 5-string in standard B-E-A-D-G tuning -- instead of trusting the
string/fret the transcriber notated.

WHY THIS EXISTS. gp_parser emits each note's notated `string` and `fret`,
which is the transcriber's fingering on the transcriber's instrument. A
2026-09-09 survey of the bass tracks in GuitarProTabs/ found that is a
different instrument most of the time:

     448  4-string  E1 A1 D2 G2      (standard 4 -- transfers as written)
     235  5-string  B0 E1 A1 D2 G2   (this rig -- transfers as written)
     177  4-string  C1 G1 C2 F2
     166  4-string  D1 A1 D2 G2
     159  4-string  D#1 G#1 C#2 F#2
     ... 51 distinct tunings in all; 1273 four-string, 326 five-or-more

Exactly 236 of 1557 are in B-E-A-D-G itself and a further 450 are in a
subset of it (standard 4-string and the like), so 686 tabs (44%) have a
notated fingering playable as written here. For the other 871 the notated
fret numbers describe positions that do not exist on this bass: a drop-C
tab's open lowest string is fret 1 on the B string, and every fret number
in that tab is offset accordingly. Measuring "how far does the hand
travel" off those numbers would be measuring somebody else's hand on
somebody else's neck.

WHY RE-FINGER EVEN THE TABS ALREADY IN B-E-A-D-G. The author re-transcribes
a tab to B-E-A-D-G by hand when it arrives in some other tuning, so the
236 B-E-A-D-G tabs are effectively the curated/played subset -- all 15
recorded songs with a score are in that group, which is the practice
showing up in the data rather than a coincidence. That makes their
notated positions a fingering for the right instrument, and the obvious
move would be to trust them. **Measured, and it is the wrong move:** the
same correlation fit run off notated positions instead of re-fingered ones
is worse on every movement feature (fret travel -0.246 vs -0.396, string
crossing +0.039 -- the wrong sign -- vs -0.407), on exactly these
already-in-tune tabs. The reading is that a notated position records how a
part was *written down*, which includes arbitrary transcription habits,
while the DP records how much movement the part *cannot avoid*. A part is
not hard because someone transcribed it awkwardly. `--fingering notated`
keeps the other behaviour available for re-checking this as the sample
grows.

Pitch is unaffected and always was -- gp_parser reads the absolute `Midi`
property, not fret + tuning -- so nothing outside the difficulty measure
is touched by any of this.

WHAT IT DOES. For each note it enumerates every (string, fret) on this
bass that produces that pitch, then picks the sequence minimising total
hand movement by dynamic programming over the whole part. That is the
standard formulation in the guitar-fingering literature (fingering as a
shortest-path/cost-minimisation problem), and it is the right notion of
difficulty here: a part is as hard as its *easiest available* fingering,
not as hard as the one a transcriber happened to write down. It also
removes transcriber-fingering noise as a variable, which is what makes
two tabs comparable to each other at all.

TRANSITION COST is |delta fret| + STRING_COST * |delta string|, plus a
small POSITION_PENALTY per fret of absolute position to break ties toward
low positions the way a player would. Both constants only decide *which*
fingering gets chosen; they are not the difficulty measure itself, which
is computed from the resulting positions by metrics.py.

KNOWN SIMPLIFICATIONS:
  - Monophonic. Simultaneous notes are fingered off the group's lowest
    note and the rest follow on distinct strings where they fit. Double
    stops are 0.65% of notes corpus-wide (9,096 of 1,393,393) and only 23
    of 1557 tabs exceed 1.15 notes per onset, so this is close to inert.
  - Open strings are treated as ordinary fret-0 positions inside the DP,
    which mildly biases it toward using them -- as a player would. The
    "an open string does not move the fretting hand" rule is applied by
    the measurement pass afterwards, not here.
  - A note outside this bass's range (below B0, or above the 24th fret of
    the G string) is moved by whole octaves until it fits and counted in
    `octave_shifted`. Tabs tuned below B0 exist in this library (e.g. the
    38 tabs in A#0-F1-A#1-D#2), and an octave displacement is what a
    bassist would actually do with those notes.
"""

STANDARD_5 = (23, 28, 33, 38, 43)   # B0, E1, A1, D2, G2 -- open-string MIDI
MAX_FRET = 24                        # Yamaha TRBX305

STRING_COST = 0.5
POSITION_PENALTY = 0.02


def positions_for(midi: int, tuning=STANDARD_5, max_fret: int = MAX_FRET):
    """Every (string_index, fret) on this bass that sounds `midi`, low
    string first. Empty if the pitch is out of range entirely."""
    out = []
    for s, open_midi in enumerate(tuning):
        fret = midi - open_midi
        if 0 <= fret <= max_fret:
            out.append((s, fret))
    return out


def _fit_to_range(midi: int, tuning=STANDARD_5, max_fret: int = MAX_FRET):
    """(midi, shifted) with `midi` moved by whole octaves into playable
    range -- see KNOWN SIMPLIFICATIONS."""
    lo, hi = tuning[0], tuning[-1] + max_fret
    shifted = False
    while midi < lo:
        midi += 12
        shifted = True
    while midi > hi:
        midi -= 12
        shifted = True
    return midi, shifted


def refinger(notes: list, tuning=STANDARD_5, max_fret: int = MAX_FRET) -> dict:
    """Assigns every note a (string, fret) on `tuning`, minimising total
    hand movement across the part.

    Returns {"notes": [...], "octave_shifted": int, "unplayable": int} --
    the note dicts are copies with `string`/`fret` replaced, so the
    caller's notated values are left intact.
    """
    if not notes:
        return {"notes": [], "octave_shifted": 0, "unplayable": 0}

    # Group simultaneous notes; the group's lowest note drives the DP.
    groups = {}
    for n in notes:
        groups.setdefault(round(n["start"], 4), []).append(n)
    ordered = [sorted(groups[t], key=lambda n: n["midi"]) for t in sorted(groups)]

    octave_shifted = 0
    unplayable = 0
    candidates = []
    for chord in ordered:
        midi, shifted = _fit_to_range(chord[0]["midi"], tuning, max_fret)
        octave_shifted += shifted
        pos = positions_for(midi, tuning, max_fret)
        if not pos:                       # cannot happen after _fit_to_range
            unplayable += 1               # kept as a guard, not dead weight
            pos = [(0, 0)]
        candidates.append(pos)

    # Viterbi over (string, fret) states.
    prev_cost = {p: POSITION_PENALTY * p[1] for p in candidates[0]}
    back = []
    for i in range(1, len(candidates)):
        cur_cost, cur_back = {}, {}
        for p in candidates[i]:
            best, best_prev = None, None
            for q, qc in prev_cost.items():
                c = qc + abs(p[1] - q[1]) + STRING_COST * abs(p[0] - q[0]) \
                    + POSITION_PENALTY * p[1]
                if best is None or c < best:
                    best, best_prev = c, q
            cur_cost[p], cur_back[p] = best, best_prev
        back.append(cur_back)
        prev_cost = cur_cost

    path = [min(prev_cost, key=prev_cost.get)]
    for cur_back in reversed(back):
        path.append(cur_back[path[-1]])
    path.reverse()

    out = []
    for chord, (s, fret) in zip(ordered, path):
        anchor = dict(chord[0], string=s, fret=fret)
        out.append(anchor)
        # Remaining notes of a double stop: nearest position on a distinct
        # higher string, else left on the anchor's own string.
        used = {s}
        for extra in chord[1:]:
            midi, shifted = _fit_to_range(extra["midi"], tuning, max_fret)
            octave_shifted += shifted
            options = [p for p in positions_for(midi, tuning, max_fret) if p[0] not in used]
            if options:
                s2, f2 = min(options, key=lambda p: abs(p[1] - fret))
                used.add(s2)
            else:
                s2, f2 = s, fret
            out.append(dict(extra, string=s2, fret=f2))

    out.sort(key=lambda n: n["start"])
    return {"notes": out, "octave_shifted": octave_shifted, "unplayable": unplayable}
