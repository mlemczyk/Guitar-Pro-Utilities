"""Read a Guitar Pro 8 (.gp) file and extract its bass track as note events.

A .gp file is a zip archive holding a `Content/score.gpif` XML score. This
module reads that XML (read-only: it never writes a .gp) and returns the bass
track as a plain list of notes with absolute start/end times in seconds, MIDI
pitch, and the notated string and fret. No MIDI file I/O is involved.

    from gp_utils.parser import parse_song
    song = parse_song(Path("Artist - Title.gp"))
    song["notes"][0]  # {"start", "end", "midi", "note", "string", "fret"}

GPIF STRUCTURE THIS RELIES ON (GP8 zip/XML format, no namespace on the root
<GPIF> element):

    <GPIF>
      <Score><Title/><Artist/></Score>
      <MasterTrack>
        <Tracks>0 1 2 3</Tracks>            -- track-id order matching the
                                                position of each track's bar
                                                id inside every MasterBar's
                                                own <Bars> list below
        <Automations><Automation><Type>Tempo</Type><Bar/><Value>115 2</Value>
      <Tracks><Track id="0"><Name/><InstrumentSet><Type/></Track>
      <MasterBars><MasterBar><Time>4/4</Time><Bars>0 1 2 3</Bars>  -- one bar
                                                id per track, in Tracks order
      <Bars><Bar id="0"><Voices>0 -1 -1 -1</Voices>  -- -1 = no content this
                                                          bar (full-bar rest)
      <Voices><Voice id="0"><Beats>0 1 2</Beats>
      <Beats><Beat id="0"><Rhythm ref="0"/><Notes>0 1</Notes>  -- Notes
                                                absent entirely on a rest
      <Notes><Note id="0"><Tie destination="true"/>
                <Properties><Property name="Midi"><Number>28</Number>
                            <Property name="String"><String>1</String>
                            <Property name="Fret"><Fret>3</Fret>
      <Rhythms><Rhythm id="0"><NoteValue>Quarter</NoteValue>
                               <AugmentationDot count="1"/>

STRING AND FRET: each note carries the notated `string` and `fret` alongside
its `midi` pitch. Both are Optional[int] and can be None (a note authored
without tab notation, or a tab that omits the property), so a consumer must
skip a note pair where either side is missing rather than assume 0. `fret` is
the notated fret including open strings (0), not an absolute position, and is
NOT cross-checked against `midi` and the track's tuning.

Known simplifications (fine for batch note/rhythm analysis, not intended as a
publishing-grade transcription tool):
  - Tempo automations are treated as step changes at a MasterBar's start,
    ignoring any sub-bar Position fraction or "Linear" (gradual) ramps.
  - Only voice slot 0 of a bar is read. A second notated voice (rare for
    bass) is silently dropped rather than merged.
  - Tuplets are honoured only if the Rhythm element has a child whose tag
    contains "Tuplet" exposing num/den (or numerator/denominator)
    attributes; anything else falls back to the plain NoteValue duration.
  - Grace notes aren't special-cased: they come through as ordinary short
    notes at whatever duration their Rhythm entry specifies.
  - Repeat barlines and alternate (1st/2nd) endings ARE expanded into
    playback order (see _expand_repeats()), but only one non-nested repeat
    span at a time. D.S./D.C./Coda/Segno jumps (`DirectionSign`, about 12% of
    a 2,061-tab library) are NOT expanded: notes past a jump come out in raw
    file order.
  - Tied notes are merged into one note, but repeated same-pitch attacks are
    not: parse_song() emits one note per notated attack.

PITFALLS, each of which once produced plausible but wrong timings:

  * Repeats. `<Repeat start="true"/>` ... `<Repeat end="true" count="N"/>`
    bracket a span that plays N times in total. Walking `<MasterBars>` once
    in file order silently skips the extra passes and shifts every later
    note early. About 13% of tabs in a 2,061-tab library use a repeat.

  * Alternate endings. Guitar Pro writes a volta bracket as an
    `<AlternateEndings>` child of the MasterBar holding a pass bitmask,
    never as `<Ending>`; a search for `<Ending>` finds nothing. Expanding a
    repeat without them plays the 1st ending on every pass and the 2nd once
    more on top: one bar too many per volta. About 6% of tabs use them. A
    "1." bracket with no "2." anywhere is decorative, and is treated as a
    plain repeat. A bracket over several bars is marked on its first bar
    only: the unmarked bars after it, up to the repeat barline, are in the
    same ending, not the body.

  * Sync-point bar index. A SyncPoint automation's `<Bar>` is an index into
    the EXPANDED PLAYBACK ORDER, not a MasterBar index. The two are equal in
    a song with no repeats. Read as a MasterBar index, a sync point on the
    second pass of a repeat fires on the first pass instead and rewinds the
    clock mid-song. A `<Bar>` larger than the MasterBar count is the
    giveaway. Read as a playback index, sync points reproduce their authored
    timestamps exactly."""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NOTE_VALUE_QUARTERS = {
    "DoubleWhole": 8.0, "Whole": 4.0, "Half": 2.0, "Quarter": 1.0,
    "Eighth": 0.5, "16th": 0.25, "32nd": 0.125, "64th": 0.0625, "128th": 0.03125,
}
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class GpParseError(Exception):
    pass


def midi_to_note_name(m: int) -> str:
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


def strip_leading_tags(stem: str) -> str:
    """Strips ALL leading "[...]" tags from a filename stem, so
    "[Hard][Tag] Artist - Title" becomes "Artist - Title".

    Every leading tag, not just the first: stripping one would leave
    "[Tag] Artist" parsing as the artist."""
    return re.sub(r"^(?:\[[^\]]*\]\s*)+", "", stem)


# The name callers used before the move to this package.
strip_preset_tag = strip_leading_tags


def parse_gp_artist_title(path: Path):
    """(artist, title) for a tab filename, splitting on the first " - "
    with its spaces required, after stripping leading "[...]" tags.

    Deliberately strict. A looser split (a bare hyphen with optional spaces,
    or dropping a trailing parenthesised qualifier) breaks real names:

      * a hyphenated artist splits in the wrong place: "Blink-182 - Dick
        Lips" would give artist "Blink", title "182 - Dick Lips" (likewise
        Static-X, O-Zone).
      * a parenthesised suffix that is part of the title is lost: "A Bar Song
        (Tipsy)", "Snow (Hey Oh)".

    A stem with no " - " returns (stem, "")."""
    stem = strip_leading_tags(path.stem)
    artist, sep, title = stem.partition(" - ")
    if not sep:
        return artist.strip(), ""
    return artist.strip(), title.strip()


def _compute_duration_quarters(rhythm_elem) -> float:
    if rhythm_elem is None:
        return 1.0
    note_value = rhythm_elem.findtext("NoteValue") or "Quarter"
    base = NOTE_VALUE_QUARTERS.get(note_value, 1.0)

    dot = rhythm_elem.find("AugmentationDot")
    if dot is not None:
        count = int(dot.get("count", "1"))
        base *= (2.0 - (0.5 ** count))

    for child in rhythm_elem:
        if "Tuplet" in child.tag:
            num = child.get("num") or child.get("numerator")
            den = child.get("den") or child.get("denominator")
            if num and den:
                base *= float(den) / float(num)
            break

    return base


def _alternate_ending_mask(masterbar):
    """The MasterBar's `<AlternateEndings>` value as an int bitmask, or None
    if it carries no volta bracket at all.

    The value is a bitfield over repeat passes, not a pass number: bit 0 set
    means "played on the 1st pass", bit 1 "on the 2nd", and so on, so a bar
    under both a 1. and a 2. bracket reads 3. A plain 1st/2nd ending pair is
    therefore 1 and 2."""
    text = masterbar.findtext("AlternateEndings")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _ending_masks(masterbars, endings):
    """The pass bitmask of each bar in an ending group, with an unmarked bar
    inheriting the mask of the nearest marked bar before it.

    Guitar Pro stores a volta bracket that spans several bars as one
    `<AlternateEndings>` on the bracket's FIRST bar only. In "Sabaton -
    Primo Victoria" bars 84-87 repeat, bar 86 carries "1", the repeat-end
    bar 87 carries nothing, and bar 88 carries "2": the record plays
    84 85 86 87 84 85 88. Reading bar 87 as a body bar played 86 and 87 on
    both passes, two bars too many (found from a cover, 2026-11-10; 47
    spans in 32 library tabs are written this way)."""
    masks, current = [], 0
    for e in endings:
        mk = _alternate_ending_mask(masterbars[e])
        if mk is not None:
            current = mk
        masks.append(current)
    return masks


def _expand_repeats(masterbars):
    """Returns the actual playback order of masterbar indices (a list of
    ints into `masterbars`, with repeated spans appearing more than once),
    built from each MasterBar's `<Repeat start="true"/>` /
    `<Repeat end="true" count="N"/>` markers and its `<AlternateEndings>`
    volta brackets (see PITFALLS in the module docstring). `count` on the
    end marker is the *total* number of times the span plays (2 = played
    twice, i.e. the span repeats once more after the first pass), which
    matches a Guitar Pro-drawn "x2" repeat and standard notation convention.

    Alternate (1st/2nd/...) endings are expanded. The ending group runs
    from the first `<AlternateEndings>` bar inside the span to the repeat
    barline, plus the run of them immediately following it; the bars before
    it are the body. An unmarked bar in the group shares the mask of the
    marked bar before it (Guitar Pro marks only a bracket's first bar; see
    `_ending_masks`). Each pass replays the body and then only those ending
    bars whose mask has that pass's bit set.

    **A volta bracket only counts as a real alternate ending when at least
    one `<AlternateEndings>` bar follows the repeat barline.** A "1."
    bracket drawn over the repeat-end bar with no second ending anywhere is
    decorative (seen in real tabs): treating one as a real 1st ending drops
    that bar from the second pass and loses a bar of time. Those spans fall
    through to the plain-repeat path.

    Single-level only: a start marker while another span is already open
    is treated as re-opening at the new bar (the still-open one is
    discarded) rather than nesting, since no tab examined so far nests
    repeats and a wrong guess here is worse than a flat,
    honest limitation. An end marker with no open start is ignored (plays
    once, same as if the markers weren't there) rather than guessing where
    an implicit start might be."""
    order = []
    start_idx = None
    mark = 0  # where in `order` the currently-open span began
    idx = 0
    total = len(masterbars)
    while idx < total:
        repeat_elem = masterbars[idx].find("Repeat")
        if repeat_elem is not None and repeat_elem.get("start") == "true":
            start_idx = idx
            mark = len(order)
        if repeat_elem is not None and repeat_elem.get("end") == "true" and start_idx is not None:
            try:
                count = max(1, int(repeat_elem.get("count", "2")))
            except ValueError:
                count = 2

            last_ending = idx
            while (last_ending + 1 < total
                   and _alternate_ending_mask(masterbars[last_ending + 1]) is not None):
                last_ending += 1

            # Every pass is rebuilt from scratch below, so drop the single
            # pass already emitted bar-by-bar on the way into the span.
            del order[mark:]

            if last_ending == idx:  # plain repeat, no real alternate ending
                order.extend(list(range(start_idx, idx + 1)) * count)
                idx += 1
            else:
                # The ending group starts at the FIRST marked bar inside the
                # span, not at the first bar of a contiguous marked run: Guitar
                # Pro marks a multi-bar volta on its first bar only, and the
                # unmarked bars after it, up to the repeat barline, belong to
                # the same bracket (see `_ending_masks`).
                marked = [j for j in range(start_idx + 1, idx + 1)
                          if _alternate_ending_mask(masterbars[j]) is not None]
                first_ending = marked[0] if marked else idx + 1
                body = list(range(start_idx, first_ending))
                endings = list(range(first_ending, last_ending + 1))
                masks = _ending_masks(masterbars, endings)
                passes = max(count, max(mk.bit_length() for mk in masks))
                for pass_no in range(passes):
                    order.extend(body)
                    for e, mk in zip(endings, masks):
                        if mk >> pass_no & 1:
                            order.append(e)
                idx = last_ending + 1
            start_idx = None
            continue
        order.append(idx)
        idx += 1
    return order


def find_bass_track(root):
    """Returns (track_id, track_elem) for the track that best matches
    "bass" -- preferring the InstrumentSet/Type field (e.g. "electricBass")
    over the free-text track Name, since Name is whatever the tab author
    typed. Returns (None, None) if nothing looks like a bass track.

    Multi-instrument tabs (full-band exports) can have more than one track
    whose InstrumentSet/Type contains "bass" -- e.g. a keyboardist's track
    tagged "synthBass" alongside the actual bassist's "electricBass" track
    (seen on "Rammstein - Bestrafe Mich": a "Flake Lorenz" synthBass track
    sits before "Oliver Riedel"'s real electricBass track in document
    order). A real stringed-bass instrument type is checked first and wins
    over a synth one regardless of document order.

    "bassoon" is not a bass. Tab authors often voice a vocal melody track
    with a bassoon, and a plain "bass" substring test picked those up
    (Tool - Schism, Godsmack - Voodoo, Hendrix - Hey Joe all measured their
    vocal line).

    Within a tier, the first track in document order is not trusted blindly.
    A track whose name marks it as something other than the main bass part
    ("Bass FX", "Additional Bass", "Bass Solo", "Vocal", "Guitar 2", ...;
    see _SIDE_PART) loses to one whose name doesn't, since authors tag
    those with a bass instrument too. Among what is left the first track
    wins unless another holds over twice its notes: "Rammstein - Rammlied"
    puts a 69-note Drop C intro track ahead of the 1,043-note real part,
    while near-equal pairs (a clean and a 5-string copy of the same line)
    keep document order."""
    tracks_container = root.find("Tracks")
    if tracks_container is None:
        return None, None

    candidates = []
    for track_elem in tracks_container.findall("Track"):
        tid = int(track_elem.get("id"))
        name = (track_elem.findtext("Name") or "").strip().lower()
        instr_type_elem = track_elem.find(".//InstrumentSet/Type")
        instr_type = (instr_type_elem.text or "").strip().lower() if instr_type_elem is not None else ""
        candidates.append((tid, track_elem, name, instr_type))

    def is_bass_type(instr_type):
        return "bass" in instr_type.replace("bassoon", "")

    tiers = [
        [c for c in candidates if is_bass_type(c[3]) and "synth" not in c[3]],
        [c for c in candidates if c[2] == "bass"],
        [c for c in candidates if "bass" in c[2].replace("bassoon", "")
         and "drum" not in c[2] and "drum" not in c[3]],
        [c for c in candidates if is_bass_type(c[3])],
    ]
    for tier in tiers:
        if tier:
            tid, elem, _, _ = _pick_main_part(root, tier)
            return tid, elem
    return None, None


# Track-name words that mark a part other than the main bass line, even when
# the author gave it a bass instrument. "guitar" is excluded after "bass "
# so "Bass Guitar" stays a bass.
_SIDE_PART = re.compile(
    r"\b(vocals?|voice|vox|lyrics|(?<!bass )guitar|gtr|lead|rhythm|synth"
    r"|keys?|key's|piano|organ|strings|fx|effects?|distortion|distorsion"
    r"|dist|overdrive|whammy|delay|additional|addit|overdub|solo|harmonic"
    r"|backing|chorus|misc|loops?|wah|octave)\b")


def _pick_main_part(root, tier):
    """The main bass part among `tier` (see find_bass_track)."""
    plain = [c for c in tier if not _SIDE_PART.search(c[2])] or tier
    if len(plain) == 1:
        return plain[0]
    counts = _note_counts(root)
    first = plain[0]
    best = max(plain, key=lambda c: counts.get(c[0], 0))
    if counts.get(best[0], 0) > 2 * counts.get(first[0], 0):
        return best
    return first


def _note_counts(root):
    """{track_id: notated note count}, each bar counted once (repeats not
    expanded, ties not merged) -- only good for comparing tracks."""
    def by_id(container_tag, item_tag):
        container = root.find(container_tag)
        return {} if container is None else {
            e.get("id"): e for e in container.findall(item_tag)}

    order = (root.findtext("MasterTrack/Tracks") or "").split()
    bars, voices, beats = by_id("Bars", "Bar"), by_id("Voices", "Voice"), by_id("Beats", "Beat")
    counts = dict.fromkeys((int(t) for t in order), 0)
    for mb in root.iterfind("MasterBars/MasterBar"):
        for pos, bar_id in enumerate((mb.findtext("Bars") or "").split()):
            if pos >= len(order) or bar_id not in bars:
                continue
            n = 0
            for v_id in (bars[bar_id].findtext("Voices") or "").split():
                voice = voices.get(v_id)
                if voice is None:
                    continue
                for b_id in (voice.findtext("Beats") or "").split():
                    beat = beats.get(b_id)
                    if beat is not None:
                        n += len((beat.findtext("Notes") or "").split())
            counts[int(order[pos])] += n
    return counts


def parse_song(gp_path: Path) -> dict:
    """Returns {"title", "artist", "bass_track_name", "notes", "bars",
    "warnings", "initial_tempo_bpm", "synced_to_audio", "sync_point_count"}.
    `notes` is a list of {"start", "end", "midi", "note", "string", "fret"}
    dicts, sorted by start time (seconds), "note" being the name ("E1").
    `tuning` is the track's open-string MIDI pitches, low string first.

    `bars` is one entry per *played* bar instance -- a bar inside a repeated
    span appears once per pass, in playback order -- as {"index" (the
    MasterBar's own index, so repeat passes share it), "start", "end",
    "nominal_sec" (what the time signature and tempo say the bar should
    last, as opposed to end-start, which is what its beats actually sum
    to), "note_count" (attacks starting in this bar; a tie continuation
    from the previous bar is not counted, matching `notes`)}. Useful for
    note density over only the bars where the bass plays: a long tacet intro
    or outro would otherwise make a fast song look slow.

    `initial_tempo_bpm` is the song's tempo at bar 0 (120.0 if the GPIF has
    no Tempo automation at all). It is only the *opening* tempo: tempo
    changes are already baked into each note's absolute "start"/"end", but
    there is no bar-by-bar tempo map in the result.

    `synced_to_audio` is True when the GPIF has at least one "SyncPoint"
    automation **inside MasterTrack/Automations**, meaning bars were aligned
    to an imported backing track in Guitar Pro. A raw text search for
    `<Type>SyncPoint</Type>` is NOT a test for this: some tools append a
    placeholder SyncPoint (Bar 0, FrameOffset 0, ModifiedTempo 120) at the
    XML root, where neither Guitar Pro nor this parser looks. In one
    1,557-tab library, 1,283 tabs had only that placeholder and 247 had real
    sync points.

    When synced, every note's "start"/"end" uses the *real, performed* tempo
    at each sync point (which can drift a long way from the authored Tempo
    automation: one song swung 98.8->123.8 BPM against an authored-constant
    118) and is re-anchored to each sync point's exact elapsed time, so a
    performance recorded along to the backing track lines up with the notes.
    Bars before the first sync point use the authored Tempo automation.

    ⚠ DO NOT "FIX" THIS BY SUBTRACTING <BackingTrack><FramePadding>.
    There are two time bases and this function deliberately uses the
    transport one:

      FrameOffset / 44100                    = GP TRANSPORT time -- when the bar
                                               is PLAYED during playback.
      (FrameOffset - FramePadding) / 44100   = position INSIDE THE AUDIO FILE.

    FramePadding is where the audio file's sample 0 sits on the transport
    timeline (negative = the audio has a lead-in before bar 1; seen up to
    -27.3s). Anything recorded while playing along to Guitar Pro's playback
    is in transport time, so the raw FrameOffset is right here: applying the
    padding to a tab with a -26s FramePadding moved its notes 26s away from
    a recording that matched them to within 50ms. Code that analyses the
    DECODED AUDIO needs the audio-file frame, and applies the padding
    itself."""
    if not zipfile.is_zipfile(gp_path):
        raise GpParseError("not a zip archive (not a GP8 .gp file)")

    with zipfile.ZipFile(gp_path) as z:
        if "Content/score.gpif" not in z.namelist():
            raise GpParseError("no Content/score.gpif inside archive")
        raw = z.read("Content/score.gpif")

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise GpParseError(f"XML parse error: {e}")

    score = root.find("Score")
    title = (score.findtext("Title") or "").strip() if score is not None else ""
    artist = (score.findtext("Artist") or "").strip() if score is not None else ""

    track_id, track_elem = find_bass_track(root)
    if track_id is None:
        raise GpParseError("no bass track found")

    tuning_elem = track_elem.find(".//Property[@name='Tuning']/Pitches")
    tuning = tuple(int(x) for x in tuning_elem.text.split()) if (
        tuning_elem is not None and tuning_elem.text) else ()

    master_track = root.find("MasterTrack")
    if master_track is None:
        raise GpParseError("no MasterTrack element")
    track_order = [int(x) for x in (master_track.findtext("Tracks") or "").split()]
    if track_id not in track_order:
        raise GpParseError("bass track id missing from MasterTrack track order")
    bass_pos = track_order.index(track_id)

    tempo_events = []
    sync_points = []  # (bar_idx, real_seconds, modified_tempo), see below
    automations = master_track.find("Automations")
    if automations is not None:
        for automation in automations.findall("Automation"):
            atype = automation.findtext("Type")
            if atype == "Tempo":
                bar_idx = int(automation.findtext("Bar") or "0")
                value_text = (automation.findtext("Value") or "120").split()
                try:
                    bpm = float(value_text[0])
                except (ValueError, IndexError):
                    continue
                tempo_events.append((bar_idx, bpm))
            elif atype == "SyncPoint":
                # Guitar Pro's "audio sync point" feature: specific bars
                # aligned to an imported backing track, so GP knows the
                # *real* elapsed time and *real* (performed) tempo at that
                # bar, which can drift a long way from the authored Tempo
                # automation above.
                # <Value><FrameOffset> is in samples at a *fixed* 44100Hz
                # reference rate, independent of the embedded backing
                # track's own sample rate (e.g. 48000Hz): cross-checking
                # FrameOffset deltas against ModifiedTempo across 5 songs,
                # dividing by 44100 always landed on a whole number of beats
                # at that tempo, exactly.
                # **<Bar> is an index into the expanded PLAYBACK order, not
                # a MasterBar index** (see PITFALLS in the module docstring).
                # The two coincide in a song with no repeats.
                try:
                    playback_idx = int(automation.findtext("Bar") or "0")
                    frame_offset = float(automation.findtext("Value/FrameOffset"))
                    modified_tempo = float(automation.findtext("Value/ModifiedTempo"))
                except (TypeError, ValueError):
                    continue
                sync_points.append((playback_idx, frame_offset / 44100.0, modified_tempo))
    tempo_events.sort()
    if not tempo_events:
        tempo_events = [(0, 120.0)]
    sync_points.sort()
    sync_seconds_by_playback_idx = {}
    for playback_idx, real_sec, _mtempo in sync_points:
        sync_seconds_by_playback_idx.setdefault(playback_idx, real_sec)

    def bpm_for_masterbar(idx):
        bpm = tempo_events[0][1]
        for bar_idx, b in tempo_events:
            if bar_idx <= idx:
                bpm = b
            else:
                break
        return bpm

    def bpm_for_playback_idx_synced(mb_idx, playback_idx):
        """Same step-function idea as bpm_for_masterbar(), but sourced from
        SyncPoint automations' real (ModifiedTempo) values instead of the
        authored Tempo automation -- used from the first sync point
        onward; bars before it still use bpm_for_masterbar()'s authored
        tempo, since there's no real-audio correction to apply yet.

        Takes both indices because the authored Tempo automation is keyed by
        MasterBar while a sync point is keyed by playback position."""
        bpm = bpm_for_masterbar(mb_idx)
        for sync_idx, _real_sec, mtempo in sync_points:
            if sync_idx <= playback_idx:
                bpm = mtempo
            else:
                break
        return bpm

    def by_id(container_tag, item_tag):
        container = root.find(container_tag)
        if container is None:
            return {}
        return {int(e.get("id")): e for e in container.findall(item_tag)}

    bars_by_id = by_id("Bars", "Bar")
    voices_by_id = by_id("Voices", "Voice")
    beats_by_id = by_id("Beats", "Beat")
    notes_by_id = by_id("Notes", "Note")
    rhythms_by_id = by_id("Rhythms", "Rhythm")

    masterbars_container = root.find("MasterBars")
    masterbars = masterbars_container.findall("MasterBar") if masterbars_container is not None else []
    playback_order = _expand_repeats(masterbars)

    time_sec = 0.0
    open_notes_by_string = {}
    notes_out = []
    bars_out = []
    warnings = []

    stale_syncs = sorted(i for i in sync_seconds_by_playback_idx if i >= len(playback_order))
    if stale_syncs:
        warnings.append(
            f"{len(stale_syncs)} sync point(s) point past the end of the expanded "
            f"playback order ({len(playback_order)} bars); ignored. This usually "
            f"means the repeat expansion produced too few bars.")

    for playback_idx, mb_idx in enumerate(playback_order):
        masterbar = masterbars[mb_idx]
        if playback_idx in sync_seconds_by_playback_idx:
            # Ground truth from the audio sync point: snap to the real
            # elapsed time at this bar's start, correcting any drift
            # accumulated since the previous sync point (or the song's
            # start) rather than trusting the tempo-based accumulation
            # alone. Keyed by position in the playback order, so a sync
            # point authored against the second pass through a repeated
            # span lands on that pass rather than rewinding the clock to
            # the first one.
            time_sec = sync_seconds_by_playback_idx[playback_idx]
        bpm = (bpm_for_playback_idx_synced(mb_idx, playback_idx) if sync_points
               else bpm_for_masterbar(mb_idx))
        sec_per_quarter = 60.0 / bpm if bpm > 0 else 0.5

        time_text = masterbar.findtext("Time") or "4/4"
        try:
            num_s, den_s = time_text.split("/")
            ts_num, ts_den = float(num_s), float(den_s)
        except ValueError:
            ts_num, ts_den = 4.0, 4.0
        fallback_bar_sec = ts_num * (4.0 / ts_den) * sec_per_quarter

        def record_bar(start, end, count):
            bars_out.append({"index": mb_idx, "start": round(start, 4), "end": round(end, 4),
                             "nominal_sec": round(fallback_bar_sec, 4), "note_count": count})

        bar_ids = [int(x) for x in (masterbar.findtext("Bars") or "").split()]
        bar_elem = bars_by_id.get(bar_ids[bass_pos]) if bass_pos < len(bar_ids) else None
        if bar_elem is None:
            record_bar(time_sec, time_sec + fallback_bar_sec, 0)
            time_sec += fallback_bar_sec
            continue

        voice_ids = [int(x) for x in (bar_elem.findtext("Voices") or "-1").split()]
        voice_id = voice_ids[0] if voice_ids else -1
        voice_elem = voices_by_id.get(voice_id) if voice_id != -1 else None
        if voice_elem is None:
            record_bar(time_sec, time_sec + fallback_bar_sec, 0)
            time_sec += fallback_bar_sec
            continue

        beat_ids = [int(x) for x in (voice_elem.findtext("Beats") or "").split()]
        cursor = time_sec
        notes_before = len(notes_out)
        for beat_id in beat_ids:
            beat_elem = beats_by_id.get(beat_id)
            if beat_elem is None:
                continue

            rhythm_ref_elem = beat_elem.find("Rhythm")
            rhythm_id = int(rhythm_ref_elem.get("ref")) if rhythm_ref_elem is not None else None
            rhythm_elem = rhythms_by_id.get(rhythm_id) if rhythm_id is not None else None
            dur_sec = _compute_duration_quarters(rhythm_elem) * sec_per_quarter

            notes_elem = beat_elem.find("Notes")
            note_ids = [int(x) for x in notes_elem.text.split()] if notes_elem is not None and notes_elem.text else []

            for note_id in note_ids:
                note_elem = notes_by_id.get(note_id)
                if note_elem is None:
                    continue
                midi_elem = note_elem.find(".//Property[@name='Midi']/Number")
                if midi_elem is None or not midi_elem.text:
                    continue
                midi_num = int(midi_elem.text)

                string_elem = note_elem.find(".//Property[@name='String']/String")
                string_num = int(string_elem.text) if string_elem is not None and string_elem.text else None

                fret_elem = note_elem.find(".//Property[@name='Fret']/Fret")
                fret_num = int(fret_elem.text) if fret_elem is not None and fret_elem.text else None

                tie_elem = note_elem.find("Tie")
                is_tie_dest = tie_elem is not None and tie_elem.get("destination") == "true"

                if is_tie_dest and string_num in open_notes_by_string:
                    open_notes_by_string[string_num]["end"] = round(cursor + dur_sec, 4)
                else:
                    new_note = {"start": round(cursor, 4), "end": round(cursor + dur_sec, 4), "midi": midi_num,
                                "string": string_num, "fret": fret_num}
                    notes_out.append(new_note)
                    open_notes_by_string[string_num] = new_note

            cursor += dur_sec

        record_bar(time_sec, cursor, len(notes_out) - notes_before)
        time_sec = cursor

    notes_out.sort(key=lambda n: n["start"])
    for n in notes_out:
        n["note"] = midi_to_note_name(n["midi"])

    return {
        "title": title,
        "artist": artist,
        "bass_track_name": (track_elem.findtext("Name") or "").strip(),
        # Open-string MIDI pitches of the bass track, low string first --
        # what the notated `fret` numbers are relative to. () when the
        # track declares no Tuning property. Don't assume standard tuning:
        # one 1,557-tab library had 51 distinct bass tunings.
        "tuning": tuning,
        "notes": notes_out,
        "bars": bars_out,
        "warnings": warnings,
        "initial_tempo_bpm": (bpm_for_playback_idx_synced(0, 0) if sync_points else tempo_events[0][1]),
        "synced_to_audio": bool(sync_points),
        "sync_point_count": len(sync_points),
    }


def discover_gp_files(input_dir: Path):
    return sorted(input_dir.glob("*.gp"))
