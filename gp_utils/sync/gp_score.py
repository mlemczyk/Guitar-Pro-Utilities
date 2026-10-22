"""Read a Guitar Pro .gp archive as a *score*: every track's notes laid out on
the tab's own authored tempo map, plus whatever sync points a human already
placed, plus the embedded backing-track mp3.

This is deliberately NOT `gp_utils.parser`, and the difference is
the whole point of this package:

  * `parser.parse_song()` returns the **bass track only**, and it **applies**
    sync points when it computes each note's time -- it answers "when is this
    note heard during Guitar Pro playback", which is what scoring a
    play-along recording needs.
  * `read_score()` here returns **all pitched tracks** at **authored** timing,
    i.e. the tab played back on its own tempo map with no audio correction at
    all. That is the thing we are trying to find a mapping *from*.

Asking the parser for this would give a score already warped by the answer.

Two facts about GP's sync-point format are load-bearing and were expensive to
learn; both are documented in `gp_utils.parser`'s module docstring:

  * `<Bar>` is an index into the **expanded playback order**, not a MasterBar
    index. They coincide only in a song with no repeats.
  * `<Value><FrameOffset>` is in samples at a **fixed 44100Hz** reference rate,
    independent of the backing track's own sample rate.

And one this package had to find for itself:

  * `<BackingTrack><FramePadding>` is a **track-level offset that every sync
    point is relative to**, in the same 44100Hz samples. A bar's true position
    in the audio is `(FrameOffset - FramePadding) / 44100`, not `FrameOffset /
    44100`. This is the author's "linear pad": one number for the whole song,
    stored separately from the per-bar sync points.

    It is non-zero on **188 of the 1530** tabs with audio in one library,
    reaching +2.367s on "Pennywise - Fuck Authority".

    ⚠ `gp_utils.parser` ignores it ON PURPOSE, and is right to. There are two
    time bases: `FrameOffset / 44100` is Guitar Pro TRANSPORT time (when the
    bar is played), and `(FrameOffset - FramePadding) / 44100` is the position
    INSIDE THE AUDIO FILE. A recording made while playing along to Guitar Pro
    is in transport time, which is what the parser serves. This package works
    on the decoded audio, so it needs the second one and applies the padding
    itself. Neither is a bug in the other.

    It was found here by measuring: the aligner showed a per-file constant
    error (+896ms on Zombie, -2030ms on Pennywise) with a *spread around that
    constant* of only 37-63ms, which is the signature of a missing offset term
    rather than a bad alignment.

Percussion tracks are excluded from `notes`. Their MIDI numbers are drum-kit
slots rather than pitches, so feeding them to a chroma feature would be
injecting noise shaped like harmony.
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from ..parser import (
    GpParseError,
    _compute_duration_quarters,
    _expand_repeats,
)

SYNC_FRAME_RATE = 44100.0  # fixed reference rate for <FrameOffset>, not the mp3's rate


@dataclass
class SyncPoint:
    """One alignment between a playback bar and the audio."""

    playback_idx: int          # index into the EXPANDED playback order
    audio_sec: float           # (FrameOffset - FramePadding) / 44100 -- see below
    frame_offset: int          # the raw <FrameOffset>, before padding
    modified_tempo: float      # the real performed tempo GP inferred there
    original_tempo: float
    bar_occurrence: int

    @property
    def is_default_anchor(self) -> bool:
        """True for the bar-0 / FrameOffset-0 point Guitar Pro writes on import.

        ⚠ THIS IS NOT A HAND-PLACED POINT AND MUST NOT BE COUNTED AS ONE.
        Measured across the library: **all 147 genuinely hand-synced tabs carry
        it in addition to their real points, and 100 further tabs carry nothing
        else** -- those 100 are effectively unsynced, and treating the anchor as
        human work excluded every one of them from the survey for no reason.

        It also produced a false finding. A second-opinion check flagged 16 tabs
        as "the anchor is misplaced", by up to 4.4s; all 16 were this default,
        and in 13 of them the reported deviation matched `FramePadding` to
        within 1ms. The check was re-reporting the pad and calling it the author's
        error. `gp_parser` meanwhile anchors bar 0 at `FrameOffset / 44100` = 0,
        so a default anchor shifts the reference by exactly nothing and no cover
        score was ever affected -- confirmed on "Dasha - Austin", whose tab has
        1238ms of padding and whose covers score a bias of -0.5ms.
        """
        return self.playback_idx == 0 and self.frame_offset == 0


@dataclass
class Track:
    track_idx: int             # position in MasterTrack/Tracks, i.e. the <Bars> column
    name: str
    instrument: str
    is_percussion: bool


@dataclass
class PlaybackBar:
    playback_idx: int
    mb_idx: int                # which MasterBar this is a visit to
    start: float               # authored score time, seconds
    end: float
    bpm: float
    time_sig: tuple[int, int]

    @property
    def quarters(self) -> float:
        """Length in quarter notes -- the unit tempo is expressed in."""
        num, den = self.time_sig
        return num * (4.0 / den)


@dataclass
class Score:
    path: Path
    title: str
    artist: str
    tracks: list[Track]
    bars: list[PlaybackBar]
    notes: list[dict] = field(default_factory=list)   # {start,end,midi,track_idx}
    human_sync: list[SyncPoint] = field(default_factory=list)
    frame_padding: int = 0        # <BackingTrack><FramePadding>, 44100Hz samples
    audio_name: str | None = None
    original_audio_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def pad_sec(self) -> float:
        """The track-level linear pad, in seconds. See the module docstring."""
        return self.frame_padding / SYNC_FRAME_RATE

    @property
    def real_sync(self) -> list[SyncPoint]:
        """Sync points a human actually placed -- the default anchor removed.

        This is what every consumer in this package should use to decide
        "is this tab synced". `human_sync` keeps the raw list so the file
        stays inspectable.
        """
        return [p for p in self.human_sync if not p.is_default_anchor]

    @property
    def duration(self) -> float:
        return self.bars[-1].end if self.bars else 0.0

    @property
    def pitched_notes(self) -> list[dict]:
        drums = {t.track_idx for t in self.tracks if t.is_percussion}
        return [n for n in self.notes if n["track_idx"] not in drums]


def _text(elem, path, default=""):
    if elem is None:
        return default
    found = elem.findtext(path)
    return default if found is None else found.strip()


def _is_percussion(track_elem) -> bool:
    itype = _text(track_elem, "InstrumentSet/Type").lower()
    if itype:
        return "drum" in itype or "perc" in itype
    name = _text(track_elem, "Name").lower()
    return any(w in name for w in ("drum", "percussion", "cymbal"))


def read_score(gp_path: Path | str) -> Score:
    """Parse `gp_path` into a Score at authored timing. Raises GpParseError."""
    gp_path = Path(gp_path)
    if not zipfile.is_zipfile(gp_path):
        raise GpParseError("not a zip archive (not a GP8 .gp file)")

    with zipfile.ZipFile(gp_path) as z:
        names = z.namelist()
        if "Content/score.gpif" not in names:
            raise GpParseError("no Content/score.gpif inside archive")
        raw = z.read("Content/score.gpif")
        audio_names = [n for n in names
                       if n.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac"))]

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise GpParseError(f"XML parse error: {e}")

    score_elem = root.find("Score")
    title = _text(score_elem, "Title")
    artist = _text(score_elem, "Artist")

    master_track = root.find("MasterTrack")
    if master_track is None:
        raise GpParseError("no MasterTrack element")

    track_order = [int(x) for x in (master_track.findtext("Tracks") or "").split()]
    track_elems = {int(t.get("id")): t for t in root.findall("Tracks/Track")}
    tracks = [
        Track(
            track_idx=pos,
            name=_text(track_elems.get(tid), "Name"),
            instrument=_text(track_elems.get(tid), "InstrumentSet/Type"),
            is_percussion=_is_percussion(track_elems.get(tid)),
        )
        for pos, tid in enumerate(track_order)
        if tid in track_elems
    ]

    backing = root.find("BackingTrack")
    try:
        frame_padding = int(_text(backing, "FramePadding", "0") or "0")
    except ValueError:
        frame_padding = 0

    tempo_events, human_sync = _read_automations(master_track, frame_padding)
    if not tempo_events:
        tempo_events = [(0, 120.0)]

    def bpm_at(mb_idx: int) -> float:
        bpm = tempo_events[0][1]
        for bar_idx, value in tempo_events:
            if bar_idx <= mb_idx:
                bpm = value
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
    masterbars = (masterbars_container.findall("MasterBar")
                  if masterbars_container is not None else [])
    if not masterbars:
        raise GpParseError("no MasterBars")
    playback_order = _expand_repeats(masterbars)

    warnings: list[str] = []
    bars_out: list[PlaybackBar] = []
    notes_out: list[dict] = []
    time_sec = 0.0

    for playback_idx, mb_idx in enumerate(playback_order):
        masterbar = masterbars[mb_idx]
        bpm = bpm_at(mb_idx)
        sec_per_quarter = 60.0 / bpm if bpm > 0 else 0.5

        time_text = masterbar.findtext("Time") or "4/4"
        try:
            num_s, den_s = time_text.split("/")
            ts_num, ts_den = int(num_s), int(den_s)
        except ValueError:
            ts_num, ts_den = 4, 4
        bar_sec = ts_num * (4.0 / ts_den) * sec_per_quarter

        bars_out.append(PlaybackBar(
            playback_idx=playback_idx, mb_idx=mb_idx,
            start=time_sec, end=time_sec + bar_sec,
            bpm=bpm, time_sig=(ts_num, ts_den),
        ))

        # Every track lays its own beats out from the bar's start. The bar's
        # NOMINAL length is authoritative rather than whichever track's beats
        # happen to fill it -- tracks must not drift apart from each other, and
        # an under-filled bar in one track is common in real tabs.
        bar_ids = [int(x) for x in (masterbar.findtext("Bars") or "").split()]
        for track in tracks:
            if track.track_idx >= len(bar_ids):
                continue
            bar_elem = bars_by_id.get(bar_ids[track.track_idx])
            if bar_elem is None:
                continue
            for voice_id in (int(x) for x in (bar_elem.findtext("Voices") or "-1").split()):
                voice_elem = voices_by_id.get(voice_id) if voice_id != -1 else None
                if voice_elem is None:
                    continue
                _lay_out_voice(
                    voice_elem, beats_by_id, notes_by_id, rhythms_by_id,
                    start=time_sec, sec_per_quarter=sec_per_quarter,
                    track_idx=track.track_idx, out=notes_out,
                )

        time_sec += bar_sec

    notes_out.sort(key=lambda n: n["start"])

    stale = [s for s in human_sync if s.playback_idx >= len(playback_order)]
    if stale:
        warnings.append(
            f"{len(stale)} sync point(s) index past the expanded playback order "
            f"({len(playback_order)} bars); ignored.")
        human_sync = [s for s in human_sync if s.playback_idx < len(playback_order)]

    return Score(
        path=gp_path, title=title, artist=artist, tracks=tracks, bars=bars_out,
        notes=notes_out, human_sync=human_sync, frame_padding=frame_padding,
        audio_name=audio_names[0] if audio_names else None,
        original_audio_path=_text(root.find("Assets/Asset"), "OriginalFilePath") or None,
        warnings=warnings,
    )


def _lay_out_voice(voice_elem, beats_by_id, notes_by_id, rhythms_by_id,
                   start, sec_per_quarter, track_idx, out):
    cursor = start
    open_by_string: dict = {}
    for beat_id in (int(x) for x in (voice_elem.findtext("Beats") or "").split()):
        beat_elem = beats_by_id.get(beat_id)
        if beat_elem is None:
            continue
        rhythm_ref = beat_elem.find("Rhythm")
        rhythm_elem = (rhythms_by_id.get(int(rhythm_ref.get("ref")))
                       if rhythm_ref is not None and rhythm_ref.get("ref") else None)
        dur = _compute_duration_quarters(rhythm_elem) * sec_per_quarter

        notes_elem = beat_elem.find("Notes")
        note_ids = ([int(x) for x in notes_elem.text.split()]
                    if notes_elem is not None and notes_elem.text else [])
        for note_id in note_ids:
            note_elem = notes_by_id.get(note_id)
            if note_elem is None:
                continue
            midi_elem = note_elem.find(".//Property[@name='Midi']/Number")
            if midi_elem is None or not midi_elem.text:
                continue
            midi_num = int(midi_elem.text)

            string_elem = note_elem.find(".//Property[@name='String']/String")
            string_num = (int(string_elem.text)
                          if string_elem is not None and string_elem.text else None)

            tie_elem = note_elem.find("Tie")
            if (tie_elem is not None and tie_elem.get("destination") == "true"
                    and string_num in open_by_string):
                open_by_string[string_num]["end"] = cursor + dur
            else:
                note = {"start": cursor, "end": cursor + dur,
                        "midi": midi_num, "track_idx": track_idx}
                out.append(note)
                open_by_string[string_num] = note
        cursor += dur


def _read_automations(master_track, frame_padding: int = 0
                      ) -> tuple[list[tuple[int, float]], list[SyncPoint]]:
    """Tempo events and sync points, read ONLY from MasterTrack/Automations.

    That restriction is deliberate. Some injector tools append a degenerate
    placeholder sync point (Bar 0, FrameOffset 0, tempo 120) at the XML
    *root* (1283 tabs in one library had one). Guitar Pro does not read
    those -- only points inside MasterTrack/Automations ever appear in the
    app -- and neither do we.
    """
    tempo_events: list[tuple[int, float]] = []
    sync: list[SyncPoint] = []
    automations = master_track.find("Automations")
    if automations is None:
        return tempo_events, sync

    for automation in automations.findall("Automation"):
        atype = automation.findtext("Type")
        if atype == "Tempo":
            try:
                bar_idx = int(automation.findtext("Bar") or "0")
                bpm = float((automation.findtext("Value") or "120").split()[0])
            except (ValueError, IndexError):
                continue
            tempo_events.append((bar_idx, bpm))
        elif atype == "SyncPoint":
            try:
                frame_offset = int(float(automation.findtext("Value/FrameOffset")))
                sync.append(SyncPoint(
                    playback_idx=int(automation.findtext("Bar") or "0"),
                    audio_sec=(frame_offset - frame_padding) / SYNC_FRAME_RATE,
                    frame_offset=frame_offset,
                    modified_tempo=float(automation.findtext("Value/ModifiedTempo")),
                    original_tempo=float(automation.findtext("Value/OriginalTempo") or "0"),
                    bar_occurrence=int(automation.findtext("Value/BarOccurrence") or "0"),
                ))
            except (TypeError, ValueError):
                continue

    tempo_events.sort()
    sync.sort(key=lambda s: s.playback_idx)
    return tempo_events, sync


def extract_audio(gp_path: Path | str, dest_dir: Path | str) -> Path | None:
    """Copy the embedded backing track out of the archive. Returns its path."""
    gp_path, dest_dir = Path(gp_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gp_path) as z:
        assets = [n for n in z.namelist()
                  if n.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac"))]
        if not assets:
            return None
        name = assets[0]
        dest = dest_dir / f"{gp_path.stem}{Path(name).suffix.lower()}"
        with z.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        return dest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a .gp file as a score.")
    ap.add_argument("gp_file")
    args = ap.parse_args()

    s = read_score(args.gp_file)
    print(f"{s.artist} - {s.title}")
    print(f"  playback bars : {len(s.bars)}  ({len(set(b.mb_idx for b in s.bars))} masterbars)")
    print(f"  authored dur  : {s.duration:.1f}s")
    print(f"  notes         : {len(s.notes)} total, {len(s.pitched_notes)} pitched")
    print(f"  embedded audio: {s.audio_name}")
    print(f"  original path : {s.original_audio_path}")
    print(f"  frame padding : {s.frame_padding} samples ({s.pad_sec:+.3f}s)")
    print(f"  human sync    : {len(s.human_sync)} point(s)")
    for t in s.tracks:
        n = sum(1 for x in s.notes if x["track_idx"] == t.track_idx)
        flag = " [PERCUSSION, excluded]" if t.is_percussion else ""
        print(f"    track {t.track_idx}: {t.name!r} ({t.instrument}) {n} notes{flag}")
    for w in s.warnings:
        print(f"  WARNING: {w}")
    if s.human_sync:
        print("  first/last sync points:")
        for sp in (s.human_sync[:3] + (["..."] if len(s.human_sync) > 6 else []) + s.human_sync[-3:]):
            if sp == "...":
                print("      ...")
                continue
            bar = s.bars[sp.playback_idx]
            print(f"      playback bar {sp.playback_idx:>4} (mb {bar.mb_idx:>4}): "
                  f"score {bar.start:7.2f}s -> audio {sp.audio_sec:7.2f}s  "
                  f"(drift {sp.audio_sec - bar.start:+7.2f}s, tempo {sp.modified_tempo:g})")
