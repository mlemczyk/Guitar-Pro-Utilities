"""Builds minimal synthetic .gp files for tests. No real tab is ever used.

A bar is a list of beats; a beat is (note_value, midi_or_None), None being a
rest. Every note goes on string 1, fret = midi - 28 (open E1 = 28).
"""

import zipfile
from pathlib import Path


def build_gp(path: Path, bars, *, tempo=120, title="Test Song", artist="Test Artist",
             masterbar_extra=None, sync_points=(), track_type="electricBass",
             track_name="Bass"):
    """Writes a one-track GP8 archive to `path` and returns it.

    `masterbar_extra` maps a bar index to raw XML inserted into that
    MasterBar (e.g. '<Repeat start="true" end="false" count="0"/>').
    `sync_points` is a list of (playback_bar, frame_offset, modified_tempo)."""
    masterbar_extra = masterbar_extra or {}
    rhythms, beats, notes, voices, bar_xml, masterbars = {}, [], [], [], [], []

    for b, bar in enumerate(bars):
        beat_ids = []
        for value, midi in bar:
            rid = rhythms.setdefault(value, len(rhythms))
            note_xml = ""
            if midi is not None:
                nid = len(notes)
                notes.append(
                    f'<Note id="{nid}"><Properties>'
                    f'<Property name="Midi"><Number>{midi}</Number></Property>'
                    f'<Property name="String"><String>1</String></Property>'
                    f'<Property name="Fret"><Fret>{midi - 28}</Fret></Property>'
                    f'</Properties></Note>')
                note_xml = f"<Notes>{nid}</Notes>"
            beat_ids.append(len(beats))
            beats.append(f'<Beat id="{len(beats)}"><Rhythm ref="{rid}"/>{note_xml}</Beat>')
        voices.append(f'<Voice id="{b}"><Beats>{" ".join(map(str, beat_ids))}</Beats></Voice>')
        bar_xml.append(f'<Bar id="{b}"><Voices>{b} -1 -1 -1</Voices></Bar>')
        masterbars.append(f'<MasterBar><Time>4/4</Time><Bars>{b}</Bars>'
                          f'{masterbar_extra.get(b, "")}</MasterBar>')

    automations = [f"<Automation><Type>Tempo</Type><Bar>0</Bar><Value>{tempo} 2</Value></Automation>"]
    for bar_idx, frame_offset, mtempo in sync_points:
        automations.append(
            f"<Automation><Type>SyncPoint</Type><Bar>{bar_idx}</Bar><Value>"
            f"<FrameOffset>{frame_offset}</FrameOffset><ModifiedTempo>{mtempo}</ModifiedTempo>"
            f"</Value></Automation>")
    rhythm_xml = "".join(f'<Rhythm id="{rid}"><NoteValue>{value}</NoteValue></Rhythm>'
                         for value, rid in rhythms.items())

    gpif = (
        "<GPIF>"
        f"<Score><Title>{title}</Title><Artist>{artist}</Artist></Score>"
        f"<MasterTrack><Tracks>0</Tracks><Automations>{''.join(automations)}</Automations></MasterTrack>"
        f'<Tracks><Track id="0"><Name>{track_name}</Name>'
        f"<InstrumentSet><Type>{track_type}</Type></InstrumentSet>"
        '<Properties><Property name="Tuning"><Pitches>28 33 38 43</Pitches></Property></Properties>'
        "</Track></Tracks>"
        f"<MasterBars>{''.join(masterbars)}</MasterBars>"
        f"<Bars>{''.join(bar_xml)}</Bars>"
        f"<Voices>{''.join(voices)}</Voices>"
        f"<Beats>{''.join(beats)}</Beats>"
        f"<Notes>{''.join(notes)}</Notes>"
        f"<Rhythms>{rhythm_xml}</Rhythms>"
        "</GPIF>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Content/score.gpif", gpif)
    return path


def four_quarters(midi):
    return [("Quarter", midi)] * 4
