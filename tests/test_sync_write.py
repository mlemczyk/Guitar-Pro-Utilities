"""Regression tests for the .gp writer, run with `python -m pytest`.

Every test here corresponds to a bug that actually happened, or to an invariant
whose violation would be silent. That is the point: the two defects this folder
hit were both invisible from the Python side -- the file parsed, the sync points
read back correctly, and Guitar Pro still could not open the audio.

Fixtures are synthetic GPIF text rather than library files, so the suite is
fast, deterministic, and cannot touch anyone's irreplaceable tabs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass

import pytest

from gp_utils.sync.cymbals import score_cymbal_bars
from gp_utils.sync.gp_score import SyncPoint
from gp_utils.sync.write_sync import (GATE_MAX_PATH_COST, GATE_MIN_DIAGONALITY, _restore_gpif,
                        build_sync_automations, gate_check, repad_pad_sec,
                        rewrite_gpif)

# A GPIF with the two things that matter: CDATA everywhere, and a MasterTrack
# whose Automations contains only a Tempo event. Spacing is irregular on
# purpose -- real files vary, and the writer must not normalise them.
GPIF = b"""<?xml version="1.0" encoding="utf-8"?>
<GPIF>
<GPVersion>8.1.5</GPVersion>
<Score>
<Title><![CDATA[Test Song]]></Title>
<SubTitle><![CDATA[]]></SubTitle>
<Artist><![CDATA[Some Artist]]></Artist>
</Score>
  <MasterTrack>
    <Tracks>0 1</Tracks>
    <Automations>
      <Automation>
        <Type>Tempo</Type>
        <Bar>0</Bar>
        <Value>150 2</Value>
      </Automation>
    </Automations>
  </MasterTrack>
<BackingTrack><FramePadding>-4500</FramePadding><AssetId>0</AssetId></BackingTrack>
<Assets><Asset id="0">
<EmbeddedFilePath><![CDATA[Content/Assets/abc-123.mp3]]></EmbeddedFilePath>
</Asset></Assets>
</GPIF>
"""

ROOT_PLACEHOLDER = (b"<Automation><Type>SyncPoint</Type><Linear>false</Linear><Bar>0</Bar>"
                    b"<Position>0</Position><Visible>true</Visible><Value><BarIndex>0</BarIndex>"
                    b"<BarOccurrence>0</BarOccurrence><ModifiedTempo>120</ModifiedTempo>"
                    b"<OriginalTempo>120</OriginalTempo><FrameOffset>0</FrameOffset>"
                    b"</Value></Automation>")

GPIF_WITH_PLACEHOLDER = GPIF.replace(b"<BackingTrack>", ROOT_PLACEHOLDER + b"<BackingTrack>")


@dataclass
class FakeBar:
    playback_idx: int
    mb_idx: int
    start: float
    end: float
    bpm: float = 150.0
    time_sig: tuple = (4, 4)

    @property
    def quarters(self) -> float:
        return self.time_sig[0] * (4.0 / self.time_sig[1])


@dataclass
class FakeTrack:
    track_idx: int
    is_percussion: bool = False
    name: str = "t"
    instrument: str = ""


@dataclass
class FakeScore:
    bars: list
    tracks: list
    notes: list
    frame_padding: int = 0

    @property
    def pad_sec(self) -> float:
        return self.frame_padding / 44100.0


def make_score(n_bars=8, frame_padding=0, mb_map=None, notes=None, tracks=None):
    bars = [FakeBar(i, (mb_map[i] if mb_map else i), i * 1.6, (i + 1) * 1.6)
            for i in range(n_bars)]
    return FakeScore(bars=bars, tracks=tracks or [FakeTrack(0)],
                     notes=notes or [], frame_padding=frame_padding)


# --------------------------------------------------------------------------
# The CDATA bug. Guitar Pro showed the sync points and then reported "failed to
# locate the audio file", because ElementTree dropped every CDATA section on
# re-serialization and the asset path is CDATA.
# --------------------------------------------------------------------------

def test_cdata_sections_survive_a_rewrite():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0), (4, 6.4)])
    out, _ = rewrite_gpif(GPIF, autos, replace_existing=False)
    assert out.count(b"<![CDATA[") == GPIF.count(b"<![CDATA[")
    assert b"<EmbeddedFilePath><![CDATA[Content/Assets/abc-123.mp3]]></EmbeddedFilePath>" in out
    assert b"<SubTitle><![CDATA[]]></SubTitle>" in out, "empty CDATA must not become <SubTitle />"


def test_everything_outside_the_automations_block_is_byte_identical():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0), (4, 6.4)])
    out, _ = rewrite_gpif(GPIF, autos, replace_existing=False)

    def blank(t: bytes) -> bytes:
        i, j = t.find(b"<MasterTrack>"), t.find(b"</MasterTrack>")
        mt = t[i:j]
        a, b = mt.find(b"<Automations>"), mt.find(b"</Automations>")
        return t[:i] + mt[:a] + b"<<AUTO>>" + mt[b:] + t[j:]

    assert blank(out) == blank(GPIF)


def test_xml_declaration_is_preserved():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0)])
    out, _ = rewrite_gpif(GPIF, autos, replace_existing=False)
    assert out.startswith(b'<?xml version="1.0" encoding="utf-8"?>')


# --------------------------------------------------------------------------
# The placement bug, which is why the previous attempt's points never appeared.
# --------------------------------------------------------------------------

def test_automations_land_inside_mastertrack_not_at_the_root():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0), (4, 6.4)])
    out, stats = rewrite_gpif(GPIF, autos, replace_existing=False)

    root = ET.fromstring(out)
    at_root = [a for a in root.findall("Automation") if a.findtext("Type") == "SyncPoint"]
    container = root.find("MasterTrack/Automations")
    in_master = [a for a in container.findall("Automation")
                 if a.findtext("Type") == "SyncPoint"]
    assert at_root == []
    assert len(in_master) == 2
    assert stats["sync_written"] == 2


def test_root_level_placeholder_is_removed():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0)])
    out, stats = rewrite_gpif(GPIF_WITH_PLACEHOLDER, autos, replace_existing=False)
    assert stats["root_placeholders_removed"] == 1
    root = ET.fromstring(out)
    assert [a for a in root.findall("Automation")] == []
    assert out.count(b"<Type>SyncPoint</Type>") == 1


def test_the_tempo_automation_is_not_disturbed():
    score = make_score()
    autos = build_sync_automations(score, [(0, 0.0)])
    out, _ = rewrite_gpif(GPIF, autos, replace_existing=False)
    container = ET.fromstring(out).find("MasterTrack/Automations")
    tempos = [a for a in container.findall("Automation") if a.findtext("Type") == "Tempo"]
    assert len(tempos) == 1
    assert tempos[0].findtext("Value") == "150 2"


def test_replace_existing_removes_only_syncpoints():
    score = make_score()
    first, _ = rewrite_gpif(GPIF, build_sync_automations(score, [(0, 0.0), (4, 6.4)]),
                            replace_existing=False)
    second, stats = rewrite_gpif(first, build_sync_automations(score, [(0, 1.0)]),
                                 replace_existing=True)
    assert stats["existing_sync_removed"] == 2
    assert second.count(b"<Type>SyncPoint</Type>") == 1
    assert second.count(b"<Type>Tempo</Type>") == 1


# --------------------------------------------------------------------------
# FrameOffset arithmetic: the padding conversion and the negative clamp.
# --------------------------------------------------------------------------

def test_frame_offset_applies_frame_padding_when_not_repadding():
    score = make_score(frame_padding=-4500)          # -0.102s
    autos = build_sync_automations(score, [(2, 10.0)], repad=False)
    offset = int(autos[0].findtext("Value/FrameOffset"))
    assert offset == round((10.0 + (-4500 / 44100.0)) * 44100)


def test_repad_puts_the_first_point_one_sample_in():
    """Guitar Pro fills any gap between the audio's start and the first sync
    point by stretching the first segment -- it showed 104.7 BPM on a segment
    whose stored tempo was 111.99. Repadding removes the gap.

    It lands on frame 1 rather than 0 so a real point can never be mistaken for
    GP's default anchor, which is exactly bar 0 / FrameOffset 0."""
    score = make_score(n_bars=12, frame_padding=0)
    autos = build_sync_automations(score, [(0, 0.60), (4, 7.0), (8, 13.4)])
    assert int(autos[0].findtext("Value/FrameOffset")) == 1


def test_repadded_first_point_is_not_read_as_the_default_anchor():
    """The regression this guards: repadding to exactly 0 made Score.real_sync
    discard the genuine first point, because is_default_anchor matches bar 0 with
    FrameOffset 0. A Teen Spirit write came back with 36 points instead of 37."""
    score = make_score(n_bars=12, frame_padding=0)
    autos = build_sync_automations(score, [(0, 0.60), (4, 7.0)])
    first = SyncPoint(playback_idx=int(autos[0].findtext("Bar")),
                      audio_sec=0.0,
                      frame_offset=int(autos[0].findtext("Value/FrameOffset")),
                      modified_tempo=120.0, original_tempo=120.0, bar_occurrence=0)
    assert not first.is_default_anchor


def test_repad_preserves_every_point_in_mp3_time():
    """Repadding must move the frame of reference, not the map. Each point's
    real position, (FrameOffset - FramePadding)/44100, has to be unchanged."""
    score = make_score(n_bars=12, frame_padding=0)
    points = [(0, 0.60), (4, 7.0), (8, 13.4)]
    autos = build_sync_automations(score, points, repad=True)
    new_pad = repad_pad_sec(points) * 44100
    for (_, wanted), a in zip(points, autos):
        offset = int(a.findtext("Value/FrameOffset"))
        assert (offset - new_pad) / 44100 == pytest.approx(wanted, abs=0.001)


def test_negative_frame_offset_is_clamped_to_zero():
    # A bar-0 point a few tens of ms in, with a negative pad, would otherwise
    # produce a negative FrameOffset. No file in the library has one. Tested on
    # the non-repadding path, since repadding chooses the pad itself and so can
    # never produce a negative offset in the first place.
    score = make_score(frame_padding=-44100)          # -1.0s
    autos = build_sync_automations(score, [(0, 0.05)], repad=False)
    assert int(autos[0].findtext("Value/FrameOffset")) == 0


def test_bar_index_is_playback_order_and_occurrence_counts_repeats():
    # Playback visits masterbars 0,1,0,1 -- a repeated two-bar span.
    score = make_score(n_bars=4, mb_map={0: 0, 1: 1, 2: 0, 3: 1})
    autos = build_sync_automations(score, [(0, 0.0), (2, 3.2)])
    assert autos[0].findtext("Bar") == "0"
    assert autos[0].findtext("Value/BarIndex") == "0"
    assert autos[0].findtext("Value/BarOccurrence") == "0"
    # The second point is playback bar 2, which is masterbar 0's SECOND visit.
    assert autos[1].findtext("Bar") == "2"
    assert autos[1].findtext("Value/BarIndex") == "0"
    assert autos[1].findtext("Value/BarOccurrence") == "1"


def test_modified_tempo_reflects_the_measured_spacing():
    # 4 bars of 4/4 = 16 quarters. Over 8 real seconds that is 120 BPM,
    # whatever the authored tempo says.
    score = make_score(n_bars=8)
    autos = build_sync_automations(score, [(0, 0.0), (4, 8.0)])
    assert float(autos[0].findtext("Value/ModifiedTempo")) == pytest.approx(120.0, abs=0.01)


# --------------------------------------------------------------------------
# The gate. Its whole job is to refuse, so the refusal path is what to test.
# --------------------------------------------------------------------------

@dataclass
class FakeAlignment:
    diagonality: float
    path_cost: float


def test_gate_passes_a_good_alignment():
    ok, reasons = gate_check(FakeAlignment(0.99, 0.30))
    assert ok and reasons == []


@pytest.mark.parametrize("diag,cost,n_reasons", [
    (0.50, 0.30, 1),   # stalling path
    (0.99, 0.60, 1),   # poor match
    (0.50, 0.60, 2),   # both
])
def test_gate_refuses_and_says_why(diag, cost, n_reasons):
    ok, reasons = gate_check(FakeAlignment(diag, cost))
    assert not ok
    assert len(reasons) == n_reasons


def test_gate_thresholds_are_the_measured_ones():
    # Guards against a silent loosening of the gate.
    assert GATE_MIN_DIAGONALITY == 0.95
    assert GATE_MAX_PATH_COST == 0.40


# --------------------------------------------------------------------------
# Cymbal landmarks: only a crash ON a downbeat counts.
# --------------------------------------------------------------------------

def test_only_downbeat_cymbals_count_as_landmarks():
    drums = FakeTrack(1, is_percussion=True)
    notes = [
        {"start": 0.0, "midi": 49, "track_idx": 1},    # crash on bar 0's downbeat
        {"start": 0.8, "midi": 49, "track_idx": 1},    # crash mid-bar -- not a landmark
        {"start": 1.6, "midi": 38, "track_idx": 1},    # snare on bar 1's downbeat -- not a cymbal
        {"start": 3.2, "midi": 57, "track_idx": 1},    # crash 2 on bar 2's downbeat
    ]
    score = make_score(n_bars=4, notes=notes, tracks=[FakeTrack(0), drums])
    bars = score_cymbal_bars(score)
    assert set(bars) == {0, 2}


def test_pitched_tracks_are_not_mistaken_for_drums():
    notes = [{"start": 0.0, "midi": 49, "track_idx": 0}]   # a pitched note that happens to be 49
    score = make_score(n_bars=2, notes=notes, tracks=[FakeTrack(0, is_percussion=False)])
    assert score_cymbal_bars(score) == {}


def test_simultaneous_cymbals_are_counted():
    drums = FakeTrack(1, is_percussion=True)
    notes = [{"start": 0.0, "midi": 49, "track_idx": 1},
             {"start": 0.0, "midi": 52, "track_idx": 1}]
    score = make_score(n_bars=2, notes=notes, tracks=[FakeTrack(0), drums])
    assert score_cymbal_bars(score) == {0: 2}


# --------------------------------------------------------------------------
# Undo. This is the safety net under batch_sync.py, so it needs its own test:
# a backup that cannot be restored is worse than no backup, because it is
# trusted.
# --------------------------------------------------------------------------

def _make_archive(path, gpif=GPIF):
    """A minimal .gp-shaped zip, including a STORED directory entry."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("Content/"), b"")
        z.writestr("Content/score.gpif", gpif, zipfile.ZIP_DEFLATED)
        z.writestr("Content/Assets/abc-123.mp3", b"\xff\xfb" + b"fake mp3 payload" * 64,
                   zipfile.ZIP_DEFLATED)
        z.writestr("VERSION", b"8.1")


def _entry_contents(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _compress_types(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: i.compress_type for i in z.infolist()}


def test_restore_gpif_puts_every_entry_back_identically(tmp_path):
    archive = tmp_path / "t.gp"
    _make_archive(archive)
    original, original_types = _entry_contents(archive), _compress_types(archive)

    # Corrupt score.gpif the way a write would change it, then restore.
    modified = GPIF.replace(b"</MasterTrack>", b"</MasterTrack><!--touched-->")
    _restore_gpif(archive, modified)
    assert _entry_contents(archive)["Content/score.gpif"] == modified

    _restore_gpif(archive, original["Content/score.gpif"])
    assert _entry_contents(archive) == original, "every entry must come back byte-identical"
    assert _compress_types(archive) == original_types, "compression must be preserved"


def test_restore_does_not_disturb_the_audio_entry(tmp_path):
    archive = tmp_path / "t.gp"
    _make_archive(archive)
    mp3_before = _entry_contents(archive)["Content/Assets/abc-123.mp3"]
    _restore_gpif(archive, GPIF.replace(b"<GPVersion>", b"<GPVersion>"))
    assert _entry_contents(archive)["Content/Assets/abc-123.mp3"] == mp3_before
