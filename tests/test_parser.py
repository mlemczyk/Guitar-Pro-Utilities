import zipfile
from pathlib import Path

import pytest

from gp_fixture import build_gp, four_quarters
from gp_utils.parser import (GpParseError, parse_gp_artist_title, parse_song,
                             strip_leading_tags)

REPEAT_START = '<Repeat start="true" end="false" count="0"/>'
REPEAT_END = '<Repeat start="false" end="true" count="2"/>'


def starts(song):
    return [n["start"] for n in song["notes"]]


def test_plain_bars_at_120bpm(tmp_path):
    song = parse_song(build_gp(tmp_path / "t.gp", [four_quarters(28), four_quarters(33)]))
    assert song["title"] == "Test Song" and song["artist"] == "Test Artist"
    assert song["tuning"] == (28, 33, 38, 43)
    assert starts(song) == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert song["notes"][0]["note"] == "E1"
    assert song["notes"][4]["fret"] == 5 and song["notes"][4]["string"] == 1
    assert [b["index"] for b in song["bars"]] == [0, 1]
    assert song["synced_to_audio"] is False


def test_rests_advance_time(tmp_path):
    bar = [("Quarter", 28), ("Quarter", None), ("Half", 31)]
    song = parse_song(build_gp(tmp_path / "t.gp", [bar]))
    assert starts(song) == [0.0, 1.0]
    assert song["notes"][1]["end"] == 2.0


def test_repeat_span_plays_count_times(tmp_path):
    bars = [four_quarters(28), four_quarters(33), four_quarters(38)]
    # bar 1 is both the start and end of a one-bar repeat: 0, 1, 1, 2
    extra = {1: '<Repeat start="true" end="true" count="2"/>'}
    song = parse_song(build_gp(tmp_path / "t.gp", bars, masterbar_extra=extra))
    assert [b["index"] for b in song["bars"]] == [0, 1, 1, 2]
    assert len(song["notes"]) == 16


def test_alternate_endings_play_once_each(tmp_path):
    # bar 0 body, bar 1 = 1st ending (repeat end), bar 2 = 2nd ending, bar 3 after.
    bars = [four_quarters(28), four_quarters(33), four_quarters(38), four_quarters(43)]
    extra = {0: REPEAT_START,
             1: REPEAT_END + "<AlternateEndings>1</AlternateEndings>",
             2: "<AlternateEndings>2</AlternateEndings>"}
    song = parse_song(build_gp(tmp_path / "t.gp", bars, masterbar_extra=extra))
    assert [b["index"] for b in song["bars"]] == [0, 1, 0, 2, 3]


def test_multi_bar_first_ending_marked_on_its_first_bar_only(tmp_path):
    # Primo Victoria's layout: bar 0 body, bars 1-2 = 1st ending with the mark
    # on bar 1 only (bar 2 is the unmarked repeat end), bar 3 = 2nd ending.
    bars = [four_quarters(28), four_quarters(33), four_quarters(38),
            four_quarters(43), four_quarters(28)]
    extra = {0: REPEAT_START,
             1: "<AlternateEndings>1</AlternateEndings>",
             2: REPEAT_END,
             3: "<AlternateEndings>2</AlternateEndings>"}
    song = parse_song(build_gp(tmp_path / "t.gp", bars, masterbar_extra=extra))
    assert [b["index"] for b in song["bars"]] == [0, 1, 2, 0, 3, 4]


def test_decorative_first_ending_is_a_plain_repeat(tmp_path):
    bars = [four_quarters(28), four_quarters(33), four_quarters(38)]
    extra = {0: REPEAT_START, 1: REPEAT_END + "<AlternateEndings>1</AlternateEndings>"}
    song = parse_song(build_gp(tmp_path / "t.gp", bars, masterbar_extra=extra))
    assert [b["index"] for b in song["bars"]] == [0, 1, 0, 1, 2]


def test_sync_point_bar_is_a_playback_index(tmp_path):
    # Bars 0-1 repeat twice, then bar 2: playback order 0, 1, 0, 1, 2.
    # A sync point on playback bar 3 (MasterBar 1, second pass) at 10s must
    # land on the second pass, not rewind the first one.
    bars = [four_quarters(28), four_quarters(33), four_quarters(38)]
    extra = {0: REPEAT_START, 1: REPEAT_END}
    song = parse_song(build_gp(tmp_path / "t.gp", bars, masterbar_extra=extra,
                               sync_points=[(3, 10 * 44100, 120)]))
    assert song["synced_to_audio"] is True and song["sync_point_count"] == 1
    assert [round(b["start"], 4) for b in song["bars"]] == [0.0, 2.0, 4.0, 10.0, 12.0]


def test_synth_bass_is_used_when_it_is_the_only_bass(tmp_path):
    song = parse_song(build_gp(tmp_path / "t.gp", [four_quarters(28)], track_type="synthBass",
                               track_name="Keys"))
    assert song["bass_track_name"] == "Keys"


def test_no_bass_track(tmp_path):
    with pytest.raises(GpParseError):
        parse_song(build_gp(tmp_path / "t.gp", [four_quarters(28)], track_type="electricGuitar",
                            track_name="Lead"))


def test_not_a_gp_file(tmp_path):
    bad = tmp_path / "bad.gp"
    bad.write_text("not a zip")
    with pytest.raises(GpParseError):
        parse_song(bad)
    empty = tmp_path / "empty.gp"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("other.txt", "")
    with pytest.raises(GpParseError):
        parse_song(empty)


@pytest.mark.parametrize("stem,expected", [
    ("[Hard][Warm] Blink-182 - Dick Lips", ("Blink-182", "Dick Lips")),
    ("Red Hot Chili Peppers - Snow (Hey Oh)", ("Red Hot Chili Peppers", "Snow (Hey Oh)")),
    ("No Separator", ("No Separator", "")),
])
def test_artist_title(stem, expected):
    assert parse_gp_artist_title(Path(stem + ".gp")) == expected


def test_strip_leading_tags():
    assert strip_leading_tags("[Easy][Dark] A - B") == "A - B"
    assert strip_leading_tags("A [live] - B") == "A [live] - B"


def _score(tracks):
    """A bare GPIF root for find_bass_track: `tracks` is [(name, type,
    notes)], each track getting one bar of `notes` single-note beats."""
    import xml.etree.ElementTree as ET
    track_xml, bars, voices, beats, bar_ids = [], [], [], [], []
    for tid, (name, itype, n) in enumerate(tracks):
        track_xml.append(f'<Track id="{tid}"><Name>{name}</Name>'
                         f'<InstrumentSet><Type>{itype}</Type></InstrumentSet></Track>')
        beat_ids = []
        for _ in range(n):
            beat_ids.append(str(len(beats)))
            beats.append(f'<Beat id="{len(beats)}"><Notes>0</Notes></Beat>')
        voices.append(f'<Voice id="{tid}"><Beats>{" ".join(beat_ids)}</Beats></Voice>')
        bars.append(f'<Bar id="{tid}"><Voices>{tid} -1 -1 -1</Voices></Bar>')
        bar_ids.append(str(tid))
    order = " ".join(str(t) for t in range(len(tracks)))
    return ET.fromstring(
        f'<GPIF><MasterTrack><Tracks>{order}</Tracks></MasterTrack>'
        f'<Tracks>{"".join(track_xml)}</Tracks>'
        f'<MasterBars><MasterBar><Bars>{" ".join(bar_ids)}</Bars></MasterBar></MasterBars>'
        f'<Bars>{"".join(bars)}</Bars><Voices>{"".join(voices)}</Voices>'
        f'<Beats>{"".join(beats)}</Beats></GPIF>')


def _picked(tracks):
    from gp_utils.parser import find_bass_track
    _, elem = find_bass_track(_score(tracks))
    return elem.findtext("Name")


def test_bassoon_is_not_a_bass():
    assert _picked([("Vocals", "bassoon", 50), ("Flea", "electricBass", 40)]) == "Flea"


def test_side_part_names_lose_to_the_main_part():
    assert _picked([("Bass FX", "electricBass", 5), ("Justin", "electricBass", 40)]) == "Justin"
    assert _picked([("Guitar 2", "electricBass", 5), ("Bass", "electricBass", 40)]) == "Bass"
    assert _picked([("Bass Guitar", "electricBass", 9), ("Rhythm II", "electricBass", 40)]) == "Bass Guitar"


def test_note_count_overrides_order_only_past_double():
    assert _picked([("Riedel (Drop C)", "electricBass", 5),
                    ("Riedel (Drop D)", "electricBass", 40)]) == "Riedel (Drop D)"
    assert _picked([("Bass", "electricBass", 30),
                    ("Acoustic Bass", "acousticBass", 40)]) == "Bass"
