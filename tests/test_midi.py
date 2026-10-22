import struct

from gp_fixture import build_gp, four_quarters
from gp_utils.midi import extract_dir, extract_tab, write_midi_file


def read_track(path):
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    _, fmt, ntracks, tpb = struct.unpack(">IHHH", data[4:14])
    assert (fmt, ntracks) == (0, 1)
    assert data[14:18] == b"MTrk"
    (length,) = struct.unpack(">I", data[18:22])
    track = data[22:22 + length]
    assert len(track) == length and track.endswith(b"\x00\xff\x2f\x00")
    return tpb, track


def test_tempo_comes_from_the_tab(tmp_path):
    gp = build_gp(tmp_path / "t.gp", [four_quarters(28)], tempo=150)
    row = extract_tab(gp, tmp_path / "t.mid")
    assert row["status"] == "ok" and row["note_count"] == 4
    tpb, track = read_track(tmp_path / "t.mid")
    assert tpb == 480
    assert track[1:4] == b"\xff\x51\x03"                  # tempo meta-event first
    assert int.from_bytes(track[4:7], "big") == 400000    # 60e6 / 150 BPM, in microseconds
    assert track.count(b"\x90\x1c\x50") == 4   # four note-ons, E1, velocity 80


def test_note_ticks(tmp_path):
    out = tmp_path / "n.mid"
    write_midi_file(out, [{"start": 0.0, "end": 0.5, "midi": 40}, {"start": 0.5, "end": 0.5, "midi": 41}],
                    bpm_for_ticks=120)
    _, track = read_track(out)
    # at 120 BPM and 480 ticks/beat, 0.5s = 480 ticks; a zero-length note lasts 1 tick
    assert b"\x00\x90\x28\x50" in track and b"\x83\x60\x80\x28\x00" in track
    assert b"\x00\x90\x29\x50\x01\x80\x29\x00" in track


def test_long_track_name_length_is_a_vlq(tmp_path):
    out = tmp_path / "long.mid"
    name = "x" * 200
    write_midi_file(out, [{"start": 0, "end": 1, "midi": 40}], track_name=name)
    _, track = read_track(out)
    i = track.index(b"\xff\x03")
    assert track[i + 2:i + 4] == b"\x81\x48"           # 200 as a VLQ
    assert track[i + 4:i + 204] == name.encode()


def test_extract_dir_skips_existing_and_logs(tmp_path):
    tabs = tmp_path / "tabs"
    tabs.mkdir()
    build_gp(tabs / "A - One.gp", [four_quarters(28)])
    build_gp(tabs / "B - Guitar only.gp", [four_quarters(40)], track_type="electricGuitar", track_name="Lead")
    out = tmp_path / "mid"
    assert extract_dir(tabs, out) == {"ok": 1, "skipped_existing": 0, "failed": 1}
    assert (out / "A - One.mid").exists() and (out / "extraction_log.csv").exists()
    assert extract_dir(tabs, out) == {"ok": 0, "skipped_existing": 1, "failed": 1}
