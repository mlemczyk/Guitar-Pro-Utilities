import zipfile
from pathlib import Path

import pytest

from gp_fixture import build_gp, four_quarters
from gp_utils.inject import (InjectError, asset_id, inject_audio, match_audio,
                             show_backing_track, splice_gpif, verify_injection)
from gp_utils.parser import parse_song

AUDIO = b"ID3\x03\x00fake mp3 payload " * 200
PREFS = (b'{"tempo":{"isActive":true},"view":{"globalViewHeight":-1,'
         b'"backingTrackVisible":false,"backingTrackHeight":240}}')


def gp_like_guitar_pro(path: Path) -> Path:
    """A synthetic tab with Guitar Pro's archive layout: folder entries, CDATA,
    Preferences.json, and STORED folders next to DEFLATED files."""
    plain = build_gp(path.with_name("plain.gp"), [four_quarters(28), four_quarters(33)])
    with zipfile.ZipFile(plain) as z:
        gpif = z.read("Content/score.gpif").replace(
            b"<Title>Test Song</Title>", b"<Title><![CDATA[Test Song]]></Title>")
    with zipfile.ZipFile(path, "w") as z:
        for name, data in [("Content/", b""), ("Content/Preferences.json", PREFS),
                           ("Content/Stylesheets/", b""), ("Content/Stylesheets/score.gpss", b"css"),
                           ("Content/score.gpif", gpif), ("VERSION", b"7.0")]:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if name.endswith("/") else zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return path


@pytest.fixture
def tab_and_audio(tmp_path):
    audio = tmp_path / "Artist - Song.mp3"
    audio.write_bytes(AUDIO)
    return gp_like_guitar_pro(tmp_path / "Artist - Song.gp"), audio


def test_injection_keeps_every_other_byte(tab_and_audio, tmp_path):
    gp, audio = tab_and_audio
    out = tmp_path / "out" / "Artist - Song.gp"
    stats = inject_audio(gp, audio, out)
    assert verify_injection(gp, out, AUDIO) == []
    with zipfile.ZipFile(gp) as a, zipfile.ZipFile(out) as b:
        new = b.read("Content/score.gpif")
        assert new.index(b"</MasterTrack>") < new.index(b"<BackingTrack>") < new.index(b"<Tracks><Track ")
        assert new.index(b"</Rhythms>") < new.index(b"<Assets>") < new.index(b"</GPIF>")
        assert b"<Title><![CDATA[Test Song]]></Title>" in new     # CDATA survives
        assert b"SyncPoint" not in new                            # no placeholder
        assert b.read(stats["embedded_path"]) == AUDIO
        assert b'"backingTrackVisible":true' in b.read("Content/Preferences.json")


def test_entry_order_and_compression_follow_guitar_pro(tab_and_audio, tmp_path):
    gp, audio = tab_and_audio
    out = tmp_path / "o.gp"
    stats = inject_audio(gp, audio, out)
    with zipfile.ZipFile(out) as z:
        infos = z.infolist()
    names = [i.filename for i in infos]
    assert names[:3] == ["Content/", "Content/Assets/", stats["embedded_path"]]
    kinds = {i.filename: i.compress_type for i in infos}
    assert kinds["Content/Assets/"] == zipfile.ZIP_STORED
    assert kinds["Content/Stylesheets/"] == zipfile.ZIP_STORED
    assert kinds[stats["embedded_path"]] == zipfile.ZIP_DEFLATED


def test_result_still_parses_and_is_unsynced(tab_and_audio, tmp_path):
    gp, audio = tab_and_audio
    out = tmp_path / "o.gp"
    inject_audio(gp, audio, out)
    before, after = parse_song(gp), parse_song(out)
    assert after == before
    assert after["synced_to_audio"] is False


def test_asset_id_is_deterministic_v5_uuid():
    a = asset_id(AUDIO)
    assert a == asset_id(AUDIO) and a != asset_id(AUDIO + b"x")
    assert a[14] == "5"


def test_original_path_defaults_to_file_name(tab_and_audio, tmp_path):
    gp, audio = tab_and_audio
    out = tmp_path / "o.gp"
    inject_audio(gp, audio, out)
    gpif = zipfile.ZipFile(out).read("Content/score.gpif")
    assert b"<OriginalFilePath><![CDATA[Artist - Song.mp3]]></OriginalFilePath>" in gpif
    assert str(tmp_path).encode() not in gpif


def test_refuses_in_place_existing_output_and_second_injection(tab_and_audio, tmp_path):
    gp, audio = tab_and_audio
    with pytest.raises(InjectError):
        inject_audio(gp, audio, gp)
    out = tmp_path / "o.gp"
    inject_audio(gp, audio, out)
    with pytest.raises(InjectError):
        inject_audio(gp, audio, out)                    # output exists
    with pytest.raises(InjectError):
        inject_audio(out, audio, tmp_path / "o2.gp")    # already has a backing track
    assert not (tmp_path / "o2.gp").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_archive_without_folder_entries_gets_audio_appended(tmp_path):
    gp = build_gp(tmp_path / "t.gp", [four_quarters(28)])
    audio = tmp_path / "a.mp3"
    audio.write_bytes(AUDIO)
    out = tmp_path / "o.gp"
    stats = inject_audio(gp, audio, out)
    names = zipfile.ZipFile(out).namelist()
    assert names[-1] == stats["embedded_path"] and "Content/Assets/" not in names
    assert verify_injection(gp, out, AUDIO) == []


def test_splice_rejects_cdata_terminator():
    with pytest.raises(InjectError):
        splice_gpif(b"<GPIF></GPIF>", asset="x", embedded_path="e", original_path="a]]>b")


def test_show_backing_track_edits_one_token():
    assert show_backing_track(PREFS) == PREFS.replace(b"false", b"true")
    assert b'"backingTrackVisible":true' in show_backing_track(b'{"tempo":{}}')


def test_match_audio(tmp_path):
    (tmp_path / "tabs").mkdir()
    (tmp_path / "music" / "sub").mkdir(parents=True)
    for n in ["The Offspring - Self Esteem", "Nobody - Nothing"]:
        (tmp_path / "tabs" / f"{n}.gp").write_bytes(b"")
    (tmp_path / "music" / "sub" / "Offspring - Self Esteem.mp3").write_bytes(b"")
    (tmp_path / "music" / "Totally Different.mp3").write_bytes(b"")
    got = {gp.stem: (a.name if a else None, r) for gp, a, r in
           match_audio(tmp_path / "tabs", tmp_path / "music")}
    assert got["The Offspring - Self Esteem"] == ("Offspring - Self Esteem.mp3", 1.0)
    assert got["Nobody - Nothing"][0] is None
