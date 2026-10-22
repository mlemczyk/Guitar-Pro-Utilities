"""Embed an audio file in a Guitar Pro 8 (.gp) tab as its backing track.

    python -m gp_utils.inject "Artist - Title.gp" "Artist - Title.mp3" out.gp
    python -m gp_utils.inject --batch TABS_DIR AUDIO_DIR OUT_DIR           # dry run
    python -m gp_utils.inject --batch TABS_DIR AUDIO_DIR OUT_DIR --apply

What Guitar Pro needs, and nothing else:

  * a `<BackingTrack>` element directly after `</MasterTrack>` and an
    `<Assets>` element directly before `<ScoreViews>` in `score.gpif`, which
    is where Guitar Pro itself puts them, in its own one-element-per-line
    format with CDATA around the names and paths;
  * the audio stored in the archive at `Content/Assets/<uuid>.<ext>`;
  * `"backingTrackVisible": true` in `Content/Preferences.json`, so the
    backing-track lane shows when the file opens.

RULES THIS MODULE KEEPS (an earlier injector broke all three):

  * **score.gpif is spliced as raw text, never re-serialized.** ElementTree
    drops every CDATA section and re-indents text content; Guitar Pro needs
    the CDATA (without it, it fails to locate its own audio). Every byte of the
    original GPIF survives, and the result is checked for that.
  * **No placeholder sync point.** A bar-0 SyncPoint automation written at the
    XML root is ignored by Guitar Pro and only misleads tools that count sync
    points. The tab comes out unsynced; `gp_utils.sync` aligns it.
  * **Zip entries are preserved.** Each existing entry keeps its order,
    compression method and metadata; only score.gpif and Preferences.json
    change, and the audio entry is added.

SAFETY: the source tab is only read. Output goes to a new file (writing over
the source is refused), through a temporary file that is verified against the
source before it replaces anything. A tab that already has a backing track is
refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
import zipfile
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

GPIF = "Content/score.gpif"
PREFS = "Content/Preferences.json"
ASSETS_DIR = "Content/Assets/"

# Guitar Pro's own <BackingTrack> and <Assets>, as written by GP 8.1 on
# import. Only the asset id, the file name and FramePadding vary between
# files; FramePadding starts at 0 and gp_utils.sync sets it.
_BACKING_TRACK = """<BackingTrack>
<IconId>21</IconId>
<Color>0 0 0</Color>
<Name><![CDATA[{name}]]></Name>
<ShortName><![CDATA[{short_name}]]></ShortName>
<PlaybackState>Solo</PlaybackState>
<ChannelStrip>
<Parameters>0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.500000 0.000000 0.500000 0.500000 0.800000 0.500000 0.500000 0.500000</Parameters>
</ChannelStrip>
<Enabled>true</Enabled>
<Source>Local</Source>
<AssetId>0</AssetId>
<YouTubeVideoUrl></YouTubeVideoUrl>
<Filter>6</Filter>
<FramesPerPixel>300</FramesPerPixel>
<FramePadding>0</FramePadding>
<Semitones>0</Semitones>
<Cents>0</Cents>
</BackingTrack>
"""
_ASSETS = """<Assets>
<Asset id="0">
<OriginalFilePath><![CDATA[{original_path}]]></OriginalFilePath>
<OriginalFileSha1><![CDATA[{asset_id}]]></OriginalFileSha1>
<EmbeddedFilePath><![CDATA[{embedded_path}]]></EmbeddedFilePath>
</Asset>
</Assets>
"""
_CDATA_ADDED = 5   # Name, ShortName, OriginalFilePath, OriginalFileSha1, EmbeddedFilePath


class InjectError(Exception):
    pass


def asset_id(audio: bytes) -> str:
    """A version-5-shaped UUID from the SHA-1 of the audio bytes.

    Guitar Pro's own asset ids are version-5 UUIDs and it repeats the id in
    `<OriginalFileSha1>`. This one is deterministic: the same audio always
    gets the same id."""
    return str(uuid.UUID(bytes=hashlib.sha1(audio).digest()[:16], version=5))


def _insertion_points(gpif: bytes) -> tuple[int, int]:
    """(after </MasterTrack> and its line break, before <ScoreViews>).

    Without a <ScoreViews> element the assets go before </GPIF>."""
    mt = gpif.find(b"</MasterTrack>")
    if mt == -1:
        raise InjectError("score.gpif has no </MasterTrack>")
    bt = mt + len(b"</MasterTrack>")
    bt += 2 if gpif[bt:bt + 2] == b"\r\n" else 1 if gpif[bt:bt + 1] == b"\n" else 0
    sv = gpif.find(b"<ScoreViews>", bt)
    if sv == -1:
        sv = gpif.rfind(b"</GPIF>")
    if sv == -1:
        raise InjectError("score.gpif has no closing </GPIF>")
    return bt, sv


def _blocks(gpif: bytes, **fields) -> tuple[bytes, bytes]:
    nl = "\r\n" if b"\r\n" in gpif[:400] else "\n"
    bt = _BACKING_TRACK.format(**fields).replace("\n", nl).encode("utf-8")
    assets = _ASSETS.format(**fields).replace("\n", nl).encode("utf-8")
    return bt, assets


def splice_gpif(gpif: bytes, *, asset: str, embedded_path: str, original_path: str,
                name: str = "Audio Track", short_name: str = "a.track") -> bytes:
    """`gpif` with <BackingTrack> after </MasterTrack> and <Assets> before
    <ScoreViews>, matching Guitar Pro.

    Raw-text splice: every original byte is kept, in order."""
    if b"<BackingTrack>" in gpif or b"<Assets>" in gpif:
        raise InjectError("the tab already has a backing track (<BackingTrack>/<Assets> present)")
    for value in (name, short_name, original_path):
        if "]]>" in value:
            raise InjectError(f"cannot store {value!r} in a CDATA section")
    at_bt, at_assets = _insertion_points(gpif)
    bt, assets = _blocks(gpif, name=name, short_name=short_name, original_path=original_path,
                         asset_id=asset, embedded_path=embedded_path)
    return gpif[:at_bt] + bt + gpif[at_bt:at_assets] + assets + gpif[at_assets:]


def show_backing_track(prefs: bytes) -> bytes:
    """Preferences.json with view.backingTrackVisible set to true.

    A one-token edit when the key is there (Guitar Pro writes it), so the rest
    of the file is untouched; otherwise a compact re-dump."""
    new, n = re.subn(rb'("backingTrackVisible"\s*:\s*)false', rb"\1true", prefs, count=1)
    if n or re.search(rb'"backingTrackVisible"\s*:\s*true', prefs):
        return new
    data = json.loads(prefs.decode("utf-8"))
    data.setdefault("view", {})["backingTrackVisible"] = True
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _new_entries(infos, embedded_path: str):
    """Where the audio (and, archive style permitting, its folder) go.

    Archives written by Guitar Pro carry folder entries ('Content/',
    'Content/Stylesheets/', ...) and list 'Content/Assets/' and the audio
    right after 'Content/'. An archive without folder entries just gets the
    audio appended."""
    names = [i.filename for i in infos]
    has_dirs = any(n.endswith("/") for n in names)
    if has_dirs and "Content/" in names:
        at = names.index("Content/") + 1
        add = ([] if ASSETS_DIR in names else [ASSETS_DIR]) + [embedded_path]
        return at, add
    return len(names), [embedded_path]


def inject_audio(gp_path, audio_path, out_path, *, original_path: str | None = None,
                 name: str = "Audio Track") -> dict:
    """Write a copy of `gp_path` with `audio_path` embedded as its backing track.

    `original_path` is what goes in `<OriginalFilePath>`; it defaults to the
    audio's file name only, so no local directory ends up in the tab.
    Returns a stats dict. Raises InjectError on any refusal or failed check."""
    gp_path, audio_path, out_path = Path(gp_path), Path(audio_path), Path(out_path)
    if out_path.resolve() == gp_path.resolve():
        raise InjectError("refusing to write over the source tab; give a different output path")
    if out_path.exists():
        raise InjectError(f"{out_path} already exists")

    audio = audio_path.read_bytes()
    asset = asset_id(audio)
    ext = (audio_path.suffix or ".mp3").lower()
    embedded_path = f"{ASSETS_DIR}{asset}{ext}"

    with zipfile.ZipFile(gp_path) as z:
        infos = z.infolist()
        original = {i.filename: z.read(i.filename) for i in infos}
    if GPIF not in original:
        raise InjectError("no Content/score.gpif in the archive")
    if any(n.startswith(ASSETS_DIR) and not n.endswith("/") for n in original):
        raise InjectError("the archive already holds an embedded asset")

    new_gpif = splice_gpif(original[GPIF], asset=asset, embedded_path=embedded_path,
                           original_path=original_path or audio_path.name, name=name)
    new_prefs = show_backing_track(original[PREFS]) if PREFS in original else None

    at, additions = _new_entries(infos, embedded_path)
    now = datetime.now().timetuple()[:6]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as out:
            def copy(info):
                new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                new_info.internal_attr = info.internal_attr
                new_info.create_system = info.create_system
                payload = (new_gpif if info.filename == GPIF
                           else new_prefs if info.filename == PREFS and new_prefs is not None
                           else original[info.filename])
                out.writestr(new_info, payload)

            # New entries borrow their attributes from the archive's own:
            # a folder entry for Content/Assets/, score.gpif for the audio.
            by_name = {i.filename: i for i in infos}
            dir_template = by_name.get("Content/")
            file_template = by_name[GPIF]

            for info in infos[:at]:
                copy(info)
            for entry in additions:
                is_dir = entry.endswith("/")
                template = dir_template if is_dir else file_template
                new_info = zipfile.ZipInfo(entry, date_time=now)
                new_info.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
                new_info.external_attr = template.external_attr
                new_info.create_system = template.create_system
                out.writestr(new_info, b"" if is_dir else audio)
            for info in infos[at:]:
                copy(info)

        problems = verify_injection(gp_path, tmp, audio)
        if problems:
            raise InjectError("verification failed: " + "; ".join(problems))
        tmp.replace(out_path)
    finally:
        if tmp.exists():
            tmp.unlink()   # our own temporary file, never the source or a result

    return {"source": gp_path.name, "target": str(out_path), "audio": audio_path.name,
            "asset_id": asset, "embedded_path": embedded_path, "audio_bytes": len(audio),
            "added_entries": additions}


def verify_injection(src, dst, audio: bytes) -> list[str]:
    """Problems with `dst` as an injection of `audio` into `src`; [] if none.

    Checks that every source entry survives byte-identical with its compression
    method (score.gpif and Preferences.json excepted), that score.gpif is the
    source plus exactly one inserted block containing the five new CDATA
    sections and no SyncPoint, and that the embedded audio matches."""
    problems = []
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst) as b:
        if b.testzip() is not None:
            problems.append("output archive fails its CRC check")
        b_infos = {i.filename: i for i in b.infolist()}
        for info in a.infolist():
            other = b_infos.get(info.filename)
            if other is None:
                problems.append(f"entry lost: {info.filename}")
                continue
            if other.compress_type != info.compress_type:
                problems.append(f"compression changed: {info.filename}")
            if info.filename not in (GPIF, PREFS) and a.read(info.filename) != b.read(info.filename):
                problems.append(f"entry changed: {info.filename}")
        old, new = a.read(GPIF), b.read(GPIF)
        spans = [re.search(rb"<BackingTrack>.*?</BackingTrack>\r?\n", new, re.S),
                 re.search(rb"<Assets>.*?</Assets>\r?\n", new, re.S)]
        if None in spans:
            problems.append("score.gpif lacks the inserted <BackingTrack>/<Assets>")
        elif new[:spans[0].start()] + new[spans[0].end():spans[1].start()] + new[spans[1].end():] != old:
            problems.append("score.gpif changed outside the inserted blocks")
        else:
            block = spans[0].group(0) + spans[1].group(0)
            if new.count(b"<![CDATA[") - old.count(b"<![CDATA[") != _CDATA_ADDED:
                problems.append("unexpected CDATA count in score.gpif")
            if b"SyncPoint" in block:
                problems.append("inserted block contains a SyncPoint")
            m = re.search(rb"<EmbeddedFilePath><!\[CDATA\[(.*?)\]\]>", block)
            if not m:
                problems.append("no EmbeddedFilePath in the inserted block")
            elif m.group(1).decode() not in b_infos:
                problems.append("EmbeddedFilePath names an entry that is not in the archive")
            elif b.read(m.group(1).decode()) != audio:
                problems.append("embedded audio differs from the source audio")
        if PREFS in b_infos and not re.search(rb'"backingTrackVisible"\s*:\s*true', b.read(PREFS)):
            problems.append("Preferences.json does not show the backing track")
    return problems


# --- batch: match tabs to audio files by name ---------------------------------

def normalize_filename(filename: str) -> str:
    """Lower-case, with every standalone word "the" removed."""
    return " ".join(w for w in filename.lower().split() if w != "the").strip()


def match_audio(gp_dir, audio_dir, threshold: float = 0.75,
                exts=(".mp3",)) -> list[tuple[Path, Path | None, float]]:
    """(tab, audio or None, similarity) for every *.gp in `gp_dir`.

    Audio is searched recursively under `audio_dir`. An exact match on the
    normalized names wins; otherwise the most similar name at or above
    `threshold` (difflib ratio)."""
    audio = [(normalize_filename(p.stem), p) for p in sorted(Path(audio_dir).rglob("*"))
             if p.suffix.lower() in exts]
    out = []
    for gp in sorted(Path(gp_dir).glob("*.gp")):
        key = normalize_filename(gp.stem)
        exact = next((p for k, p in audio if k == key), None)
        if exact is not None:
            out.append((gp, exact, 1.0))
            continue
        best, best_ratio = None, 0.0
        for k, p in audio:
            r = SequenceMatcher(None, key, k).ratio()
            if r >= threshold and r > best_ratio:
                best, best_ratio = p, r
        out.append((gp, best, best_ratio))
    return out


def batch_inject(gp_dir, audio_dir, out_dir, apply: bool = False, threshold: float = 0.75) -> dict:
    """Inject matched audio into copies of every tab in `gp_dir`, under `out_dir`.

    Dry run unless `apply`. Tabs that already have a backing track, or have
    no match, are skipped. Sources are never modified."""
    matches = match_audio(gp_dir, audio_dir, threshold)
    counts = {"written": 0, "would_write": 0, "skipped": 0, "failed": 0}
    t0, n = time.time(), len(matches)
    for i, (gp, audio, ratio) in enumerate(matches, 1):
        elapsed = time.time() - t0
        eta = elapsed / (i - 1) * (n - i + 1) if i > 1 else 0.0
        prefix = (f"{datetime.now():%Y-%m-%dT%H:%M:%S} [{i}/{n}] {100 * i / n:5.1f}% "
                  f"eta {eta / 60:4.1f}min")
        if audio is None:
            counts["skipped"] += 1
            print(f"{prefix}  skip (no audio match)  {gp.name}")
            continue
        label = "exact" if ratio == 1.0 else f"{ratio:.2f}"
        if not apply:
            counts["would_write"] += 1
            print(f"{prefix}  would inject ({label}) {audio.name}  ->  {gp.name}")
            continue
        try:
            inject_audio(gp, audio, Path(out_dir) / gp.name)
            counts["written"] += 1
            print(f"{prefix}  injected ({label}) {audio.name}  ->  {gp.name}")
        except InjectError as e:
            counts["skipped"] += 1
            print(f"{prefix}  skip ({e})  {gp.name}")
        except Exception as e:  # keep going; report at the end
            counts["failed"] += 1
            print(f"{prefix}  ERROR {type(e).__name__}: {e}  {gp.name}")
    print(f"\n{counts}" + ("" if apply else "\nDry run: nothing written. Pass --apply."))
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs=3, metavar=("TAB_OR_DIR", "AUDIO_OR_DIR", "OUT"),
                    help="tab, audio and output file; with --batch, three folders")
    ap.add_argument("--batch", action="store_true", help="match and inject a folder of tabs")
    ap.add_argument("--apply", action="store_true", help="with --batch: actually write")
    ap.add_argument("--threshold", type=float, default=0.75, help="with --batch: fuzzy-match cutoff")
    args = ap.parse_args(argv)
    if args.batch:
        counts = batch_inject(*args.paths, apply=args.apply, threshold=args.threshold)
        return 1 if counts["failed"] else 0
    try:
        stats = inject_audio(*args.paths)
    except InjectError as e:
        print(f"Refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
