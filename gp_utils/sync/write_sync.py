"""Write sync points into a .gp archive, where Guitar Pro will actually read them.

THE BUG THIS EXISTS TO NOT REPEAT. An earlier sync writer wrote its sync
points and they never appeared in Guitar Pro. The reason was visible in the
library: 1283 tabs carried a byte-identical placeholder automation --
`Bar 0, FrameOffset 0, ModifiedTempo 120, OriginalTempo 120` -- appended at the
**XML root**, as a sibling of <MasterTrack> rather than a child of
MasterTrack/Automations. Guitar Pro ignores automations placed there. So does
`gp_utils.parser`, which is why it reports only 247 of those tabs as synced
while a naive text search finds 1530.

Everything this module does follows from that:

  * automations go **inside MasterTrack/Automations**, never at the root;
  * the stray root-level placeholder is removed when found, so a file does not
    end up with two bar-0 anchors disagreeing about the tempo;
  * `<Bar>` is written as an index into the **expanded playback order**, not a
    MasterBar index -- writing MasterBar indices silently corrupts any song
    with repeats, which is the bug that once rewound a reference clock by
    70.7 seconds mid-song;
  * `<FrameOffset>` is in samples at a fixed 44100Hz reference, and is relative
    to `<BackingTrack><FramePadding>`, so it is written as
    `(audio_sec + pad_sec) * 44100`.

SAFETY. Tab libraries are irreplaceable hand-collected files, so: dry run is
the default and `--apply` is required; every write is preceded by a full
backup of the original archive (in BACKUP_DIR); every run is journalled (in
JOURNAL) and reversible with `--undo`;
and there is no delete path anywhere in this module. A tab that already has
hand-placed sync points is refused unless `--overwrite` is passed, because
overwriting a human's work is exactly the mistake worth making impossible by
default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

from .gp_score import SYNC_FRAME_RATE, Score, read_score

# Where backups and the undo journal go: $GP_UTILS_SYNC_STATE, else
# ./gp_sync_state under the working directory. A caller may also set
# BACKUP_DIR and JOURNAL on this module directly; they are read at call time.
STATE_DIR = Path(os.environ.get("GP_UTILS_SYNC_STATE") or Path.cwd() / "gp_sync_state")
BACKUP_DIR = STATE_DIR / "write_backups"
JOURNAL = STATE_DIR / "write_journal.jsonl"

# The write gate, measured over the 138 clean ground-truth files on the final
# configuration. Both quantities are computable without ground truth, which is
# what makes them usable on the 1283 files that have none.
#
#   rule                              kept   median     p90    worst
#   no gate                      138 (100%)   134ms   694ms  11930ms
#   diag >= 0.95                  95  (69%)   105ms   300ms   3623ms
#   diag >= 0.95 & cost <= 0.45   78  (57%)   120ms   299ms   3623ms
#   diag >= 0.95 & cost <= 0.40   61  (44%)   107ms   265ms    934ms  <- adopted
#
# The WORST column is what this is chosen on, not the median. Every rule gives a
# good median; only this one keeps the worst accepted file under a second. An
# error of several seconds written into a tab is worse than leaving the tab
# alone, because afterwards it looks synced and nothing downstream will question
# its tempo map.
GATE_MIN_DIAGONALITY = 0.95
GATE_MAX_PATH_COST = 0.40


def gate_check(alignment) -> tuple[bool, list[str]]:
    """Should this alignment be written? Returns (ok, reasons for refusal)."""
    reasons = []
    if alignment.diagonality < GATE_MIN_DIAGONALITY:
        reasons.append(f"diagonality {alignment.diagonality:.3f} < {GATE_MIN_DIAGONALITY}"
                       " (the warping path stalls, which is how large errors arise)")
    if alignment.path_cost > GATE_MAX_PATH_COST:
        reasons.append(f"path cost {alignment.path_cost:.3f} > {GATE_MAX_PATH_COST}"
                       " (the score and the recording do not match well)")
    return (not reasons), reasons


def build_sync_automations(score: Score, points: list[tuple[int, float]],
                           repad: bool = True) -> list[ET.Element]:
    """Turn [(playback_idx, audio_sec), ...] into GP <Automation> elements.

    ⚠ `repad` MOVES THE LEAD-IN INTO FramePadding, AND WITHOUT IT GUITAR PRO
    DISTORTS THE FIRST SEGMENT. Guitar Pro anchors the audio's sample 0 at
    transport time `FramePadding`, so if the first sync point sits at a non-zero
    FrameOffset, GP has a gap between the start of the audio and that point and
    stretches the first segment to cover it.

    Caught on "Nirvana - Smells Like Teen Spirit", where the author's screenshot
    showed **104.7 BPM** on the first sync point while the file stores 111.99.
    The arithmetic gives it away: 16 quarters over 9.17s -- the span from audio
    *zero* to the second point -- is 104.7, whereas 16 quarters over the 8.57s
    between the two actual points is 112.0. GP was filling from 0, not from the
    first point at 0.60s. Every later point in that file displayed exactly its
    stored tempo, which is why only the first segment looked wrong.

    Writing `FramePadding = -first_audio_time` puts the first point at
    FrameOffset 0 and leaves nothing to fill. The map is unchanged in mp3-position
    terms -- `(FrameOffset - FramePadding)/44100` is identical for every point --
    so this only changes how GP reads the head of the file.
    """
    pad_sec = score.pad_sec
    if repad and points:
        pad_sec = repad_pad_sec(points)
    occurrence_of: dict[int, int] = {}
    seen: dict[int, int] = {}
    for bar in score.bars:
        seen[bar.mb_idx] = seen.get(bar.mb_idx, -1) + 1
        occurrence_of[bar.playback_idx] = seen[bar.mb_idx]

    points = sorted(points, key=lambda p: p[0])
    out: list[ET.Element] = []
    for n, (playback_idx, audio_sec) in enumerate(points):
        bar = score.bars[playback_idx]

        # ModifiedTempo is the real performed tempo from this point to the next.
        if n + 1 < len(points):
            nxt_idx, nxt_sec = points[n + 1]
            quarters = sum(score.bars[i].quarters for i in range(playback_idx, nxt_idx))
            elapsed = nxt_sec - audio_sec
            modified = (quarters * 60.0 / elapsed) if elapsed > 0 and quarters > 0 else bar.bpm
        else:
            modified = bar.bpm

        automation = ET.Element("Automation")
        ET.SubElement(automation, "Type").text = "SyncPoint"
        ET.SubElement(automation, "Linear").text = "false"
        ET.SubElement(automation, "Bar").text = str(playback_idx)
        ET.SubElement(automation, "Position").text = "0"
        ET.SubElement(automation, "Visible").text = "true"
        value = ET.SubElement(automation, "Value")
        ET.SubElement(value, "BarIndex").text = str(bar.mb_idx)
        ET.SubElement(value, "BarOccurrence").text = str(occurrence_of[playback_idx])
        ET.SubElement(value, "ModifiedTempo").text = f"{modified:.4f}".rstrip("0").rstrip(".")
        ET.SubElement(value, "OriginalTempo").text = f"{bar.bpm:g}"
        # Clamped at zero: with a negative FramePadding, a bar-0 point that the
        # aligner puts a few tens of ms into the audio converts to a NEGATIVE
        # FrameOffset (measured: -1656 samples on the Opeth round trip). No
        # observed file in the library has one, so it is not a value Guitar Pro
        # is known to accept, and the truncation it costs is bounded by the pad
        # itself -- 37ms in that case, well under the 106ms median error.
        frame_offset = int(round((audio_sec + pad_sec) * SYNC_FRAME_RATE))
        ET.SubElement(value, "FrameOffset").text = str(max(0, frame_offset))
        out.append(automation)
    return out


SYNC_AUTOMATION_RE = re.compile(
    rb"[ \t]*<Automation>\s*<Type>SyncPoint</Type>.*?</Automation>[ \t]*\r?\n?",
    re.S)


def _span(text: bytes, open_tag: bytes, close_tag: bytes,
          start: int = 0, end: int | None = None):
    """Byte span of an element, or None. Assumes the element is not nested."""
    end = len(text) if end is None else end
    i = text.find(open_tag, start, end)
    if i == -1:
        return None
    j = text.find(close_tag, i, end)
    if j == -1:
        return None
    return i, j + len(close_tag)


FRAME_PADDING_RE = re.compile(rb"(<FramePadding>)(-?\d+)(</FramePadding>)")

# The first sync point is placed one sample in, not at zero.
#
# ⚠ AT EXACTLY ZERO IT BECOMES INDISTINGUISHABLE FROM GUITAR PRO'S DEFAULT
# ANCHOR. `SyncPoint.is_default_anchor` identifies that anchor as bar 0 with
# FrameOffset 0 -- the point GP writes on import, which 1283 library tabs carry
# and which nobody placed. Repadding so a real first point landed on exactly
# those coordinates made `Score.real_sync` discard it: a Teen Spirit write came
# back with 36 points instead of 37, its genuine bar-0 anchor silently dropped.
#
# One sample is 0.023ms, which is inaudible and leaves GP essentially no gap to
# fill, while keeping the two cases distinct in the data.
FIRST_POINT_FRAMES = 1


def repad_pad_sec(points: list[tuple[int, float]]) -> float:
    """The FramePadding, in seconds, that puts the first point one sample in."""
    return FIRST_POINT_FRAMES / SYNC_FRAME_RATE - min(t for _, t in points)


def _set_frame_padding(xml_bytes: bytes, frames: int) -> tuple[bytes, bool]:
    """Rewrite <BackingTrack><FramePadding> in place, as text.

    ⚠ THIS MUST HAPPEN WHENEVER `build_sync_automations` CHANGES THE PAD.
    Every FrameOffset is stored relative to FramePadding, so writing offsets
    computed against a new pad while leaving the old value in the XML shifts the
    entire map by the difference -- silently, and by up to the pad's own size.
    """
    new, n = FRAME_PADDING_RE.subn(
        lambda m: m.group(1) + str(int(frames)).encode() + m.group(3),
        xml_bytes, count=1)
    return new, n > 0


def rewrite_gpif(xml_bytes: bytes, automations: list[ET.Element],
                 replace_existing: bool, frame_padding: int | None = None
                 ) -> tuple[bytes, dict]:
    """Return new score.gpif bytes with `automations` installed correctly.

    ⚠ THIS EDITS THE XML AS TEXT, AND MUST KEEP DOING SO. The obvious
    implementation -- ElementTree parse, modify the tree, `ET.tostring` -- was
    tried first and produced files whose sync points Guitar Pro displayed
    correctly but which failed to open their audio with "failed to locate the
    audio file". The mp3 was byte-identical and not at fault. **ElementTree
    silently discards CDATA sections**, so

        <EmbeddedFilePath><![CDATA[Content/Assets/<uuid>.mp3]]></EmbeddedFilePath>

    came back as

        <EmbeddedFilePath>Content/Assets/<uuid>.mp3</EmbeddedFilePath>

    which is equivalent XML by the standard and is not what Guitar Pro's reader
    accepts. It also rewrote `<SubTitle><![CDATA[]]></SubTitle>` as
    `<SubTitle />` and changed the XML declaration's quoting. Every title,
    artist and asset path in a GPIF is CDATA, so a round trip through ET
    rewrites most of the file's text content.

    Editing byte spans instead means everything outside the automations block
    is preserved exactly, which is the only way to be sure nothing else moved.
    """
    stats = {"root_placeholders_removed": 0, "existing_sync_removed": 0,
             "frame_padding_written": None}

    if frame_padding is not None:
        xml_bytes, ok = _set_frame_padding(xml_bytes, frame_padding)
        if not ok:
            raise ValueError("no <FramePadding> element to update; refusing to "
                             "write offsets computed against a pad the file "
                             "does not declare")
        stats["frame_padding_written"] = int(frame_padding)

    master = _span(xml_bytes, b"<MasterTrack>", b"</MasterTrack>")
    if master is None:
        raise ValueError("no MasterTrack element to write into")
    m_start, m_end = master
    head, master_text, tail = (xml_bytes[:m_start],
                               xml_bytes[m_start:m_end],
                               xml_bytes[m_end:])

    # 1. Drop stray root-level SyncPoint automations -- the injector's
    #    placeholder. Guitar Pro ignores them, but leaving them behind means a
    #    file with two bar-0 anchors disagreeing about the tempo.
    head, n1 = SYNC_AUTOMATION_RE.subn(b"", head)
    tail, n2 = SYNC_AUTOMATION_RE.subn(b"", tail)
    stats["root_placeholders_removed"] = n1 + n2

    # 2. Find (or create) MasterTrack/Automations and edit inside it.
    container = _span(master_text, b"<Automations>", b"</Automations>")
    if container is None:
        insert_at = master_text.rfind(b"</MasterTrack>")
        master_text = (master_text[:insert_at]
                       + b"<Automations>\n</Automations>\n"
                       + master_text[insert_at:])
        container = _span(master_text, b"<Automations>", b"</Automations>")

    c_start, c_end = container
    inner = master_text[c_start + len(b"<Automations>"):c_end - len(b"</Automations>")]
    if replace_existing:
        inner, n3 = SYNC_AUTOMATION_RE.subn(b"", inner)
        stats["existing_sync_removed"] = n3

    block = b"".join(ET.tostring(a, encoding="utf-8") + b"\n" for a in automations)
    if inner and not inner.endswith(b"\n"):
        inner += b"\n"
    new_master = (master_text[:c_start] + b"<Automations>\n" + inner + block
                  + b"</Automations>" + master_text[c_end:])

    stats["sync_written"] = len(automations)
    return head + new_master + tail, stats


def write_sync_points(gp_path: Path, points: list[tuple[int, float]],
                      apply: bool = False, overwrite: bool = False,
                      out_path: Path | None = None,
                      light_backup: bool = False,
                      batch_label: str | None = None) -> dict:
    """Install `points` into `gp_path`, or into `out_path` if given.

    `out_path` is how a round-trip test is run without risking the library
    original: the source tab is read but never written, and the result lands
    somewhere the library scanners do not walk.
    """
    gp_path = Path(gp_path)
    score = read_score(gp_path)

    if score.real_sync and not overwrite and out_path is None:
        raise RuntimeError(
            f"{gp_path.name} already has {len(score.real_sync)} hand-placed sync "
            f"point(s). Refusing to overwrite without --overwrite.")

    # The pad and the offsets are one decision: FrameOffset is stored relative to
    # FramePadding, so whatever `build_sync_automations` used must also be
    # written into the file. Computed once here and passed to both.
    new_pad = int(round(repad_pad_sec(points) * SYNC_FRAME_RATE)) if points \
        else score.frame_padding
    automations = build_sync_automations(score, points, repad=True)

    with zipfile.ZipFile(gp_path) as z:
        entries = z.infolist()
        original = {i.filename: z.read(i.filename) for i in entries}
    new_gpif, stats = rewrite_gpif(original["Content/score.gpif"], automations,
                                   replace_existing=overwrite or out_path is not None,
                                   frame_padding=new_pad)

    target = Path(out_path) if out_path else gp_path
    stats.update({
        "source": gp_path.name,
        "target": str(target),
        "in_place": out_path is None,
        "pad_sec": score.pad_sec,
        "playback_bars": len(score.bars),
        "applied": bool(apply),
    })
    if not apply:
        stats["note"] = "DRY RUN -- nothing written. Pass --apply to write."
        return stats

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if out_path is None:
        # Writing in place, so the original must be preserved first.
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if light_backup:
            # Only Content/score.gpif is ever modified, so saving just that
            # entry is a complete undo record and a much smaller one. How much
            # smaller varies with how big the tab is next to its mp3 -- 7.6% of
            # the archive on "Rise Against - Savior", 47.8% on the much shorter
            # "Linkin Park - One Step Closer" -- so budget on the order of a few
            # GB for a full-library batch rather than the ~13GB a whole-archive
            # backup would take. Verified by the archive-integrity check, which
            # asserts no other entry's bytes ever change.
            backup = BACKUP_DIR / f"{stamp}__{gp_path.name}.score.gpif"
            backup.write_bytes(original["Content/score.gpif"])
            stats["backup_kind"] = "gpif"
        else:
            backup = BACKUP_DIR / f"{stamp}__{gp_path.name}"
            shutil.copy2(gp_path, backup)   # copy, never move: original stays put
            stats["backup_kind"] = "archive"
        stats["backup"] = str(backup)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        stats["backup"] = "(not needed: source untouched)"
        stats["backup_kind"] = "none"

    # Rebuild the archive preserving each entry's ORIGINAL order, compression
    # method and metadata. Writing everything with a blanket ZIP_DEFLATED also
    # re-compressed the STORED directory entries, and while that is legal zip,
    # the point of this rewrite is that nothing except score.gpif changes.
    tmp = target.with_suffix(".gp.tmp")
    with zipfile.ZipFile(tmp, "w") as out:
        for info in entries:
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.internal_attr = info.internal_attr
            new_info.create_system = info.create_system
            payload = new_gpif if info.filename == "Content/score.gpif" else original[info.filename]
            out.writestr(new_info, payload)
    tmp.replace(target)

    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": stamp, "batch": batch_label, **stats}) + "\n")
    return stats


def _restore_gpif(archive: Path, gpif_bytes: bytes) -> None:
    """Put a saved Content/score.gpif back, leaving every other entry alone."""
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        data = {i.filename: z.read(i.filename) for i in entries}
    tmp = archive.with_suffix(".gp.undo")
    with zipfile.ZipFile(tmp, "w") as out:
        for info in entries:
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.internal_attr = info.internal_attr
            new_info.create_system = info.create_system
            payload = gpif_bytes if info.filename == "Content/score.gpif" else data[info.filename]
            out.writestr(new_info, payload)
    tmp.replace(archive)


def undo_last(apply: bool = False, batch_label: str | None = None) -> list[dict]:
    """Restore every target in the most recent journalled batch.

    Handles both backup kinds: a whole-archive copy, and the lighter
    score.gpif-only record used for batch runs. The light form is a complete
    undo because no other entry in the archive is ever modified.

    ⚠ AFTER A LIGHT UNDO THE FILE'S CONTENT IS RESTORED BUT ITS HASH IS NOT.
    Verified end to end: every zip entry comes back byte-identical with its
    original compression method, and the archive's own sha256 still differs,
    because rebuilding a zip writes fresh local headers. Guitar Pro reads the
    entries and does not care. Two things follow: **do not verify an undo by
    hashing the .gp** -- compare entry contents, as `verify_roundtrip.py` does
    -- and note that an undone tab still looks "changed" to any cache keyed on
    mtime plus size, so it will be re-measured once more. Harmless, but not
    free.

    A whole-archive backup (`--apply` on a single file, the default) restores
    the original bytes exactly and has neither caveat.
    """
    if not JOURNAL.exists():
        return []
    entries = [json.loads(line) for line in
               JOURNAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [e for e in entries if e.get("in_place")]
    if not entries:
        return []

    if batch_label:
        batch = [e for e in entries if e.get("batch") == batch_label]
    else:
        last_ts = entries[-1]["ts"]
        batch = [e for e in entries if e["ts"] == last_ts]

    for entry in batch:
        backup = Path(entry["backup"])
        entry["will_restore"] = backup.exists() and Path(entry["target"]).exists()
        if not (apply and entry["will_restore"]):
            continue
        if entry.get("backup_kind") == "gpif":
            _restore_gpif(Path(entry["target"]), backup.read_bytes())
        else:
            shutil.copy2(backup, entry["target"])
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gp_file", nargs="?", help="the .gp file to write into")
    ap.add_argument("--every-n-bars", type=int, default=4,
                    help="place a sync point every N bars (default 4)")
    ap.add_argument("--apply", action="store_true", help="actually write (default is a dry run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing hand-placed sync points IN PLACE")
    ap.add_argument("--out", default=None,
                    help="write to this path instead of in place; the source tab is "
                         "then never modified (use this for round-trip tests)")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="write even if the alignment fails the quality gate")
    ap.add_argument("--undo", action="store_true", help="restore the most recent batch")
    args = ap.parse_args()

    if args.undo:
        batch = undo_last(apply=args.apply)
        if not batch:
            print("Nothing to undo.")
            return
        print(f"{'Restoring' if args.apply else 'Would restore'} {len(batch)} file(s):")
        for e in batch:
            mark = "ok" if e["will_restore"] else "BACKUP MISSING"
            print(f"  [{mark}] {Path(e['target']).name}")
        if not args.apply:
            print("\nDry run. Pass --apply to actually restore.")
        return

    if not args.gp_file:
        ap.error("gp_file is required unless --undo is given")

    import shutil as _shutil
    import tempfile
    from .align import align, sync_points_from_alignment
    from .gp_score import extract_audio

    gp_path = Path(args.gp_file)
    score = read_score(gp_path)
    tmpdir = Path(tempfile.mkdtemp(prefix="gpsync_"))
    try:
        audio = extract_audio(gp_path, tmpdir)
        alignment = align(score, audio, verbose=True)
        points = sync_points_from_alignment(score, alignment, args.every_n_bars)
    finally:
        _shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n  {alignment.summary}")
    print(f"  {len(points)} sync points every {args.every_n_bars} bars")

    ok, reasons = gate_check(alignment)
    print(f"  gate: {'PASS' if ok else 'FAIL'} "
          f"(diag {alignment.diagonality:.3f} >= {GATE_MIN_DIAGONALITY}, "
          f"cost {alignment.path_cost:.3f} <= {GATE_MAX_PATH_COST})")
    for reason in reasons:
        print(f"    - {reason}")
    if not ok and not args.ignore_gate:
        print("\n  REFUSED: this alignment is not good enough to write.")
        print("  Leaving the tab unsynced is the safe outcome -- a wrong tempo map")
        print("  is worse than none, because afterwards the tab looks synced.")
        print("  Pass --ignore-gate to override, or --out to write a copy and listen.")
        raise SystemExit(3)

    try:
        stats = write_sync_points(gp_path, points, apply=args.apply,
                                  overwrite=args.overwrite,
                                  out_path=Path(args.out) if args.out else None)
    except RuntimeError as e:
        print(f"\n  REFUSED: {e}")
        print("  Use --out <path> to write a copy instead, which never touches "
              "the source.")
        raise SystemExit(2)
    print()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


