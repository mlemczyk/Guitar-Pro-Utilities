"""Check that a written .gp file reads back as intended, and how it compares
to the hand-placed sync points of the tab it was derived from.

This is the machine half of the round-trip test. It confirms the file parses,
that the automations landed in MasterTrack/Automations rather than at the XML
root, that FrameOffset survives the FramePadding conversion, and that the
resulting map agrees with the human's. The other half -- does Guitar Pro
itself display and use these points -- can only be checked by opening the file
in the app, which is the whole reason the previous attempt's failure went
unnoticed for so long.
"""

from __future__ import annotations

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path  # noqa: F401  (used throughout)

import numpy as np

from .gp_score import read_score


_ROOT_SYNC_RE = re.compile(
    rb"[ \t]*<Automation>\s*<Type>SyncPoint</Type>.*?</Automation>[ \t]*\r?\n?", re.S)
_PADDING_RE = re.compile(rb"<FramePadding>-?\d+</FramePadding>")


def _blank_automations(text: bytes) -> bytes:
    """Normalise every place a write is ALLOWED to change, so the rest can be
    diffed byte for byte. There are three:

    * **MasterTrack/Automations**, where sync points belong;
    * **any SyncPoint automation at the XML ROOT**, which
      `write_sync.rewrite_gpif` deliberately deletes -- the placeholder some
      injector tools leave (on 1283 tabs in one library), invisible to Guitar
      Pro, and leaving it would mean two bar-0 anchors disagreeing about the
      tempo;
    * **`<BackingTrack><FramePadding>`**, which repadding rewrites so the first
      sync point sits at the head of the audio.

    ⚠ THIS LIST HAS BEEN INCOMPLETE TWICE, AND BOTH TIMES THE CHECK CRIED
    DAMAGE ON CORRECT FILES. First the root-placeholder deletion was missing,
    and two of three spot-checks in the 421-tab batch "failed" on nothing else.
    Then repadding was added and 402 of 421 failed the same way. A verifier that
    flags the specification as a defect is worse than no verifier, because the
    natural response is to distrust the output rather than the test. **When
    `write_sync.py` learns to change something new, teach this function about it
    in the same commit.**
    """
    text = _ROOT_SYNC_RE.sub(b"", text)
    text = _PADDING_RE.sub(b"<FramePadding>0</FramePadding>", text)
    i = text.find(b"<MasterTrack>")
    j = text.find(b"</MasterTrack>", i)
    if i == -1 or j == -1:
        return text
    master = text[i:j]
    a = master.find(b"<Automations>")
    b = master.find(b"</Automations>", a)
    if a == -1 or b == -1:
        return text
    return text[:i] + master[:a] + b"<<AUTOMATIONS>>" + master[b:] + text[j:]


def compare_archives(src_path: Path, new_path: Path) -> dict:
    """Check that writing changed the automations block and nothing else.

    This is a regression guard for a real bug. The first version of
    `write_sync.py` round-tripped the XML through ElementTree, which silently
    dropped every CDATA section; Guitar Pro then showed the sync points but
    reported "failed to locate the audio file", because the asset path is
    CDATA. The mp3 itself was byte-identical and blameless. Counting CDATA
    sections and diffing everything outside the automations block catches that
    class of damage immediately.
    """
    with zipfile.ZipFile(src_path) as z:
        src_entries = z.infolist()
        src_data = {i.filename: z.read(i.filename) for i in src_entries}
    with zipfile.ZipFile(new_path) as z:
        new_entries = z.infolist()
        new_data = {i.filename: z.read(i.filename) for i in new_entries}

    src_gpif = src_data.get("Content/score.gpif", b"")
    new_gpif = new_data.get("Content/score.gpif", b"")
    return {
        "entry_count": (len(src_entries), len(new_entries)),
        "order_preserved": ([i.filename for i in src_entries]
                            == [i.filename for i in new_entries]),
        "compression_preserved": all(a.compress_type == b.compress_type
                                     for a, b in zip(src_entries, new_entries)),
        "changed_entries": [n for n in src_data if src_data[n] != new_data.get(n)],
        "identical_outside_automations":
            _blank_automations(src_gpif) == _blank_automations(new_gpif),
        "cdata_sections": (src_gpif.count(b"<![CDATA["), new_gpif.count(b"<![CDATA[")),
        "audio_identical": all(
            src_data[n] == new_data.get(n)
            for n in src_data if n.lower().endswith((".mp3", ".wav", ".m4a", ".ogg"))),
    }


def xml_placement(gp_path: Path) -> dict:
    """Where do this file's SyncPoint automations physically live?"""
    with zipfile.ZipFile(gp_path) as z:
        raw = z.read("Content/score.gpif")
    root = ET.fromstring(raw)
    at_root = sum(1 for c in root.findall("Automation")
                  if c.findtext("Type") == "SyncPoint")
    master = root.find("MasterTrack")
    in_master = 0
    if master is not None and master.find("Automations") is not None:
        in_master = sum(1 for c in master.find("Automations").findall("Automation")
                        if c.findtext("Type") == "SyncPoint")
    total = raw.decode("utf-8", "ignore").count("<Type>SyncPoint</Type>")
    return {"in_master_automations": in_master, "at_xml_root": at_root,
            "total_in_text": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("written", help="the .gp file that was written")
    ap.add_argument("--original", default=None,
                    help="the hand-synced tab it was derived from, to compare against")
    args = ap.parse_args()

    written = Path(args.written)
    place = xml_placement(written)
    score = read_score(written)

    print(f"WRITTEN: {written.name}")
    print(f"  SyncPoint automations inside MasterTrack/Automations : "
          f"{place['in_master_automations']}")
    print(f"  SyncPoint automations at the XML root (invisible to GP): "
          f"{place['at_xml_root']}")
    print(f"  total occurrences in the raw text                    : "
          f"{place['total_in_text']}")
    print(f"  parsed back by gp_score                              : "
          f"{len(score.human_sync)}")
    print(f"  FramePadding                                         : "
          f"{score.frame_padding} ({score.pad_sec:+.3f}s)")

    ok = (place["in_master_automations"] == len(score.human_sync)
          and place["at_xml_root"] == 0
          and len(score.human_sync) > 0)
    print(f"  placement correct                                    : "
          f"{'YES' if ok else 'NO'}")

    if score.human_sync:
        first, last = score.human_sync[0], score.human_sync[-1]
        print(f"  first point: playback bar {first.playback_idx}, "
              f"audio {first.audio_sec:.2f}s, FrameOffset {first.frame_offset}")
        print(f"  last  point: playback bar {last.playback_idx}, "
              f"audio {last.audio_sec:.2f}s, FrameOffset {last.frame_offset}")
        times = np.array([p.audio_sec for p in score.human_sync])
        print(f"  monotonic in audio time                              : "
              f"{'YES' if np.all(np.diff(times) > 0) else 'NO'}")

    if not args.original:
        return

    arch = compare_archives(Path(args.original), written)
    print("\nARCHIVE INTEGRITY vs the source tab")
    print(f"  entries                          : {arch['entry_count'][0]} -> "
          f"{arch['entry_count'][1]}")
    print(f"  entry order preserved            : "
          f"{'YES' if arch['order_preserved'] else 'NO'}")
    print(f"  compression method preserved     : "
          f"{'YES' if arch['compression_preserved'] else 'NO'}")
    print(f"  entries whose bytes changed      : {arch['changed_entries']}")
    print(f"  embedded audio byte-identical    : "
          f"{'YES' if arch['audio_identical'] else 'NO'}")
    print(f"  CDATA sections (source -> written): {arch['cdata_sections'][0]} -> "
          f"{arch['cdata_sections'][1]}")
    print(f"  score.gpif identical OUTSIDE the automations block: "
          f"{'YES' if arch['identical_outside_automations'] else 'NO'}")

    clean = (arch["order_preserved"] and arch["compression_preserved"]
             and arch["audio_identical"]
             and arch["cdata_sections"][0] == arch["cdata_sections"][1]
             and arch["identical_outside_automations"]
             and arch["changed_entries"] == ["Content/score.gpif"])
    print(f"  VERDICT                          : "
          f"{'clean -- only the automations changed' if clean else 'DAMAGE, do not ship'}")

    original = read_score(args.original)
    if not original.human_sync:
        print("\nOriginal has no hand-placed sync points to compare against.")
        return

    # Interpolate the written map at the human's bars and compare.
    w_bars = np.array([p.playback_idx for p in score.human_sync], dtype=float)
    w_times = np.array([p.audio_sec for p in score.human_sync])
    h_bars = np.array([p.playback_idx for p in original.human_sync], dtype=float)
    h_times = np.array([p.audio_sec for p in original.human_sync])
    pred = np.interp(h_bars, w_bars, w_times)
    err = (pred - h_times) * 1000.0
    bias = float(np.median(err))

    print(f"\nAGAINST THE HUMAN'S {len(h_times)} HAND-PLACED POINTS")
    print(f"  median |error| : {np.median(np.abs(err)):8.0f} ms")
    print(f"  bias           : {bias:+8.0f} ms")
    print(f"  spread         : {np.median(np.abs(err - bias)):8.0f} ms")
    print(f"  90th pct |err| : {np.percentile(np.abs(err), 90):8.0f} ms")
    print(f"  worst          : {np.abs(err).max():8.0f} ms")
    print(f"  points over 1s : {int(np.sum(np.abs(err) > 1000))} of {len(err)}")
    drift = h_times - np.array([original.bars[int(b)].start for b in h_bars])
    print(f"  (the drift being tracked spans {drift.max() - drift.min():.1f}s, "
          f"so 'do nothing' would err by that much)")


if __name__ == "__main__":
    main()
