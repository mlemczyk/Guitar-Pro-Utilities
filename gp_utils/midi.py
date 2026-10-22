"""Write a Guitar Pro tab's bass track out as a standalone .mid file.

    python -m gp_utils.midi TABS_DIR OUT_DIR
    python -m gp_utils.midi TABS_DIR OUT_DIR --overwrite --limit 20

Each tab becomes a single-track Standard MIDI File (format 0) named after the
tab, written with the standard library only. Notes come from
`gp_utils.parser.parse_song()`, so repeats, alternate endings and audio sync
points are already applied: every note's absolute start and end time is baked
in, and the file's tempo is only the tick<->second conversion constant.

That constant is the song's opening tempo (`initial_tempo_bpm`, from the
Tempo automation at bar 0, or 120 if there is none), not a fixed 120: the
embedded tempo is the only tempo a MIDI reader sees, and anything that groups
notes into bars from it needs a musically meaningful value. Mid-song tempo
changes are not written as tempo events (the note times already include
them), so a reader's bar grid drifts in a song that changes tempo.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

from .parser import GpParseError, discover_gp_files, parse_song

TICKS_PER_BEAT = 480
FALLBACK_BPM = 120.0  # used only if a song has no usable tempo at all


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def write_midi_file(path: Path, notes: list, track_name: str = "Bass",
                    ticks_per_beat: int = TICKS_PER_BEAT, bpm_for_ticks: float = FALLBACK_BPM,
                    velocity: int = 80):
    """Write `notes` ({"start", "end", "midi"} dicts, seconds) to `path`."""
    if bpm_for_ticks is None or bpm_for_ticks <= 0:
        bpm_for_ticks = FALLBACK_BPM
    sec_to_tick = ticks_per_beat * bpm_for_ticks / 60.0

    events = []  # (tick, priority, bytes) -- priority 0 (note-off) sorts before 1 (note-on) at equal ticks
    for n in notes:
        on_tick = round(n["start"] * sec_to_tick)
        off_tick = round(n["end"] * sec_to_tick)
        if off_tick <= on_tick:
            off_tick = on_tick + 1
        pitch = max(0, min(127, int(n["midi"])))
        events.append((on_tick, 1, bytes([0x90, pitch, velocity])))
        events.append((off_tick, 0, bytes([0x80, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))

    track_data = bytearray()
    tempo_us = int(round(60_000_000 / bpm_for_ticks))
    track_data += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + tempo_us.to_bytes(3, "big")

    # Meta-event lengths are variable-length quantities: a single length byte
    # (as an earlier version wrote) corrupts the file for a name of 128+ bytes.
    name_bytes = track_name.encode("utf-8")
    track_data += _vlq(0) + bytes([0xFF, 0x03]) + _vlq(len(name_bytes)) + name_bytes

    last_tick = 0
    for tick, _, data in events:
        track_data += _vlq(tick - last_tick) + data
        last_tick = tick

    track_data += _vlq(0) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    track_chunk = b"MTrk" + struct.pack(">I", len(track_data)) + bytes(track_data)
    Path(path).write_bytes(header + track_chunk)


LOG_FIELDS = ["gp_file", "status", "bass_track_name", "note_count", "warning_count", "error"]


def extract_tab(gp_path: Path, out_path: Path) -> dict:
    """Extract one tab's bass track to `out_path`; returns its log row."""
    row = {"gp_file": gp_path.name, "bass_track_name": "", "note_count": "",
           "warning_count": "", "error": ""}
    try:
        result = parse_song(gp_path)
    except GpParseError as e:
        return {**row, "status": "parse_error", "error": str(e)}
    except Exception as e:  # malformed/unexpected GPIF shape in one file shouldn't kill the batch
        return {**row, "status": "unexpected_error", "error": str(e)}
    notes = result["notes"]
    row.update(bass_track_name=result["bass_track_name"], warning_count=len(result["warnings"]))
    if not notes:
        return {**row, "status": "no_notes", "note_count": 0}
    write_midi_file(out_path, notes, track_name=result["bass_track_name"] or "Bass",
                    bpm_for_ticks=result.get("initial_tempo_bpm"))
    return {**row, "status": "ok", "note_count": len(notes)}


def extract_dir(input_dir: Path, output_dir: Path, overwrite: bool = False,
                limit: int | None = None, log_csv: Path | None = None) -> dict:
    """Extract every *.gp in `input_dir` to `output_dir`/<tab name>.mid.

    Writes a per-file log (default `output_dir`/extraction_log.csv) and
    returns the counts."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(log_csv) if log_csv else output_dir / "extraction_log.csv"

    gp_files = discover_gp_files(input_dir)
    if limit:
        gp_files = gp_files[:limit]
    print(f"Found {len(gp_files)} .gp files in {input_dir}")

    rows, counts = [], {"ok": 0, "skipped_existing": 0, "failed": 0}
    t0, n = time.time(), len(gp_files)
    for i, gp_path in enumerate(gp_files, 1):
        out_path = output_dir / f"{gp_path.stem}.mid"
        if out_path.exists() and not overwrite:
            row = {"gp_file": gp_path.name, "status": "skipped_existing", "bass_track_name": "",
                   "note_count": "", "warning_count": "", "error": ""}
            counts["skipped_existing"] += 1
        else:
            row = extract_tab(gp_path, out_path)
            counts["ok" if row["status"] == "ok" else "failed"] += 1
        rows.append(row)
        if i % 100 == 0 or i == n:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"{datetime.now():%Y-%m-%dT%H:%M:%S}  {i}/{n}  {100 * i / n:5.1f}%  "
                  f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s")

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {counts['ok']} extracted, {counts['skipped_existing']} skipped (already "
          f"extracted), {counts['failed']} failed.")
    print(f"MIDI files written to: {output_dir}")
    print(f"Per-file log: {log_csv}")
    if counts["failed"]:
        print(f"({counts['failed']} failures logged with reasons in the log CSV -- common causes: "
              f"no bass track found, or the file isn't a GP8 zip archive.)")
    return counts


def main(argv=None, default_input: Path | None = None, default_output: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, nargs="?" if default_input else None, default=default_input,
                    help="folder of .gp tabs")
    ap.add_argument("output_dir", type=Path, nargs="?" if default_output else None, default=default_output,
                    help="folder for the .mid files")
    ap.add_argument("--input-dir", dest="input_opt", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--output-dir", dest="output_opt", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--overwrite", action="store_true", help="Re-extract files whose .mid output already exists")
    ap.add_argument("--limit", type=int, help="Only process the first N files (for a quick test run)")
    ap.add_argument("--log-csv", type=Path, default=None,
                    help="Per-file extraction log (default: <output_dir>/extraction_log.csv)")
    args = ap.parse_args(argv)
    try:
        counts = extract_dir(args.input_opt or args.input_dir, args.output_opt or args.output_dir,
                             overwrite=args.overwrite, limit=args.limit, log_csv=args.log_csv)
    except FileNotFoundError as e:
        sys.exit(str(e))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
