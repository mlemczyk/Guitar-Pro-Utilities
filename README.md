# Guitar Pro Utilities

Python tools for Guitar Pro 8 (`.gp`) tab files, built around bass tabs:

| module | what it does | status |
|---|---|---|
| `gp_utils.parser` | Reads a `.gp` file and returns its bass track as notes with absolute start/end times, MIDI pitch, string and fret. Expands repeats and alternate endings, and follows audio sync points | **here** |
| [`gp_utils.sync`](gp_utils/sync/README.md) | Aligns a tab to its embedded backing track (chroma DTW, gated) and writes Guitar Pro sync points, with backups and undo | **here** |
| `gp_utils.inject` | Embeds an audio file in a `.gp` as its backing track, writing exactly what Guitar Pro writes; batch mode matches tabs to audio files by name | **here** |
| `gp_utils.midi` | Writes each tab's bass track out as a standalone `.mid` file (standard library only), with repeats and sync points applied | **here** |
| difficulty tagging | Scores how hard a bass tab is to play, from the notes alone | moving in |

The tools are being moved here one at a time from a private project, and
this table shows what has arrived.

## Install

```bash
pip install -e .
```

The parser uses only the Python standard library (3.10+). The sync tools
need numpy, scipy and librosa, plus ffmpeg on PATH:

```bash
pip install -e .[sync]
```

## Use

```python
from pathlib import Path
from gp_utils.parser import parse_song, GpParseError

song = parse_song(Path("Artist - Title.gp"))
song["artist"], song["title"], song["initial_tempo_bpm"]
for n in song["notes"][:4]:
    print(n["start"], n["end"], n["note"], n["string"], n["fret"])
```

`parse_song` returns the title, artist, bass track name, tuning, notes, bars
(in playback order, with repeats unrolled), warnings, and whether the tab is
synced to its backing track. The module docstring of
[`gp_utils/parser.py`](gp_utils/parser.py) documents the GPIF structure it
reads, what it does not handle (D.S./D.C./Coda jumps, nested repeats, second
voices), and the format pitfalls that produce plausible but wrong timings.

## Tests

```bash
pip install -e .[test]
python -m pytest
```

The tests build small `.gp` files in code. No real tabs are included: tabs
are their authors' transcriptions, and this repo does not redistribute them.

## License

Public domain, under the [Unlicense](UNLICENSE).
