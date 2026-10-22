# Guitar Pro Utilities

Python tools for Guitar Pro 8 (`.gp`) tab files, built around bass tabs:

| module | what it does |
|---|---|
| [`gp_utils.parser`](gp_utils/parser.py) | Reads a `.gp` file and returns its bass track as notes with absolute start/end times, MIDI pitch, string and fret. Expands repeats and alternate endings, and follows audio sync points |
| [`gp_utils.sync`](gp_utils/sync/README.md) | Aligns a tab to its embedded backing track (chroma DTW, gated) and writes Guitar Pro sync points, with backups and undo |
| [`gp_utils.inject`](gp_utils/inject.py) | Embeds an audio file in a `.gp` as its backing track, writing exactly what Guitar Pro writes; batch mode matches tabs to audio files by name |
| [`gp_utils.midi`](gp_utils/midi.py) | Writes each tab's bass track out as a standalone `.mid` file (standard library only), with repeats and sync points applied |
| [`gp_utils.difficulty`](gp_utils/difficulty/metrics.py) | Measures how hard a bass part is: notes per minute over the bars that play, burst density, fret travel, string crossings and shifts (after re-fingering onto a 5-string), and a predicted note accuracy from a model fitted to one player's recorded takes |

## Install

```bash
pip install -e .            # parser, injector, MIDI, difficulty: standard library only
pip install -e .[sync]      # adds numpy, scipy and librosa for gp_utils.sync
```

Python 3.10+. The sync tools also want ffmpeg and ffprobe on PATH.

## Use

Read a tab:

```python
from pathlib import Path
from gp_utils.parser import parse_song

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

Measure difficulty:

```python
from gp_utils.difficulty import measure

row = measure(Path("Artist - Title.gp"))
row["predicted_accuracy_pct"], row["notes_per_minute"], row["shifts_per_min"]
```

From the command line:

```bash
# embed a backing track (writes a new file; the source is never modified)
python -m gp_utils.inject "Artist - Title.gp" "Artist - Title.mp3" "out/Artist - Title.gp"
python -m gp_utils.inject --batch TABS_DIR AUDIO_DIR OUT_DIR          # dry run; add --apply

# align a tab to its backing track and write sync points (dry run; add --apply)
python -m gp_utils.sync.write_sync "Artist - Title.gp"

# write every tab's bass track as a .mid file
python -m gp_utils.midi TABS_DIR OUT_DIR
```

## About the difficulty prediction

`predicted_accuracy_pct` comes from a per-note logistic model whose fitted
constants are in [`gp_utils/difficulty/weights.json`](gp_utils/difficulty/weights.json),
with the date and data of the fit. It was calibrated on 82 recorded takes of
42 songs by one fingerstyle player on a 5-string bass, scored by a
note-matching scorer. Use it to rank tabs by difficulty; its absolute
percentages are that player's. The file is updated when the model is refit,
and its `measure_version` changes with it, so anything caching the output
should key on `gp_utils.difficulty.MEASURE_VERSION`.

## Tests

```bash
pip install -e .[sync,test]
python -m pytest
```

The tests build small `.gp` files in code. No real tabs are included: tabs
are their authors' transcriptions, and this repo does not redistribute them.

## License

Public domain, under the [Unlicense](UNLICENSE).
