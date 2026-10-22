# gp_utils.sync: align a tab to its backing track

Aligns a Guitar Pro 8 tab to the backing track embedded in it and writes
Guitar Pro sync points, so playback follows the real recording instead of the
tab's authored tempo map.

## The problem

A tab and its recording drift apart in two ways: a **linear offset** (the
recording doesn't start where bar 1 does) and **tempo drift** (the band
didn't play to a click). These are the intercept and slope of one monotonic
map from **score time** (the tab played on its own tempo map) to **audio
time**. Guitar Pro stores that map as a `FramePadding` plus per-bar sync
points, so producing it is the task.

## Modules

| module | role |
|---|---|
| `gp_score` | `read_score()`: every pitched track at *authored* timing, the existing sync points, the embedded audio (`extract_audio()`) |
| `audio_io` | mono decoding through ffmpeg, length-checked (soundfile silently truncates some mp3s) |
| `align` | `align()`: chroma DTW of score against audio, then `sync_points_from_alignment()` |
| `cymbals` | downbeat crash detection, used to snap points to transients |
| `fit_points` | sparse anchor fitting (the hand method: align one transient, extrapolate) |
| `grid_align` | refines a map against a rendered beat grid |
| `tempo_lock` | tempo-locked points that remove local wobble |
| `write_sync` | `write_sync_points()`, `gate_check()`, `--undo`: the only writer |
| `verify_roundtrip` | proves a write changed nothing outside the allowed regions |

Install with `pip install -e .[sync]` (numpy, scipy, librosa). ffmpeg and
ffprobe on PATH are strongly recommended (see `audio_io`).

```bash
python -m gp_utils.sync.write_sync "Artist - Title.gp"          # dry run
python -m gp_utils.sync.write_sync "Artist - Title.gp" --apply  # write, with backup
python -m gp_utils.sync.write_sync --undo <batch-label> --apply
```

Backups and the undo journal go to `$GP_UTILS_SYNC_STATE`, else
`./gp_sync_state/`. A caller can also set `write_sync.BACKUP_DIR` and
`write_sync.JOURNAL`.

## The sync-point format: facts that are all load-bearing

Getting any of these wrong produces a confident, plausible, wrong result.

1. **`<Bar>` is an index into the expanded playback order**, not a MasterBar
   index. They coincide only in a song with no repeats.
2. **`<Value><FrameOffset>` is samples at a fixed 44100Hz**, independent of
   the audio's own sample rate.
3. **`<BackingTrack><FramePadding>` is a track-level offset** every sync
   point is relative to. A bar's position *inside the audio file* is
   `(FrameOffset - FramePadding) / 44100`. Its position on Guitar Pro's
   *transport* (when it is played) is `FrameOffset / 44100`. This package
   needs the first; a recording made by playing along to Guitar Pro is in
   the second, which is why `gp_utils.parser` ignores the padding on
   purpose. Neither is a bug.
4. **Automations must be inside `MasterTrack/Automations`.** Guitar Pro
   ignores SyncPoint automations at the XML root, where some tools put a
   placeholder. `write_sync` removes such a placeholder when it writes.
5. **Guitar Pro's own default anchor is not a sync point.** Importing a
   backing track writes one automation at bar 0, `FrameOffset` 0, that no
   human placed. `SyncPoint.is_default_anchor` identifies it and
   `Score.real_sync` filters it out. Counting it as hand-placed work once hid
   100 tabs from a batch run and invented a false "misplaced anchor"
   finding that was really the FramePadding.
6. **A negative `FrameOffset` never occurs** in real files, so the writer
   clamps at zero.

## ⚠ Never re-serialize score.gpif with ElementTree

An early writer parsed the GPIF with `ET.fromstring`, appended automations
and wrote it back with `ET.tostring`. Guitar Pro displayed the sync points
and then refused the audio: **"failed to locate the audio file"**.
ElementTree silently drops CDATA sections, and Guitar Pro's reader requires
them (`<EmbeddedFilePath><![CDATA[...]]></EmbeddedFilePath>`). A GPIF has
about 200 of them.

`rewrite_gpif()` therefore splices **byte spans of the raw text** inside
`<MasterTrack><Automations>` and leaves every other byte alone. The zip gets
the same care: each entry keeps its original order, compression method and
metadata (a blanket `ZIP_DEFLATED` re-compresses the STORED directory
entries). `verify_roundtrip` checks both.

## Method: chroma DTW, not stem separation

Separating stems and pitch-tracking each one solves a harder problem (note
identities) than the one at hand (a monotonic time map), costs hours over a
library, and fails worst on polyphonic guitar. Instead: build a chromagram of
the tab from all pitched tracks at authored timing, build one of the audio,
and dynamic-time-warp them. It runs in 1–4s per song.

Three decisions mattered more than any tuning:

- **`subseq=False`.** Subsequence DTW lets the audio endpoints float, and on
  repetitive music it maps the score onto a fraction of the recording (one
  song's bar 0 landed at 27.9s). Fixing it took that file's worst error from
  52s to 1.0s. Exception: when the audio is over 20% *and* 30s longer than
  the score, the recording holds material the tab doesn't, and `align()`
  switches to subsequence DTW there only.
- **CENS chroma over CQT**, with `win_len_smooth=21` instead of the default
  41 (1.9s of smoothing, too vague to place a bar), then snapping each point
  to the nearest transient within 0.2s, and to a downbeat crash where there
  is one.
- **A stall penalty on non-diagonal steps.** librosa's default step set lets
  the path park on one score frame while the audio runs on, which is how a
  208-second error happens. Penalising it took catastrophic failures from 17
  files to 1.

Measured against 138 hand-synced tabs (error at the human's own points):
median **134ms**, 61 files under 100ms, 103 under 250ms. On files with real
drift it beats the best possible single offset (177ms vs 497ms median), even
though that baseline cheats by fitting to the ground truth.

## The gate is mandatory

`diagonality` and `path_cost` need no ground truth, so the writer gates on
them. The `worst` column decides the rule, not the median:

| rule | kept | median | p90 | worst |
|---|---|---|---|---|
| no gate | 138 (100%) | 134ms | 694ms | 11930ms |
| `diag ≥ 0.95` | 95 (69%) | 105ms | 300ms | 3623ms |
| **`diag ≥ 0.95` and `cost ≤ 0.40`** | **61 (44%)** | **107ms** | **265ms** | **934ms** |

**Refusing is the correct result on a rejected file.** An unsynced tab costs
nothing. A wrong tempo map written into it looks synced and silently
corrupts everything downstream.

Over 1,282 unsynced tabs, 29.7% passed. Tabs that were hard to sync by hand
are hard to sync automatically, so expect less than the 44% above. The main
failure is material that doesn't match, not a weak aligner, and one number
predicts it for free: **no tab whose length differs from its recording by
more than 5% passed.** Those are wrong-recording, partial-transcription or
wrong-version cases that want a human. More notated tracks help (7+ tracks:
49% pass; 2–3: 12%).

## Cautions

- Never write into a library without the gate, and keep the backups.
- Rewriting a `.gp` changes its mtime and size. Anything cached on those, or
  derived from the tempo map (note density, difficulty), will re-measure and
  genuinely move.
- **Structural mismatch is the real failure mode, not precision.** Global DTW
  returns a confident, monotonic, wrong path when the tab and recording
  disagree about repeats or intros. Check the duration gap first.
- Long songs align on a pooled frame grid (`MAX_DTW_FRAMES`): an 831s song
  aligns at 139ms resolution rather than 46ms.
- Hand-placed ground truth can be wrong. Between two sync points the score
  spans a known number of quarter notes, so each pair implies a tempo that
  must match the stored `<ModifiedTempo>`. In one library 9 of 147
  hand-synced tabs failed that check. Passing it is not proof: a uniformly
  shifted set of points is self-consistent and still wrong.
