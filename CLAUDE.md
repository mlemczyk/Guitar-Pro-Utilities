# Guitar Pro Utilities: rules for AI agents

Public repo (Unlicense) of tools for Guitar Pro 8 `.gp` files. It is the
canonical home of this code: a private project imports it from here, so a
change here changes that project's results.

## Hard rules

- **Never commit a tab or audio.** No `.gp`, `.gpif`, mp3, wav or stems,
  not even as a test fixture. Tests build their `.gp` files in code
  (`tests/gp_fixture.py`). `.gitignore` blocks the extensions as a backstop.
  Song and artist names may appear in docs and comments.
- **No hard-coded local paths.** Paths are arguments. Defaults that point
  into someone's library belong in the caller, not here.
- **Behaviour changes are deliberate.** Moving or tidying code must leave its
  output identical. Prove it by comparing output before and after on real
  tabs (kept locally, outside the repo), not just by passing the tests.
- **Never modify `score.gpif` through ElementTree** (`ET.parse` → edit →
  `ET.write`). Reading it with ElementTree is fine. Writing through it
  rewrites the whole document (attribute order, quoting, whitespace, CDATA)
  and Guitar Pro is sensitive to that. Writers edit the raw text.
- Keep the parser standard-library only. Heavier tools declare optional
  extras in `pyproject.toml`.

## Before changing the sync tools

Read `gp_utils/sync/README.md`. The write gate (`diag ≥ 0.95` and
`cost ≤ 0.40`) is mandatory and is chosen on the worst accepted error, not
the median: never loosen it to sync more tabs. `write_sync` is the only
writer: dry run by default, a backup before every write, a journal for
`--undo`, and no delete path. Keep it that way.

## Before changing the injector

`gp_utils/inject.py` reproduces Guitar Pro's own import: `<BackingTrack>`
after `</MasterTrack>`, `<Assets>` before `<ScoreViews>`, one element per
line, CDATA on the names and paths. Checked by stripping the audio from 272
tabs Guitar Pro had written and injecting it back: identical apart from the
asset id, the file name, FramePadding and the user's own mixer/view settings.
Keep `verify_injection()` running on every write, write no SyncPoint, and
never write over the source.

## Before changing the parser

Read the module docstring of `gp_utils/parser.py`, in particular PITFALLS.
Each entry there (repeats, alternate endings as `<AlternateEndings>`, the
sync-point `<Bar>` being a playback index, FrameOffset vs FramePadding) is a
mistake that once shifted every later note while still looking right.
