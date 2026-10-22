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

## Before changing the parser

Read the module docstring of `gp_utils/parser.py`, in particular PITFALLS.
Each entry there (repeats, alternate endings as `<AlternateEndings>`, the
sync-point `<Bar>` being a playback index, FrameOffset vs FramePadding) is a
mistake that once shifted every later note while still looking right.
