# Poster (DML course, A2) — draft

`poster.tex` on the course template (`beamerposter.sty`, `beamerthemeZH.sty`, `img/`: leave the style files
as they are). It compares four ways of training a Fox, with one colour each in every figure:

| | Agent | Model |
|---|---|---|
| 1 | Pure RL (red) | `super_smash/models/alejandro/MK5.zip` |
| 2 | RL + 200 ms reaction delay (purple) | `super_smash/models/alejandro/MK9.zip` |
| 3 | Imitation learning (blue) | `super_smash/models/kepiti/fox_imitation_18h.pt` |
| 4 | Imitation + RL (orange) | `super_smash/models/kepiti/fox_rl_3h.pt` |

Colours 3 and 4 are the ones of Kepiti's poster ([Kepitition/super_smash_RL](https://github.com/Kepitition/super_smash_RL), `poster/`).

## Build

```bash
latexmk -pdf poster.tex        # Linux (TeX Live); or upload this folder to Overleaf and compile with pdfLaTeX
```

`img/chalmers_logo.pdf` is the template's EPS logo converted with Ghostscript, so pdfLaTeX doesn't need
`epstopdf`. Only English is loaded in `babel` (the template's German option is not needed).
Check that the PDF is 420 x 594 mm (A2) and one page.

## Still to fill in

- **Training curves** (dashed box in "Training (preliminary)") and the numbers in its tiles, once the final
  runs end.
- **Final tournament**: the win-rate table (`?` cells, games per pair) and the dashed figure box.
- **Conclusion**, after the tournament.
- Names: `\author{...}` in `poster.tex`. E-mail in the footer: `\href{mailto:...}` in `beamerthemeZH.sty`
  (the one content line in the theme).

## Figures

| File | From |
|---|---|
| `fig/hero.jpg` | `make_figures.py`: three frames of MK5 against the level 9 CPU (lasers, rush, knockout) |
| `fig/agent_*.jpg` | `make_figures.py`: one frame per agent; agents 1-3 from our captures, agent 4 cropped from `fig/kepiti_hero.jpg` |
| `fig/kepiti_hero.jpg`, `fig/kepiti_combo_strip.jpg` | Kepiti's poster (his imitation + RL model against the level 9 CPU) |

Screenshots: `capture_frames.py` (run from `super_smash/` with its `.venv`) plays the bots in a windowed
Dolphin at 3x resolution and saves the window whenever someone takes a big hit, into `frames/<session>/`
(not in git) with `captures.jsonl` (frame, reason, damage and action of each player). `make_figures.py`
lists which frame and crop each figure uses.

```bash
cd ../super_smash && source .venv/bin/activate
python ../Poster/capture_frames.py --out ../Poster/frames/mk5 --p1 ours:models/alejandro/MK5.zip --p2 cpu:FOX:9
python ../Poster/capture_frames.py --out ../Poster/frames/ffa --seconds 150 \
    --p1 ours:models/alejandro/MK5.zip --p2 ours:models/alejandro/MK9.zip \
    --p3 kepiti:models/kepiti/fox_imitation_18h.pt --p4 kepiti:models/kepiti/fox_rl_3h.pt
cd ../Poster && python make_figures.py && latexmk -pdf poster.tex
```

It opens a Dolphin window for the length of the session (`--seconds` of play, 75 by default).
