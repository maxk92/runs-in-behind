# runs-in-behind

Football/soccer analytics research project extracting and analyzing **deep runs** (Tiefenläufe) — a dynamic vertical movement toward the opponent's goal by a player not in ball possession — from DFL (Deutsche Fußball Liga) Open Data. See `CLAUDE.md` for full pipeline documentation, data-path configuration, and CSV schemas.

## Layout

| Folder | Contents |
|---|---|
| `common/` | Shared config/filter/IoU/timecode/annotation-loading utilities used across the pipeline |
| `segmentation/` | Task 1: discretize position data into candidate movements, filter to run-in-behind candidates |
| `manual_annotations/` | Task 2: inter-annotator agreement (2a), automated-vs-manual evaluation (2b), merge + enrichment (2c) |
| `video-processing/` | Downstream of segmentation, not part of it: cuts video clips for each detected run |
| `pitch-control/` | Independent contribution: a from-scratch pitch-control model, not yet wired into filtering |
| `movement-classification/` | Independent contribution: run-shape classifier (straight/diagonal/curvilinear), not yet wired into filtering |
| `match-context/` | Independent contribution: possession-phase/attack-type/player-role/zone context features, exploratory |
| `clustering-runs/` | Side-quest: clustering analysis of extracted runs |
| `old_code/` | Archived/superseded notebooks and scripts, kept for reference |
| `data/` | Git-ignored working data (raw XML paths, annotations, script outputs) — see `common/config.py`'s `RIB_*` environment variables |

## Running scripts

From the repo root, with `.venv` activated:

```bash
python -m segmentation.extract_runs_behind        # preferred: module invocation
python segmentation/extract_runs_behind.py        # also supported: direct file invocation
```

Both styles work for almost every script here, including the hyphenated `movement-classification/` and `match-context/` folders (Python's `-m` resolves module paths as strings/directory lookups, not via the tokenizer, so a hyphenated folder name works fine as long as the script inside doesn't do a bare same-folder import). The one exception is `pitch-control/pitch_control_examples.py`, which does `from pitch_control_tool import ...` (same folder, unqualified) — that only resolves under direct file invocation (`python pitch-control/pitch_control_examples.py`), since `-m` puts the repo root, not `pitch-control/`, on `sys.path`.
