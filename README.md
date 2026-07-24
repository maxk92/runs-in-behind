# runs-in-behind

Football/soccer analytics research project extracting and analyzing **deep runs** (Tiefenläufe) — a dynamic vertical movement toward the opponent's goal by a player not in ball possession — from DFL (Deutsche Fußball Liga) Open Data. See `CLAUDE.md` for full pipeline documentation, data-path configuration, and CSV schemas.

## Layout

| Folder | Contents |
|---|---|
| `common/` | Shared config/filter/IoU/timecode/annotation-loading utilities used across the pipeline |
| `segmentation/` | Task 1: discretize position data into candidate movements, filter to run-in-behind candidates, generate video QA clips |
| `manual_annotations/` | Task 2: inter-annotator agreement (2a), automated-vs-manual evaluation (2b), merge + enrichment (2c) |
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

Both styles work for every script under `segmentation/` and `manual_annotations/`; scripts under the hyphenated `pitch-control/`, `movement-classification/`, and `match-context/` folders support direct file invocation only (hyphens aren't valid Python module-path segments).
