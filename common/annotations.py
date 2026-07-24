"""
Shared access to the manual-annotation CSVs written by rib-annotation-app
(see /home/max/drive/projects/0_applications/rib-annotation-app/data_manager.py),
following the canonical schema:

    segment_id, half, start_frame, end_frame, start_time_s, end_time_s,
    duration_s, player, player_jid, team, outcome, annotation_source

Filenames follow `DFL-MAT-{match_id}_{half}_{annotator}.csv` once cleaned up
(see data/annotations/); the app's native output additionally has a
timestamp suffix (`{match_id}_{half}_{annotator}_{timestamp}.csv`) which
`discover_match_annotation_files` also matches via glob.

`load_annotations` deliberately does NOT normalize `player_jid` NaNs --
Extraction_Evaluation.py drops unparsable-jid rows while
Stats_Manual_Annotation_App.py keeps them with a sentinel value, and each
caller keeps applying its own existing policy after loading.
"""

import glob
import os

import pandas as pd


def load_annotations(path: str) -> pd.DataFrame:
    """Read one manual-annotation CSV as-is."""
    return pd.read_csv(path)


def annotation_file_path(annotation_dir: str, match_id: str, half: str,
                          annotator: str) -> str:
    return os.path.join(annotation_dir, f"DFL-MAT-{match_id}_{half}_{annotator}.csv")


def discover_annotators(annotation_dir: str) -> list:
    """
    Return sorted distinct annotator names found in `annotation_dir`, parsed
    from `DFL-MAT-{match_id}_{half}_{annotator}.csv` filenames.
    """
    annotators = set()
    if not os.path.exists(annotation_dir):
        return []
    for f in os.listdir(annotation_dir):
        if f.startswith("DFL-MAT-") and f.endswith(".csv"):
            parts = f.replace(".csv", "").split("_")
            annotators.add(parts[-1])
    return sorted(annotators)


def discover_match_annotation_files(annotation_dir: str, match_id: str) -> dict:
    """
    Glob every annotation CSV for one match: `DFL-MAT-{match_id}_*.csv`.
    Returns {"{half}_{annotator}": path}, keyed the same way
    Stats_Manual_Annotation_App.py's load_annotations() used to.
    """
    pattern = os.path.join(annotation_dir, f"DFL-MAT-{match_id}_*.csv")
    files = sorted(glob.glob(pattern))
    result = {}
    for path in files:
        basename = os.path.splitext(os.path.basename(path))[0]
        parts = basename.split("_", 2)
        if len(parts) < 3:
            continue
        key = f"{parts[1]}_{parts[2]}"
        result[key] = path
    return result
