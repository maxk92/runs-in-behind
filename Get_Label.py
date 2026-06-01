"""
consolidate_gt_and_label.py
===========================
Multi-annotator ground truth consolidation and prediction labelling
for runs-in-behind detection in football.

PIPELINE:
  1. Dynamic scan of GT_DIR → detects all MATCH_IDs and annotators.
  2. Consolidation via sweep-line Union-Find: for each (team, player_jid, half)
     triplet, merges overlapping segments → start = min, end = max;
     the highest-priority outcome is kept.
  3. Prediction labelling: label=1 if overlap (±TOLERANCE) with the
     consolidated GT on (team, jID, half).

FILE NAMING:
  Annotations : DFL-MAT-{MATCH_ID}_{HALF}_{ANNOTATOR}.csv
  Predictions : runs_spatial_{MATCH_ID}.csv
                or runs_spatial_{SHORT_ID}.csv  (e.g. J03WMX instead of DFL-MAT-J03WMX)
                → both forms are tried automatically.

HOME/AWAY MAPPING:
  GT annotations use "Home"/"Away" as team identifiers.
  Predictions use real club names (e.g. "FC Bayern München").
  The mapping is built automatically from the `location` column
  of the prediction file — no manual configuration required.
"""

import os
import re
import glob
import logging
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PRED_DIR  = r"C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_spatial"
GT_DIR    = r"C:\Users\arnau\Documents\projetde\runs-in-behind\annotation_app_output"

GT_CONSOLIDATED_DIR = r"C:\Users\arnau\Documents\projetde\runs-in-behind\GT_consolidated"
LABELED_PRED_DIR    = r"C:\Users\arnau\Documents\projetde\runs-in-behind\labeled_outputs"

FPS              = 25
TOLERANCE_FRAMES = 25   # ± frame tolerance for prediction ↔ GT matching

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ANNOT_PATTERN = re.compile(
    r"^(DFL-MAT-[A-Z0-9]+)_(firstHalf|secondHalf)_([^.]+)\.csv$",
    re.IGNORECASE,
)
OUTCOME_PRIORITY = {
    "running_player_received": 3,
    "other_player_received":   2,
    "none_received":           1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create output directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        log.info("Directory ready: %s", d)


def overlaps(s1: int, e1: int, s2: int, e2: int, tol: int = 0) -> bool:
    """Return True if [s1-tol, e1+tol] and [s2, e2] overlap."""
    return (s1 - tol) <= e2 and s2 <= (e1 + tol)


def find_prediction_file(pred_dir: str, match_id: str) -> str | None:
    """
    Try runs_spatial_{match_id}.csv then runs_spatial_{short_id}.csv.
    Returns the path if found, None otherwise.
    """
    candidates = [match_id]
    if match_id.upper().startswith("DFL-MAT-"):
        candidates.append(match_id[len("DFL-MAT-"):])
    for cid in candidates:
        fp = os.path.join(pred_dir, f"runs_spatial_{cid}.csv")
        if os.path.exists(fp):
            return fp
    return None


def build_location_mapping(pred_df: pd.DataFrame) -> dict[str, str]:
    """
    Dynamically build the {"Home": "real name", "Away": "real name"} mapping
    from the `location` column of the prediction file.

    GT annotations use "Home"/"Away"; predictions use real club names.
    This mapping reconciles the two.

    Returns an empty dict if the `location` column is absent
    (direct team-name matching is used in that case).
    """
    if "location" not in pred_df.columns or "team" not in pred_df.columns:
        return {}

    mapping = (
        pred_df[["team", "location"]]
        .drop_duplicates()
        .dropna()
        .set_index("location")["team"]
        .to_dict()
    )
    # Normalise keys to Title-case (Home/Away)
    mapping = {k.strip().title(): v.strip() for k, v in mapping.items()}
    log.info("  Home/Away → team mapping: %s", mapping)
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DYNAMIC SCAN
# ─────────────────────────────────────────────────────────────────────────────

def scan_annotation_files(gt_dir: str) -> dict[str, dict[str, list]]:
    """
    Scan gt_dir and return:
        { match_id: { half: [ (annotator, filepath), ... ] } }
    Fully dynamic — no annotator names or match IDs hardcoded.
    """
    index: dict[str, dict[str, list]] = {}
    for fp in glob.glob(os.path.join(gt_dir, "*.csv")):
        m = ANNOT_PATTERN.match(os.path.basename(fp))
        if not m:
            log.debug("File skipped (pattern not recognised): %s", os.path.basename(fp))
            continue
        match_id, half, annotator = m.group(1), m.group(2), m.group(3)
        index.setdefault(match_id, {}).setdefault(half, []).append((annotator, fp))

    if not index:
        log.warning("No annotation files found in: %s", gt_dir)
    else:
        for mid, hmap in sorted(index.items()):
            annotators = sorted({a for segs in hmap.values() for a, _ in segs})
            log.info(
                "Match detected: %-22s | halves: %-30s | annotators: %s",
                mid, list(hmap.keys()), annotators,
            )
    return index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GT CONSOLIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_annotation_file(fp: str, annotator: str) -> pd.DataFrame:
    """
    Load an annotation file, normalise column names, and drop rows
    without team/player_jid (orphan annotations).
    """
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "player_jid" not in df.columns and "jid" in df.columns:
        df.rename(columns={"jid": "player_jid"}, inplace=True)

    required = {"start_frame", "end_frame", "team", "player_jid", "half"}
    missing = required - set(df.columns)
    if missing:
        log.warning("Missing columns in %s: %s — file skipped.", fp, missing)
        return pd.DataFrame()

    before = len(df)
    df = df.dropna(subset=["team", "player_jid"])
    df["player_jid"] = df["player_jid"].astype(int)
    dropped = before - len(df)
    if dropped:
        log.debug("%s: %d row(s) without team/player_jid dropped.", fp, dropped)

    df["annotator"] = annotator
    return df


def consolidate_half(half_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep-line Union-Find:
    For each (team, player_jid) group, merge overlapping segments →
    start = min(starts), end = max(ends).
    The highest-priority outcome in the merged group is kept.
    """
    if half_df.empty:
        return pd.DataFrame()

    rows = []
    for (team, jid), grp in half_df.groupby(["team", "player_jid"], sort=False):
        segs = (
            grp[["start_frame", "end_frame", "outcome", "player", "annotator"]]
            .sort_values("start_frame")
            .reset_index(drop=True)
        )
        merged: list[dict] = []

        for _, r in segs.iterrows():
            s, e = int(r["start_frame"]), int(r["end_frame"])
            outcome   = str(r["outcome"]) if pd.notna(r["outcome"]) else "none_received"
            player    = r["player"]
            annotator = r["annotator"]

            if merged and overlaps(merged[-1]["start_frame"], merged[-1]["end_frame"], s, e):
                merged[-1]["start_frame"] = min(merged[-1]["start_frame"], s)
                merged[-1]["end_frame"]   = max(merged[-1]["end_frame"],   e)
                merged[-1]["annotators"].add(annotator)
                if OUTCOME_PRIORITY.get(outcome, 0) > OUTCOME_PRIORITY.get(merged[-1]["outcome"], 0):
                    merged[-1]["outcome"] = outcome
            else:
                merged.append({
                    "start_frame": s,  "end_frame":  e,
                    "team":        str(team),
                    "player_jid":  int(jid),
                    "player":      player,
                    "outcome":     outcome,
                    "annotators":  {annotator},
                })

        for seg in merged:
            seg["annotator_count"] = len(seg["annotators"])
            seg["annotators"]      = "|".join(sorted(seg["annotators"]))
            seg["start_time_s"]    = round(seg["start_frame"] / FPS, 3)
            seg["end_time_s"]      = round(seg["end_frame"]   / FPS, 3)
            seg["duration_s"]      = round(
                (seg["end_frame"] - seg["start_frame"]) / FPS, 3
            )
            rows.append(seg)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["team", "player_jid", "start_frame"])
        .reset_index(drop=True)
    )


def consolidate_match(
    match_id: str, half_map: dict[str, list], gt_out_dir: str
) -> pd.DataFrame:
    """
    Consolidate all halves for one match, save the GT CSV,
    and return the consolidated GT DataFrame (all halves).
    """
    all_halves: list[pd.DataFrame] = []

    for half, annotator_files in half_map.items():
        dfs = []
        for annotator, fp in annotator_files:
            df = load_annotation_file(fp, annotator)
            if not df.empty:
                df["half"] = half
                dfs.append(df)
            log.info(
                "  ↳ Loaded: %-12s | %-10s | %d valid rows",
                annotator, half, len(df),
            )

        if not dfs:
            log.warning("   No valid annotations for %s — %s", match_id, half)
            continue

        half_df      = pd.concat(dfs, ignore_index=True)
        consolidated = consolidate_half(half_df)

        if not consolidated.empty:
            consolidated["half"] = half
            all_halves.append(consolidated)
            log.info(
                "   Consolidated GT %-10s: %d runs  (from %d raw annotations)",
                half, len(consolidated), len(half_df),
            )

    if not all_halves:
        log.warning("No consolidated GT for %s.", match_id)
        return pd.DataFrame()

    gt = pd.concat(all_halves, ignore_index=True)

    col_order = [
        "half", "team", "player_jid", "player",
        "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s",
        "outcome", "annotator_count", "annotators",
    ]
    gt = gt[[c for c in col_order if c in gt.columns]]

    out_path = os.path.join(gt_out_dir, f"GT_consolidated_{match_id}.csv")
    gt.to_csv(out_path, index=False)
    log.info("   GT saved → %s  (%d runs)", out_path, len(gt))
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREDICTION LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def label_predictions(
    match_id: str, gt: pd.DataFrame, pred_dir: str, out_dir: str
) -> None:
    """
    Load runs_behind_{match_id}.csv, resolve the Home/Away → real name mapping,
    add a 'label' column (1 if overlap ±TOLERANCE with GT, 0 otherwise),
    and save to out_dir.
    """
    pred_path = find_prediction_file(pred_dir, match_id)

    if pred_path is None:
        short = match_id[len("DFL-MAT-"):] if match_id.upper().startswith("DFL-MAT-") else match_id
        log.warning(
            "⚠ No prediction file for %s "
            "(looked for: runs_behind_%s.csv and runs_behind_%s.csv)",
            match_id, match_id, short,
        )
        return

    preds = pd.read_csv(pred_path)
    preds.columns = [c.strip() for c in preds.columns]

    # GT annotations use "Home"/"Away"; predictions have real club names.
    # Build the mapping from the `location` column.
    loc_map = build_location_mapping(preds)

    # jID column in predictions (case-insensitive lookup)
    jid_col = next((c for c in preds.columns if c.lower() == "jid"), None)
    if jid_col is None:
        log.error(
            "jID column not found in %s.\nAvailable columns: %s",
            pred_path, list(preds.columns),
        )
        return

    # GT index with resolved team names.
    # Key: (team_in_pred, player_jid, half) → [(start, end), ...]
    gt_index: dict[tuple, list[tuple[int, int]]] = {}
    for _, row in gt.iterrows():
        gt_team_raw = str(row["team"]).strip()   # "Home" or "Away"
        team_key = loc_map.get(gt_team_raw.title(), gt_team_raw)
        key = (team_key, int(row["player_jid"]), str(row["half"]).strip())
        gt_index.setdefault(key, []).append(
            (int(row["start_frame"]), int(row["end_frame"]))
        )

    # Match prediction rows against GT index
    labels = []
    for _, pred in preds.iterrows():
        try:
            p_team = str(pred["team"]).strip()
            p_jid  = int(pred[jid_col])
            p_half = str(pred["half"]).strip()
            p_s    = int(pred["start_frame"])
            p_e    = int(pred["end_frame"])
        except (ValueError, KeyError) as exc:
            log.debug("Prediction row skipped (%s)", exc)
            labels.append(0)
            continue

        key = (p_team, p_jid, p_half)
        matched = any(
            overlaps(p_s, p_e, gt_s, gt_e, tol=TOLERANCE_FRAMES)
            for gt_s, gt_e in gt_index.get(key, [])
        )
        labels.append(1 if matched else 0)

    preds["label"] = labels
    pos = sum(labels)
    log.info(
        "   Predictions labelled: %d total | %d label=1 (%.1f%%) | %d label=0",
        len(labels), pos,
        100 * pos / len(labels) if labels else 0.0,
        len(labels) - pos,
    )

    out_path = os.path.join(out_dir, f"labeled_runs_behind_{match_id}.csv")
    preds.to_csv(out_path, index=False)
    log.info("   Labelled predictions saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("  GT CONSOLIDATION & LABELLING  —  starting")
    log.info("=" * 70)

    ensure_dirs(GT_CONSOLIDATED_DIR, LABELED_PRED_DIR)

    log.info("\n── Scanning annotations (%s)", GT_DIR)
    index = scan_annotation_files(GT_DIR)
    if not index:
        log.error(
            "No annotations detected. "
            "Check GT_DIR and file naming: DFL-MAT-{ID}_{half}_{annotator}.csv"
        )
        return

    for match_id, half_map in sorted(index.items()):
        log.info("\n── MATCH: %s", match_id)
        gt = consolidate_match(match_id, half_map, GT_CONSOLIDATED_DIR)
        if gt.empty:
            log.warning("Empty GT for %s — labelling skipped.", match_id)
            continue
        label_predictions(match_id, gt, PRED_DIR, LABELED_PRED_DIR)

    log.info("\n%s", "=" * 70)
    log.info("  DONE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()