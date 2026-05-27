"""
best_candidate_iou.py
=====================
Consolidation of fragmented predictions using the "Best Candidate" strategy.

PROBLEM:
  A single real run (GT) can generate multiple label=1 predictions (fragments).
  Only the fragment with the highest IoU against the GT is kept,
  while label=0 predictions (False Positives) are preserved for precision computation.

PIPELINE:
  1. Load labeled_runs_behind_{MATCH_ID}.csv  (already-labeled predictions).
  2. Load GT_consolidated_{MATCH_ID}.csv       (consolidated ground truth).
     → If missing, reconstruct GT groups by temporal proximity (fallback).
  3. For each GT run, find all label=1 fragments that overlap it.
  4. Compute temporal IoU between each fragment and the GT.
  5. Keep only the fragment with maximum IoU; discard the others.
  6. Save best_candidate_{MATCH_ID}.csv.

FIX v1.1:
  The consolidated GT uses "Home"/"Away" as team names, while predictions use
  real club names. resolve_gt_team() now applies the mapping in ALL comparisons
  (selection AND iou_gt column), fixing the systematic iou_gt=0 bug from v1.0.

Version: 1.1
"""

import os
import logging
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MATCH_ID = "DFL-MAT-J03WMX"

LABELED_PRED_DIR    = r"C:\Users\arnau\Documents\projetde\runs-in-behind\labeled_outputs"
GT_CONSOLIDATED_DIR = r"C:\Users\arnau\Documents\projetde\runs-in-behind\GT_consolidated"
OUTPUT_DIR          = r"C:\Users\arnau\Documents\projetde\runs-in-behind\best_candidate"

TOLERANCE_FRAMES    = 25    # must match Get_Label.py
GAP_FRAMES_FALLBACK = 75    # ≈ 3 s at 25 fps (fallback without GT)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def temporal_iou(s_pred: int, e_pred: int, s_gt: int, e_gt: int) -> float:
    """Temporal IoU between [s_pred, e_pred] and [s_gt, e_gt]."""
    inter = max(0, min(e_pred, e_gt) - max(s_pred, s_gt))
    union = (e_pred - s_pred) + (e_gt - s_gt) - inter
    return 0.0 if union <= 0 else inter / union


def overlaps_with_tolerance(s_pred: int, e_pred: int,
                             s_gt: int, e_gt: int,
                             tol: int = TOLERANCE_FRAMES) -> bool:
    """Overlap check with ±tol frame tolerance (same logic as Get_Label.py)."""
    return (s_pred - tol) <= e_gt and s_gt <= (e_pred + tol)


# ─────────────────────────────────────────────────────────────────────────────
# Home/Away → real names MAPPING  [FIX v1.1]
# ─────────────────────────────────────────────────────────────────────────────

def build_location_mapping(pred_df: pd.DataFrame) -> dict:
    """
    Builds {"Home": "FC Bayern München", "Away": "..."} from the `location`
    column of the predictions — same logic as Get_Label.py.
    Returns {} if the column is missing (direct team name matching).
    """
    if "location" not in pred_df.columns or "team" not in pred_df.columns:
        log.warning("Column 'location' missing — falling back to direct team name matching.")
        return {}
    mapping = (
        pred_df[["team", "location"]]
        .drop_duplicates()
        .dropna()
        .set_index("location")["team"]
        .to_dict()
    )
    mapping = {k.strip().title(): v.strip() for k, v in mapping.items()}
    log.info("Home/Away → team mapping: %s", mapping)
    return mapping


def resolve_gt_team(raw: str, loc_map: dict) -> str:
    """Translates 'Home'/'Away' to the real club name. Returns raw if not in mapping."""
    return loc_map.get(raw.strip().title(), raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_labeled_predictions(match_id: str, labeled_dir: str) -> pd.DataFrame:
    path = os.path.join(labeled_dir, f"labeled_runs_behind_{match_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    log.info("Predictions loaded: %d rows  (%s)", len(df), path)
    return df


def load_gt(match_id: str, gt_dir: str) -> pd.DataFrame | None:
    path = os.path.join(gt_dir, f"GT_consolidated_{match_id}.csv")
    if not os.path.exists(path):
        log.warning("Consolidated GT missing (%s) — falling back to proximity grouping.", path)
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    log.info("Consolidated GT loaded: %d runs  (%s)", len(df), path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_gt_from_predictions(preds: pd.DataFrame,
                                     gap: int = GAP_FRAMES_FALLBACK) -> pd.DataFrame:
    """
    Groups label=1 rows sharing the same (team, jID, half) that are fewer than
    `gap` frames apart → pseudo GT runs using real team names (not Home/Away).
    """
    log.info("Reconstructing pseudo-GT (gap ≤ %d frames).", gap)
    positives = preds[preds["label"] == 1].copy()

    jid_col = next((c for c in positives.columns if c.lower() == "jid"), None)
    if jid_col is None:
        raise ValueError("jID column not found in predictions.")

    rows, gt_id = [], 0
    for (team, jid, half), grp in positives.groupby(["team", jid_col, "half"], sort=False):
        segs  = grp.sort_values("start_frame").reset_index(drop=True)
        cur_s = int(segs.loc[0, "start_frame"])
        cur_e = int(segs.loc[0, "end_frame"])

        for i in range(1, len(segs)):
            s = int(segs.loc[i, "start_frame"])
            e = int(segs.loc[i, "end_frame"])
            if s - cur_e <= gap:
                cur_e = max(cur_e, e)
            else:
                rows.append({"gt_id": gt_id, "team": team, "player_jid": jid,
                              "half": half, "start_frame": cur_s, "end_frame": cur_e})
                gt_id += 1
                cur_s, cur_e = s, e

        rows.append({"gt_id": gt_id, "team": team, "player_jid": jid,
                     "half": half, "start_frame": cur_s, "end_frame": cur_e})
        gt_id += 1

    gt = pd.DataFrame(rows)
    log.info("Pseudo-GT reconstructed: %d runs.", len(gt))
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# BEST CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best_candidates(preds: pd.DataFrame,
                           gt: pd.DataFrame,
                           loc_map: dict) -> pd.DataFrame:
    """
    For each GT run:
      - Find all label=1 fragments overlapping it (±TOLERANCE).
      - Keep the fragment with the highest IoU, discard the rest.
    label=0 predictions (FP) are preserved untouched.
    loc_map: Home/Away → real name mapping, applied to every comparison.
    """
    jid_col_pred = next((c for c in preds.columns if c.lower() == "jid"), None)
    jid_col_gt   = next((c for c in gt.columns if c.lower() in ("player_jid", "jid")), None)

    if jid_col_pred is None:
        raise ValueError("jID column not found in predictions.")
    if jid_col_gt is None:
        raise ValueError("player_jid/jid column not found in GT.")

    if "gt_id" not in gt.columns:
        gt = gt.copy().reset_index(drop=True)
        gt["gt_id"] = gt.index

    pos_idx  = preds.index[preds["label"] == 1].tolist()
    to_drop: set = set()
    stats = {"gt_runs": 0, "single": 0, "multi": 0, "dropped": 0}

    for _, run in gt.iterrows():
        gt_team  = resolve_gt_team(str(run["team"]), loc_map)
        gt_jid   = int(run[jid_col_gt])
        gt_half  = str(run["half"]).strip()
        gt_start = int(run["start_frame"])
        gt_end   = int(run["end_frame"])
        stats["gt_runs"] += 1

        candidates = [
            idx for idx in pos_idx
            if (
                str(preds.at[idx, "team"]).strip() == gt_team
                and int(preds.at[idx, jid_col_pred]) == gt_jid
                and str(preds.at[idx, "half"]).strip() == gt_half
                and overlaps_with_tolerance(
                    int(preds.at[idx, "start_frame"]),
                    int(preds.at[idx, "end_frame"]),
                    gt_start, gt_end,
                )
            )
        ]

        if len(candidates) <= 1:
            stats["single"] += 1
            continue

        iou_scores = {
            idx: temporal_iou(
                int(preds.at[idx, "start_frame"]),
                int(preds.at[idx, "end_frame"]),
                gt_start, gt_end,
            )
            for idx in candidates
        }
        best_idx = max(iou_scores, key=iou_scores.get)
        losers   = [i for i in candidates if i != best_idx]
        to_drop.update(losers)
        stats["multi"]   += 1
        stats["dropped"] += len(losers)

        log.debug(
            "GT run %d (%s | jID=%d | %s | [%d–%d]): "
            "%d candidates → best=%d IoU=%.3f, dropped=%d",
            int(run["gt_id"]), gt_team, gt_jid, gt_half,
            gt_start, gt_end, len(candidates),
            best_idx, iou_scores[best_idx], len(losers),
        )

    log.info(
        "Selection summary:\n"
        "  GT runs processed  : %d\n"
        "  Unambiguous        : %d\n"
        "  With fragments     : %d\n"
        "  Fragments dropped  : %d",
        stats["gt_runs"], stats["single"], stats["multi"], stats["dropped"],
    )

    result = preds.drop(index=list(to_drop)).reset_index(drop=True)
    _add_iou_column(result, gt, jid_col_pred, jid_col_gt, loc_map)
    return result


def _add_iou_column(preds: pd.DataFrame, gt: pd.DataFrame,
                    jid_col_pred: str, jid_col_gt: str,
                    loc_map: dict) -> None:
    """
    Adds the `iou_gt` column for label=1 rows.
    FIX v1.1: resolve_gt_team() is called here, fixing the iou_gt=0 bug
    caused by comparing 'Home' against 'FC Bayern München'.
    """
    ious = []
    for _, pred in preds.iterrows():
        if pred["label"] != 1:
            ious.append(None)
            continue

        p_team = str(pred["team"]).strip()
        p_jid  = int(pred[jid_col_pred])
        p_half = str(pred["half"]).strip()
        p_s    = int(pred["start_frame"])
        p_e    = int(pred["end_frame"])

        best_iou = 0.0
        for _, run in gt.iterrows():
            gt_team = resolve_gt_team(str(run["team"]), loc_map)
            if (
                gt_team == p_team
                and int(run[jid_col_gt]) == p_jid
                and str(run["half"]).strip() == p_half
                and overlaps_with_tolerance(p_s, p_e,
                                            int(run["start_frame"]),
                                            int(run["end_frame"]))
            ):
                best_iou = max(
                    best_iou,
                    temporal_iou(p_s, p_e,
                                 int(run["start_frame"]),
                                 int(run["end_frame"])),
                )

        ious.append(round(best_iou, 4))

    preds["iou_gt"] = ious


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("  BEST CANDIDATE SELECTION v1.1  —  match %s", MATCH_ID)
    log.info("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load labeled predictions
    preds = load_labeled_predictions(MATCH_ID, LABELED_PRED_DIR)
    n_pos_before = (preds["label"] == 1).sum()
    log.info("Before selection: %d label=1  |  %d label=0",
             n_pos_before, (preds["label"] == 0).sum())

    # 2. Build Home/Away → real names mapping from predictions
    loc_map = build_location_mapping(preds)

    # 3. Load consolidated GT or fall back
    gt = load_gt(MATCH_ID, GT_CONSOLIDATED_DIR)
    if gt is None:
        gt      = reconstruct_gt_from_predictions(preds)
        loc_map = {}   # real names already used in fallback, no resolution needed

    # 4. Best candidate selection + IoU computation
    result = select_best_candidates(preds, gt, loc_map)

    n_pos_after = (result["label"] == 1).sum()
    log.info(
        "After selection : %d label=1  |  %d label=0  "
        "(%d redundant fragments dropped)",
        n_pos_after, (result["label"] == 0).sum(),
        n_pos_before - n_pos_after,
    )

    # 5. Save output
    out_path = os.path.join(OUTPUT_DIR, f"best_candidate_{MATCH_ID}.csv")
    result.to_csv(out_path, index=False)
    log.info("Result saved → %s", out_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()