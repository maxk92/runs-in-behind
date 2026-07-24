import pandas as pd
import os

from common import config
from common.iou import match_by_frame_tolerance
from common.timecodes import frames_to_timecode
from common.annotations import annotation_file_path, discover_annotators

# ── Configuration ─────────────────────────────────────────────────────────────
MATCH_IDS = config.DEFAULT_MATCH_IDS
HALVES    = ['firstHalf', 'secondHalf']
PRED_DIR  = config.AUTO_OUTPUT_DIR
GT_DIR    = config.ANNOTATION_DIR
FN_OUTPUT = config.FALSE_NEGATIVES_PATH
FPS       = 25   # DFL framerate

def frames_to_tc(frames):
    """Convert a frame number to a hh:mm:ss timecode."""
    return frames_to_timecode(frames, fps=FPS, fmt="hms")


def load_pred(match_id):
    path = os.path.join(PRED_DIR, f"runs_behind_{match_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={'location': 'team_ha'})
    return df


def load_gt(match_id, half, annotator):
    path = annotation_file_path(GT_DIR, match_id, half, annotator)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=['player_jid'])
    df['player_jid'] = df['player_jid'].astype(int)
    return df


def compute_metrics_with_fn(match_id, annotator):
    """
    Return (TP, FP, FN, fn_rows) for one match and one annotator.
    fn_rows: list of dicts representing unmatched GT annotations (false negatives).
    """
    pred_all = load_pred(match_id)
    if pred_all is None:
        return None

    total_tp = total_fp = total_fn = 0
    fn_rows = []

    for half in HALVES:
        gt = load_gt(match_id, half, annotator)
        if gt is None:
            continue

        pred = pred_all[pred_all['half'] == half].reset_index(drop=True)
        gt   = gt.reset_index(drop=True)

        pred_matched, gt_matched = match_by_frame_tolerance(pred, gt, tolerance_frames=25)

        tp = sum(pred_matched)
        total_tp += tp
        total_fp += len(pred_matched) - tp
        fn_count = sum(1 for m in gt_matched if not m)
        total_fn += fn_count

        # Collect unmatched GT rows (false negatives)
        for gi, grow in gt.iterrows():
            if not gt_matched[gi]:
                row = grow.to_dict()
                row['match_id']  = match_id
                row['half']      = half
                row['annotator'] = annotator
                fn_rows.append(row)

    return total_tp, total_fp, total_fn, fn_rows


def print_report(results, annotators):
    print("=" * 72)
    print("  EVALUATION REPORT — Runs-in-Behind Detection")
    print("  TP criteria: team (Home/Away) + jID + start_frame ∈ [GT_start, GT_end]")
    print("=" * 72)

    for ann in annotators:
        print(f"\n    Annotator: {ann.upper()}")
        print("  " + "-" * 68)
        print(f"  {'Match':<12} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>12} {'Recall':>10}")
        print("  " + "-" * 68)

        total_tp = total_fp = total_fn = 0
        for mid, (tp, fp, fn, _) in results[ann].items():
            prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            print(f"  {mid:<12} {tp:>6} {fp:>6} {fn:>6} {prec:>12.3f} {rec:>10.3f}")
            total_tp += tp; total_fp += fp; total_fn += fn

        t_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
        t_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
        print("  " + "-" * 68)
        print(f"  {'TOTAL':<12} {total_tp:>6} {total_fp:>6} {total_fn:>6} {t_prec:>12.3f} {t_rec:>10.3f}")

    print("\n" + "=" * 72)


def export_false_negatives(results, output_path):
    all_fn = []
    for ann, matches in results.items():
        for mid, (tp, fp, fn, fn_rows) in matches.items():
            all_fn.extend(fn_rows)

    if not all_fn:
        print("\n    No false negatives detected.")
        return

    df_fn = pd.DataFrame(all_fn)

    # Add hh:mm:ss timecodes next to frame numbers
    df_fn['start_tc'] = df_fn['start_frame'].apply(frames_to_tc)
    df_fn['end_tc']   = df_fn['end_frame'].apply(frames_to_tc)

    # Context columns first (frames + timecodes side by side)
    priority_cols = ['match_id', 'half', 'annotator', 'team', 'player_jid',
                     'start_frame', 'start_tc', 'end_frame', 'end_tc']
    other_cols    = [c for c in df_fn.columns if c not in priority_cols]
    df_fn = df_fn[priority_cols + other_cols]

    df_fn.to_csv(output_path, index=False)
    print(f"\n    {len(df_fn)} false negative(s) exported → {output_path}")


if __name__ == "__main__":
    annotators = discover_annotators(GT_DIR)
    print(f"Annotators found: {annotators}\n")

    results = {}
    for ann in annotators:
        results[ann] = {}
        for mid in MATCH_IDS:
            res = compute_metrics_with_fn(mid, ann)
            if res is not None:
                results[ann][mid] = res

    print_report(results, annotators)
    export_false_negatives(results, FN_OUTPUT)