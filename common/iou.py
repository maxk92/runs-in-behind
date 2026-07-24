"""
Temporal-overlap matching helpers shared by the manual-vs-automated
evaluation/merging scripts (Extraction_Evaluation.py, Stats_Manual_Annotation_App.py,
enriched_tool.py).

`temporal_iou` is the one canonical IoU implementation -- previously defined
identically (but independently) in Stats_Manual_Annotation_App.py and
enriched_tool.py. `match_by_frame_tolerance` replaces the greedy first-fit
matching loop that used to live inline in Extraction_Evaluation.py.

Note: Stats_Manual_Annotation_App.py's `match_segments` (directional
best-overlap match across a player+half candidate pool, used for
inter-annotator agreement) and enriched_tool.py's `find_best_segment`
(vectorized best-IoU match with its own column auto-detection, used to find
the automated segment overlapping a manual annotation) are algorithmically
distinct from each other and from `match_by_frame_tolerance` below, so they
remain file-local rather than being forced into one generic matcher -- both
now call `temporal_iou` from here instead of defining their own copy.
"""


def temporal_iou(start_a, end_a, start_b, end_b) -> float:
    """
    Intersection-over-Union of two temporal intervals [start_a, end_a] and
    [start_b, end_b], expressed in the same unit (frames or seconds).

    Returns a value in [0, 1]; returns 0.0 if either interval is degenerate
    (end <= start) or the union is empty.
    """
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union if union > 0 else 0.0


def match_by_frame_tolerance(pred_df, gt_df, tolerance_frames=25,
                              pred_team_col="team_ha", pred_jid_col="jID",
                              pred_start_col="start_frame", pred_end_col="end_frame",
                              gt_team_col="team", gt_jid_col="player_jid",
                              gt_start_col="start_frame", gt_end_col="end_frame"):
    """
    Greedy first-fit match between predicted (automated) segments and
    ground-truth (manual) segments: a pair matches if team + jID are equal
    and either segment's start frame falls within the other's
    [start - tolerance_frames, end + tolerance_frames] window.

    This is NOT a true IoU match (see `temporal_iou` for that) -- it
    preserves the exact semantics originally inlined in
    Extraction_Evaluation.py's `compute_metrics_with_fn`.

    Returns (pred_matched, gt_matched): two lists of booleans, aligned with
    `pred_df.reset_index(drop=True)` / `gt_df.reset_index(drop=True)`.
    """
    pred = pred_df.reset_index(drop=True)
    gt = gt_df.reset_index(drop=True)

    gt_matched = [False] * len(gt)
    pred_matched = [False] * len(pred)

    for pi, prow in pred.iterrows():
        for gi, grow in gt.iterrows():
            if gt_matched[gi]:
                continue
            if (prow[pred_team_col] == grow[gt_team_col]
                    and int(prow[pred_jid_col]) == int(grow[gt_jid_col])
                    and ((grow[gt_start_col] - tolerance_frames <= prow[pred_start_col] <= grow[gt_end_col] + tolerance_frames)
                         or (prow[pred_start_col] - tolerance_frames <= grow[gt_start_col] <= prow[pred_end_col] + tolerance_frames))):
                gt_matched[gi] = True
                pred_matched[pi] = True
                break

    return pred_matched, gt_matched
