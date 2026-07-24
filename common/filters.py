"""
Single source of truth for the "run in behind" candidate filter, applied to
the movements DataFrame produced by `segmentation.discretisation.process_match`.

Previously this predicate was re-implemented independently (with drifting
thresholds) in segmentation/extract_runs_behind.py, movement_classification.py,
extraction_method_comparison.py, and old_code/extract_discrete_efforts.ipynb.
"""

# NOTE on the sports-science definition (Definition_Tiefenlauf.pdf):
# it specifies a minimum distance of 3m and a movement angle of 45-135
# degrees. The values below are the team's practical stand-ins that have
# actually been used to produce every runs_behind_*.csv file so far:
#   - RIB_MIN_DISTANCE_M = 2.5, not 3.0 ("in the definition the threshold is
#     3m, but we use 2.5m to avoid computing error" -- original comment in
#     segmentation/extract_runs_behind.py).
#   - RIB_MIN_DIRECTION = 0.3 thresholds the `direction` column (an L1
#     displacement ratio dx/(|dx|+|dy|) computed in process_match, NOT a
#     literal degree angle) as a proxy for the 45-135 degree criterion.
# Changing these numbers changes which movements qualify as runs-in-behind
# and will change every downstream runs_behind_*.csv output.
RIB_MIN_DISTANCE_M = 2.5
RIB_MIN_DIRECTION = 0.3
RIB_SPEED_CATEGORIES = {"running", "sprinting"}


def is_run_in_behind(df_movements,
                      min_distance_m=RIB_MIN_DISTANCE_M,
                      min_direction=RIB_MIN_DIRECTION,
                      speed_categories=RIB_SPEED_CATEGORIES):
    """
    Boolean mask selecting rows of `df_movements` that qualify as a
    "run in behind" candidate: the performing team is in possession (or
    contested possession), the peak speed reaches running/sprinting, the
    covered distance exceeds `min_distance_m`, and `direction` (vertical
    progress toward goal) exceeds `min_direction`.

    `df_movements` must have the columns produced by
    `segmentation.discretisation.process_match`: `possession`, `location`,
    `possession_contested`, `speed_category`, `distance_m`, `direction`.
    """
    return (
        ((df_movements["possession"] == df_movements["location"])
         | df_movements["possession_contested"])
        & df_movements["speed_category"].isin(speed_categories)
        & (df_movements["distance_m"] > min_distance_m)
        & (df_movements["direction"] > min_direction)
    )
