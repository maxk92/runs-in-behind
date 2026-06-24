"""
classify_movement_shape.py
==========================
Classifies each movement extracted by process_match into three shape categories:

  - 'straight'    : nearly straight run along the attack axis
  - 'diagonal'    : straight run but at a significant angle to the attack axis
  - 'curvilinear' : curved path, best described by 2 or more linear segments

Classification relies on raw (x, y) trajectories stored in dict_trajectories,
enabling reliable geometric analysis frame by frame.

Usage (as a module):
--------------------
    from classify_movement_shape import classify_runs_shape
    df_runs_classified = classify_runs_shape(df_runs, dict_traj)

Usage (standalone):
-------------------
    python classify_movement_shape.py
"""

import json
import os
import traceback

import numpy as np
import pandas as pd
from Discretisation_optimised2_1 import process_match


# ============================================================================
# Configuration
# ============================================================================

# Paths
DATA_DIR   = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223'
JSON_PATH  = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\direction_idsse_video.json'
OUTPUT_DIR = r'C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_shape_classified'

MATCH_IDS = ['J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']

# Geometric thresholds
LINEARITY_THRESHOLD_M = 0.35   # max perpendicular RMSE (m) to be considered linear
DIAGONAL_ANGLE_DEG    = 22.5   # angle threshold (°) separating straight from diagonal
MIN_FRAMES            = 5      # minimum frames required to attempt classification


# ============================================================================
# Geometric helpers
# ============================================================================

def _resample_trajectory(x: np.ndarray, y: np.ndarray,
                          n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to n_points equally spaced along arc-length.
    Normalises trajectories of different durations for fair comparison.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx ** 2 + dy ** 2)
    arc = np.concatenate([[0], np.cumsum(seg_len)])
    total = arc[-1]
    if total < 1e-6:
        return x, y
    t_uniform = np.linspace(0, total, n_points)
    x_r = np.interp(t_uniform, arc, x)
    y_r = np.interp(t_uniform, arc, y)
    return x_r, y_r


def _linear_residual_rms(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a line to (x, y) via PCA and return the RMSE of perpendicular residuals (m).
    A low RMSE indicates a nearly straight trajectory.
    """
    if len(x) < 2:
        return 0.0
    cx, cy = x.mean(), y.mean()
    xc, yc = x - cx, y - cy
    cov = np.array([[xc @ xc, xc @ yc],
                    [xc @ yc, yc @ yc]]) / len(x)
    _, vecs = np.linalg.eigh(cov)
    u = vecs[:, 1]                         # principal direction
    perp = xc * (-u[1]) + yc * u[0]       # perpendicular residuals
    return float(np.sqrt((perp ** 2).mean()))


def _attack_angle_deg(x: np.ndarray, y: np.ndarray,
                      attack_sign: float) -> float:
    """
    Absolute angle (0-90°) between the start->end vector and the horizontal attack axis.
    attack_sign: +1 if the team plays left-to-right, -1 otherwise.
    0° = running straight toward goal, 90° = purely lateral.
    """
    dx = (x[-1] - x[0]) * attack_sign
    dy = y[-1] - y[0]
    if abs(dx) + abs(dy) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))


def _estimate_n_segments(x: np.ndarray, y: np.ndarray,
                          threshold: float) -> int:
    """
    Estimate the minimum number of linear segments needed to describe the trajectory.
    Greedy algorithm: open a new segment whenever RMSE exceeds the threshold.
    """
    if len(x) < 4:
        return 1
    n_seg   = 1
    start   = 0
    min_seg = max(4, len(x) // 10)

    for end in range(min_seg, len(x)):
        rmse = _linear_residual_rms(x[start:end + 1], y[start:end + 1])
        if rmse > threshold:
            n_seg += 1
            start  = end

    return n_seg


# ============================================================================
# Core classifier
# ============================================================================

def classify_movement(x: np.ndarray, y: np.ndarray,
                      attack_sign: float = 1.0,
                      linearity_threshold: float = LINEARITY_THRESHOLD_M,
                      diagonal_angle_deg: float = DIAGONAL_ANGLE_DEG) -> dict:
    """
    Classify a single movement from its raw trajectory.

    Parameters
    ----------
    x, y                : raw coordinates (one value per frame)
    attack_sign         : attack direction (+1 or -1)
    linearity_threshold : max perpendicular RMSE (m) to be considered linear
    diagonal_angle_deg  : min angle (°) to call a linear movement 'diagonal'

    Returns
    -------
    dict with keys:
        shape          : 'straight' | 'diagonal' | 'curvilinear' | 'unknown'
        linearity_rmse : perpendicular RMSE of the linear fit (m)
        angle_deg      : angle relative to the attack axis (°)
        n_segments     : 1 for linear, >=2 for curvilinear (estimated)
    """
    if len(x) < MIN_FRAMES:
        return {'shape': 'unknown', 'linearity_rmse': np.nan,
                'angle_deg': np.nan, 'n_segments': np.nan}

    xr, yr = _resample_trajectory(x, y, n_points=min(50, len(x)))
    rmse   = _linear_residual_rms(xr, yr)
    angle  = _attack_angle_deg(x, y, attack_sign)

    if rmse <= linearity_threshold:
        shape = 'straight' if angle <= diagonal_angle_deg else 'diagonal'
        n_seg = 1
    else:
        shape = 'curvilinear'
        n_seg = _estimate_n_segments(xr, yr, linearity_threshold)

    return {
        'shape':          shape,
        'linearity_rmse': round(rmse, 4),
        'angle_deg':      round(angle, 2),
        'n_segments':     n_seg,
    }


# ============================================================================
# Apply classification to a runs DataFrame
# ============================================================================

def classify_runs_shape(df_runs: pd.DataFrame,
                        dict_traj: dict,
                        linearity_threshold: float = LINEARITY_THRESHOLD_M,
                        diagonal_angle_deg: float = DIAGONAL_ANGLE_DEG
                        ) -> pd.DataFrame:
    """
    Add shape classification columns to df_runs using raw trajectories from dict_traj.

    Parameters
    ----------
    df_runs             : filtered runs DataFrame (output of Loop_with_setpiece_offsets)
    dict_traj           : {team: {xID: [movement_df, ...]}} from process_match
    linearity_threshold : RMSE threshold (m) for linearity test
    diagonal_angle_deg  : angle threshold (°) for straight vs diagonal split

    Returns
    -------
    df_runs with added columns:
        shape           : 'straight' | 'diagonal' | 'curvilinear'
        linearity_rmse  : perpendicular RMSE of the linear fit (m)
        angle_deg       : angle relative to attack axis (°)
        n_segments      : estimated number of linear segments
        shape_method    : 'trajectory' (raw frames) or 'fallback' (start/peak/end only)
    """
    df = df_runs.copy()

    shapes  = []
    rmses   = []
    angles  = []
    n_segs  = []
    methods = []

    # Build index: (start_frame, xID, team) -> movement DataFrame
    traj_index: dict[tuple, pd.DataFrame] = {}
    for team, player_dict in dict_traj.items():
        for xid, mov_list in player_dict.items():
            for mov_df in mov_list:
                if mov_df.empty:
                    continue
                key = (int(mov_df['frame'].iloc[0]), int(xid), team)
                traj_index[key] = mov_df

    for _, row in df.iterrows():
        attack_sign = float(row.get('attack_sign', 1.0))
        team        = row['location']
        xid         = int(row['xID'])
        sf          = int(row['start_frame'])

        mov_df = traj_index.get((sf, xid, team))

        if mov_df is not None and len(mov_df) >= MIN_FRAMES:
            x = mov_df['x'].values.astype(float)
            y = mov_df['y'].values.astype(float)
            methods.append('trajectory')
        else:
            # Fallback: use summary start / peak / end points
            x = np.array([row['x_start'], row['x_peak'], row['x_end']])
            y = np.array([row['y_start'], row['y_peak'], row['y_end']])
            methods.append('fallback')

        result = classify_movement(x, y, attack_sign,
                                   linearity_threshold, diagonal_angle_deg)
        shapes.append(result['shape'])
        rmses.append(result['linearity_rmse'])
        angles.append(result['angle_deg'])
        n_segs.append(result['n_segments'])

    df['shape']          = shapes
    df['linearity_rmse'] = rmses
    df['angle_deg']      = angles
    df['n_segments']     = n_segs
    df['shape_method']   = methods

    return df


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(JSON_PATH, 'r') as f:
        dict_direction = json.load(f)

    all_runs_classified = []

    print('=' * 80)
    print('MOVEMENT SHAPE CLASSIFICATION  (straight / diagonal / curvilinear)')
    print(f'  linearity_threshold = {LINEARITY_THRESHOLD_M} m  |  '
          f'diagonal_angle_threshold = {DIAGONAL_ANGLE_DEG} deg')
    print('=' * 80)

    for match_id in MATCH_IDS:
        print(f'\n{"=" * 60}')
        print(f'Processing match: {match_id}')
        print(f'{"=" * 60}')

        try:
            df_mov, _df_dist, dict_traj = process_match(
                match_id       = match_id,
                DATA_DIR       = DATA_DIR,
                dict_direction = dict_direction,
            )

            # Filter runs-in-behind (same criteria as Loop_with_setpiece_offsets)
            df_runs = df_mov[
                (
                    (df_mov['possession'] == df_mov['location']) |
                    df_mov['possession_contested']
                ) &
                df_mov['speed_category'].isin(['running', 'sprinting']) &
                (df_mov['distance_m'] > 3) &
                (df_mov['direction'] > 0.3)
            ].copy()

            # Classify movement shapes
            df_runs_c = classify_runs_shape(
                df_runs,
                dict_traj,
                linearity_threshold = LINEARITY_THRESHOLD_M,
                diagonal_angle_deg  = DIAGONAL_ANGLE_DEG,
            )

            # Save per-match CSV
            out_path = os.path.join(OUTPUT_DIR, f'runs_classified_{match_id}.csv')
            df_runs_c.to_csv(out_path, index=False)

            # Print summary
            total  = len(df_runs_c)
            counts = df_runs_c['shape'].value_counts()
            fb     = (df_runs_c['shape_method'] == 'fallback').sum()
            print(f'  -> {total} runs classified for {match_id}')
            for shape, cnt in counts.items():
                print(f'     {shape:<12}: {cnt:>4}  ({cnt / total * 100:.1f}%)')
            if fb:
                print(f'     [fallback used for {fb} runs with no raw trajectory]')

            all_runs_classified.append(df_runs_c)

        except Exception as e:
            print(f'\n[ERROR] {match_id} - {e}')
            traceback.print_exc()

    # Concatenate and save global file
    if all_runs_classified:
        df_all = pd.concat(all_runs_classified, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, 'all_runs_classified.csv'), index=False)

        print(f'\n{"=" * 60}')
        print('GLOBAL SUMMARY')
        print(f'{"=" * 60}')
        total = len(df_all)
        print(f'Total runs: {total}')
        for shape, cnt in df_all['shape'].value_counts().items():
            print(f'  {shape:<12}: {cnt:>5}  ({cnt / total * 100:.1f}%)')
        print(f'\nAll files saved in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()