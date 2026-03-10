"""
Discrete_efforts_Ilana.py
--------------------------
Processing module for DFL tracking data (single match).
Contains all helper functions and the main process_match() function
designed to be imported and called in a loop by Loop_Ilana.py.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import floodlight.io.dfl as dfl
from floodlight.models.kinematics import DistanceModel, VelocityModel
from floodlight.transforms.filter import butterworth_lowpass


# ============================================================================
# Constants
# ============================================================================

SPEED_THRESHOLDS = {
    'walking':  7,   # km/h — below this: walking
    'jogging': 14,   # km/h — below this: jogging
    'running': 21,   # km/h — below this: running / above: sprinting
}


# ============================================================================
# Utility functions
# ============================================================================

def sec_to_minute(seconds):
    """Convert seconds to a MM:SS string."""
    minutes = int(seconds) // 60
    secs    = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def frame_to_minute(frame, frate=25):
    """Convert a frame number to a MM:SS string (default 25 fps)."""
    return sec_to_minute(frame / frate)


def classify_speed_category(peak_speed_kmh, thresholds=None):
    """
    Classify a peak speed value into a movement category.

    Parameters
    ----------
    peak_speed_kmh : float
    thresholds     : dict with keys 'walking', 'jogging', 'running'
                     (falls back to SPEED_THRESHOLDS if None)

    Returns
    -------
    str : 'walking' | 'jogging' | 'running' | 'sprinting'
    """
    if thresholds is None:
        thresholds = SPEED_THRESHOLDS
    if peak_speed_kmh < thresholds['walking']:
        return 'walking'
    elif peak_speed_kmh < thresholds['jogging']:
        return 'jogging'
    elif peak_speed_kmh < thresholds['running']:
        return 'running'
    else:
        return 'sprinting'


# ============================================================================
# Discrete movement extraction
# ============================================================================

def extract_movements_for_player(player_id, velocities, tracking_data, possession,
                                  min_valley_distance=10,valley_depth_abs=0.05,
                                  valley_depth_rel=0.01): #for valley_depth_abs, valley_depth_rel still need to find good values through experimentation
    """
    Segment a player's speed signal into discrete movements via valley detection.

    Valleys (local minima) in the speed signal define segment boundaries.
    Each valley-to-valley interval represents one discrete movement.
    The start of each segment is the valley frame itself
    (origin = point of acceleration onset).

    Parameters
    ----------
    player_id            : int   — column index in `velocities`
    velocities           : np.ndarray (n_frames, n_players) — speed in m/s
    tracking_data        : np.ndarray (n_frames, n_players*2) — x,y interleaved
    possession           : np.ndarray (n_frames,) — 1=Home, 2=Away
    min_valley_distance  : int   — minimum number of frames between two valleys
                           At 25 fps, 10 frames = 0.4 s

    Returns
    -------
    list[pd.DataFrame] : one DataFrame per discrete movement
    """
    player_velocity = velocities[:, player_id]      # m/s

    # Replace NaN with 0 for valley detection (NaN = player not tracked = stationary)
    vel_for_peaks = np.where(np.isnan(player_velocity), 0.0, player_velocity)

    # Valleys = peaks of the negated signal
    raw_valleys, _ = find_peaks(-vel_for_peaks, distance=min_valley_distance)

    # ── Filter out shallow valleys ────────────────────────────────────────────
    significant_valleys = []
    for v in raw_valleys:
        valley_speed = vel_for_peaks[v]
        left  = max(0, v - min_valley_distance)
        right = min(len(vel_for_peaks), v + min_valley_distance)
        local_peak = vel_for_peaks[left:right].max()

        abs_drop = local_peak - valley_speed
        rel_drop = abs_drop / local_peak if local_peak > 0 else 0.0

        if abs_drop >= valley_depth_abs and rel_drop >= valley_depth_rel:
            significant_valleys.append(v)

    valleys = np.array(significant_valleys, dtype=int)

    boundaries = np.unique(np.concatenate([[0], valleys, [len(player_velocity) - 1]]))

    movements = []
    mov_idx   = 0
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end   = int(boundaries[i + 1])

        if end - start < 3:
            continue

        frames = np.arange(start, end)
        v      = player_velocity[start:end]

        # Skip segments where the player is entirely untracked
        if np.all(np.isnan(v)):
            continue

        x = tracking_data[start:end, ::2][:, player_id]
        y = tracking_data[start:end, 1::2][:, player_id]
        p = possession[start:end]

        movements.append(pd.DataFrame({
            'frame':       frames,
            'x':           x,
            'y':           y,
            'velocity':    v,
            'possession':  p,
            'movement_id': mov_idx,
        }))
        mov_idx += 1

    return movements


def extract_discrete_movements_all_players(velocities, tracking_data, possession,
                                            min_valley_distance=10,valley_depth_abs=0.1,
                                  valley_depth_rel=0.01):
    """
    Extract discrete movements for every player in a tracking data array.

    Parameters
    ----------
    velocities           : np.ndarray (n_frames, n_players)
    tracking_data        : np.ndarray (n_frames, n_players*2)
    possession           : np.ndarray (n_frames,)
    min_valley_distance  : int

    Returns
    -------
    dict {player_id: list[pd.DataFrame]}
    """
    n_players = velocities.shape[1]
    return {
        player_id: extract_movements_for_player(
            player_id, velocities, tracking_data, possession, min_valley_distance)
        for player_id in range(n_players)
    }


# ============================================================================
# Movement summary DataFrame
# ============================================================================

def summarize_movements_to_dataframe(movements_dict, teamsheet, framerate=25):
    """
    Convert discrete movement data into a summary DataFrame (one row per movement).

    Output columns:
        start_frame, end_frame, peak_frame,
        xID, player, position, team, possession, jID,
        x_start, y_start, x_end, y_end, x_peak, y_peak,
        peak_speed_kmh, speed_category,
        avg_velocity_kmh, distance_m, duration_s

    Parameters
    ----------
    movements_dict : dict {xID: list[pd.DataFrame]}
    teamsheet      : pd.DataFrame — must contain columns
                     ['xID', 'player', 'position', 'team', 'jID']
    framerate      : int

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for xID, movements in movements_dict.items():
        player_info = teamsheet.loc[teamsheet['xID'] == xID]
        if player_info.empty:
            continue

        player_name = player_info['player'].values[0]
        position    = player_info['position'].values[0]
        team_name   = player_info['team'].values[0]
        jersey_id   = player_info['jID'].values[0]

        for df in movements:
            if df.empty or df['velocity'].isna().all():
                continue

            v_ms     = df['velocity'].values
            valid_v  = np.where(np.isnan(v_ms), -np.inf, v_ms)
            peak_idx = int(np.argmax(valid_v))
            peak_speed_kmh = (float(v_ms[peak_idx]) * 3.6
                              if not np.isnan(v_ms[peak_idx]) else 0.0)

            # Total distance covered during the movement
            deltas     = df[['x', 'y']].diff().dropna()
            distance_m = np.sqrt((deltas ** 2).sum(axis=1)).sum()

            rows.append({
                'start_frame':      df['frame'].iloc[0],
                'end_frame':        df['frame'].iloc[-1],
                'peak_frame':       df['frame'].iloc[peak_idx],
                'xID':              xID,
                'player':           player_name,
                'position':         position,
                'team':             team_name,
                'possession':       "Home" if df['possession'].iloc[0] == 1 else "Away",
                'jID':              jersey_id,
                'x_start':          df['x'].iloc[0],
                'y_start':          df['y'].iloc[0],
                'x_end':            df['x'].iloc[-1],
                'y_end':            df['y'].iloc[-1],
                'x_peak':           df['x'].iloc[peak_idx],   # position at peak speed
                'y_peak':           df['y'].iloc[peak_idx],
                'peak_speed_kmh':   peak_speed_kmh,
                'speed_category':   classify_speed_category(peak_speed_kmh),
                'avg_velocity_kmh': np.nanmean(v_ms) * 3.6,
                'distance_m':       distance_m,
                'duration_s':       len(df) / framerate,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Main pipeline function
# ============================================================================

def process_match(match_id, DATA_DIR, dict_direction,
                  butterworth_Wn=0.5, butterworth_order=1,
                  min_valley_distance=10, framerate=25,
                  speed_thresholds=None,valley_depth_abs=0.1,
                                  valley_depth_rel=0.01):
    """
    Full processing pipeline for a single match:
    loading -> filtering -> discrete movements -> distances -> playing direction.

    Parameters
    ----------
    match_id            : str   — match identifier (used to filter filenames)
    DATA_DIR            : str   — directory containing DFL data files
    dict_direction      : dict  — playing directions loaded from JSON
                          format: {match_id: {"Home": {"firstHalf": "left_to_right", ...}}}
    butterworth_Wn      : float — cutoff frequency for the low-pass filter
    butterworth_order   : int   — Butterworth filter order
    min_valley_distance : int   — minimum frames between two valleys
    framerate           : int   — acquisition frame rate (fps)
    speed_thresholds    : dict  — speed thresholds (falls back to SPEED_THRESHOLDS if None)

    Returns
    -------
    df_movements      : pd.DataFrame  — one discrete movement per row
    df_distances      : pd.DataFrame  — total distance covered per player
    dict_trajectories : dict          — raw trajectories per team/player
    """
    if speed_thresholds is None:
        speed_thresholds = SPEED_THRESHOLDS

    # ------------------------------------------------------------------
    # 1. Locate data files
    # ------------------------------------------------------------------
    files = os.listdir(DATA_DIR)

    def find_file(keyword):
        matches = [x for x in files if match_id in x and keyword in x]
        if not matches:
            raise FileNotFoundError(
                f"No file containing '{match_id}' and '{keyword}' found in {DATA_DIR}")
        return os.path.join(DATA_DIR, matches[0])

    path_positions = find_file('positions')
    path_events    = find_file('events')
    path_info      = find_file('matchinformation')

    # ------------------------------------------------------------------
    # 2. Load tracking and event data
    # ------------------------------------------------------------------
    xy, possession, ballstatus, teamsheets, pitch = dfl.read_position_data_xml(
        path_positions, path_info,
        teamsheet_home=None, teamsheet_away=None,
    )
    _events, _teamsheets_ev, _pitch = dfl.read_event_data_xml(path_events, path_info)
    pitch.sport = "football"

    # Convenience dict for iterating over halves and teams
    xy_all = {
        'firstHalf':  {'Home': xy['firstHalf']['Home'],  'Away': xy['firstHalf']['Away']},
        'secondHalf': {'Home': xy['secondHalf']['Home'], 'Away': xy['secondHalf']['Away']},
    }

    # ------------------------------------------------------------------
    # 3. Compute filtered velocities
    # ------------------------------------------------------------------
    velocity_filtered_dict = {half: {} for half in ['firstHalf', 'secondHalf']}

    for half, teams in xy_all.items():
        for team, xy_data in teams.items():
            xy_f = butterworth_lowpass(xy_data, remove_short_seqs=True,
                                       Wn=butterworth_Wn, order=butterworth_order)
            vm_f = VelocityModel()
            vm_f.fit(xy_f)
            velocity_filtered_dict[half][team] = vm_f.velocity().property

    # ------------------------------------------------------------------
    # 4. Extract discrete movements
    # ------------------------------------------------------------------
    ls_movement_dfs   = []
    dict_trajectories = {}

    for team in ['Home', 'Away']:
        ls_movements_team = []

        for half in ['firstHalf', 'secondHalf']:
            # add_xIDs() is required because read_event_data_xml does not call it,
            # so the xID column would otherwise be missing from the teamsheet.
            teamsheets[team].add_xIDs()

            xy_f = butterworth_lowpass(
                xy[half][team], remove_short_seqs=True,
                Wn=butterworth_Wn, order=butterworth_order,
            ).xy

            movements_dict = extract_discrete_movements_all_players(
                velocity_filtered_dict[half][team],
                xy_f,
                possession[half],
                min_valley_distance=min_valley_distance,
            )

            summary_df = summarize_movements_to_dataframe(
                movements_dict, teamsheets[team].teamsheet, framerate=framerate)

            summary_df['half']     = half
            summary_df['location'] = team

            # Human-readable timecodes
            # Second half frames are offset by 67 500 frames (= 45 min at 25 fps)
            summary_df['start_timecode'] = summary_df.apply(
                lambda row: frame_to_minute(row['start_frame'], framerate)
                if row['half'] == 'firstHalf'
                else frame_to_minute(row['start_frame'] + 67500, framerate),
                axis=1,
            )
            summary_df['end_timecode'] = summary_df.apply(
                lambda row: frame_to_minute(row['end_frame'], framerate)
                if row['half'] == 'firstHalf'
                else frame_to_minute(row['end_frame'] + 67500, framerate),
                axis=1,
            )

            ls_movement_dfs.append(summary_df)
            ls_movements_team.append(movements_dict)

        # Merge both halves into a single trajectory dict per team
        dict_trajectories[team] = {
            k: ls_movements_team[0][k] + ls_movements_team[1][k]
            for k in ls_movements_team[0]
        }

    df_movements = pd.concat(ls_movement_dfs, ignore_index=True)
    df_movements = df_movements[df_movements['distance_m'] > 0]

    # ------------------------------------------------------------------
    # 5. Total distance covered per player
    # ------------------------------------------------------------------
    ls_dfs_distance = []

    for half, teams in xy_all.items():
        for team, xy_data in teams.items():
            xy_f = butterworth_lowpass(xy_data, remove_short_seqs=True,
                                       Wn=butterworth_Wn, order=butterworth_order)
            dm   = DistanceModel()
            dm.fit(xy_f)

            # .copy() avoids SettingWithCopyWarning when adding columns to a slice
            df_players = teamsheets[team].teamsheet.copy()
            df_players['distance_covered'] = np.nansum(
                dm.distance_covered().property, axis=0)
            df_players['location'] = team
            df_players['half']     = half
            ls_dfs_distance.append(df_players)

    df_distances = (
        pd.concat(ls_dfs_distance)
        .groupby(['location', 'team', 'player'])
        .agg(
            total_distance_m=pd.NamedAgg(column='distance_covered', aggfunc='sum'),
            position=pd.NamedAgg(column='position', aggfunc='first'),
            jID=pd.NamedAgg(column='jID', aggfunc='first'),
        )
    )

    # ------------------------------------------------------------------
    # 6. Playing direction
    # ------------------------------------------------------------------
    if match_id not in dict_direction:
        print(f"[WARN] '{match_id}' not found in direction JSON — "
              f"'direction' column will be NaN.")

    dx = df_movements["x_end"] - df_movements["x_start"]
    dy = df_movements["y_end"] - df_movements["y_start"]

    # Normalise to [-1, 1] using the L1 norm of the displacement vector
    base_direction = dx.div(dx.abs().add(dy.abs())).fillna(0)

    # Map home team direction per half, then invert for the away team
    home_by_half     = dict_direction.get(match_id, {}).get("Home", {})
    home_direction   = df_movements["half"].map(home_by_half)
    player_direction = home_direction.where(
        df_movements["location"].ne("Away"),
        home_direction.map({
            "left_to_right": "right_to_left",
            "right_to_left": "left_to_right",
        }),
    )
    multiplier = player_direction.map(
        {"left_to_right": 1, "right_to_left": -1}).fillna(1)

    df_movements["direction"] = base_direction * multiplier
    df_movements["attack_sign"] = multiplier

    # Add match identifier — useful when concatenating results across matches
    df_movements["match_id"] = match_id

    print(f"[OK] {match_id} — {len(df_movements)} discrete movements extracted.")
    return df_movements, df_distances, dict_trajectories