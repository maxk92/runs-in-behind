"""
Discretisation pipeline
=======================
Optimised version — pure NumPy possession smoothing, vectorised valley
filtering, NumPy interval blackout filter.
"""

import os
import xml.etree.ElementTree as ET

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
    'walking':  7,
    'jogging': 14,
    'running': 21,
}

SETPIECE_OFFSETS = {
    'ThrowIn':    2.0,
    'FreeKick':   2.0,
    'GoalKick':   2.0,
    'KickOff':    6.0,
    'CornerKick': 2.0,
}

RECIPIENT_DELAY_S = 0.0

# --- Pre-run window -----------------------------------------------------
# Length (seconds) of the window looked at BEFORE a movement's start frame
# to compute "pre_run_mean_vel_kmh" / "pre_run_peak_vel_kmh".
PRE_RUN_WINDOW_S = 2.0

# --- Shot-follow-up flag -------------------------------------------------
# A movement is flagged "shot_within_Ns" if the SAME team takes a shot within
# this many seconds after the movement's end frame (same half).
SHOT_WINDOW_S = 10.0

# --- Pitch zones (D/M/A x 1-5) -------------------------------------------
# Standard DFL coordinate system:
#   x in [-52.5, 52.5]  (negative = left goal, positive = right goal)
#   y in [-34,   34  ]  (negative = bottom touchline)
#
# Horizontal thirds (3 zones):
#   Defensive third   x < -17.5
#   Middle third     -17.5 <= x <= 17.5
#   Attacking third   x >  17.5
#
# Vertical channels (5 zones) based on the penalty-box width (40.32 m -> +-20.16)
# and the goal width (7.32 m -> +-3.66):
#   corner -> box line -> near post -> far post -> box line -> corner
#
# Combined: 3 thirds x 5 channels = 15 zones, e.g. "D1", "M3", "A5"
#   Third prefix: D = defensive, M = middle, A = attacking (from the
#   perspective of the team performing the movement, once normalised for
#   playing direction -- see assign_zone()).
#   Channel suffix 1-5, from bottom (most negative y) to top (most positive y).
THIRD_BOUNDARIES_X   = [-17.5, 17.5]                  # two cut-points -> three thirds
CHANNEL_BOUNDARIES_Y = [-20.16, -3.66, 3.66, 20.16]   # four cut-points -> five channels

# Canonical direction used ONLY for zone assignment (D/M/A x 1-5).
# Every movement is mirrored on x/y so that, for zoning purposes, everyone
# "plays" in this direction -- the raw x_start/x_mid/x_end/y_* columns are
# NOT touched, only the zone label changes.
ZONE_TARGET_DIRECTION = "right_to_left"


# ============================================================================
# Utility functions
# ============================================================================

def sec_to_minute(seconds):
    minutes = int(seconds) // 60
    secs    = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def frame_to_minute(frame, frate=25):
    return sec_to_minute(frame / frate)


def classify_speed_category(peak_speed_kmh, thresholds=None):
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


def get_play_direction(dict_direction, match_id, team, half):
    """
    Return 'left_to_right' or 'right_to_left' -- the direction `team`
    ('Home' or 'Away') is attacking during `half`.

    Only "Home" is stored in dict_direction; "Away" is derived as the
    opposite direction in the same half. Returns None if unknown (missing
    match_id, missing half, or unrecognised team) so callers can skip
    zone-normalisation rather than guess.
    """
    opposite = {'left_to_right': 'right_to_left', 'right_to_left': 'left_to_right'}
    home_dir = dict_direction.get(match_id, {}).get('Home', {}).get(half)
    if home_dir is None:
        return None
    if team == 'Home':
        return home_dir
    if team == 'Away':
        return opposite.get(home_dir)
    return None


def assign_zone(x, y, direction=None, target=ZONE_TARGET_DIRECTION,
                third_boundaries_x=THIRD_BOUNDARIES_X,
                channel_boundaries_y=CHANNEL_BOUNDARIES_Y):
    """
    Return a pitch-zone label such as 'D1', 'M3', 'A5' for a given (x, y)
    position.

    Methodology
    -----------
    1. Direction normalisation: if `direction` is given and differs from
       `target`, x and y are mirrored (negated) first, so that every
       movement is zoned as if the team were playing in `target` direction
       (default: 'right_to_left', i.e. attacking towards negative x).
       This only affects which zone label is returned -- it never mutates
       the caller's raw x_start/x_mid/x_end/y_* coordinates.
    2. Horizontal third: x is compared against `third_boundaries_x`
       ([-17.5, 17.5] by default) to decide Defensive / Middle / Attacking.
    3. Vertical channel: y is located among `channel_boundaries_y`
       ([-20.16, -3.66, 3.66, 20.16] by default) via np.searchsorted,
       giving a channel index from 1 (most negative y) to 5 (most positive y).
    4. The label is the concatenation of the third letter and the channel
       number, e.g. "A5" = attacking third, top channel.

    If (x, y) is NaN, returns NaN (no zone can be assigned).
    """
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)) \
            or (isinstance(y, float) and np.isnan(y)):
        return np.nan

    if direction is not None and direction == target:
        x = -x
        y = -y
    # Horizontal third
    if x < THIRD_BOUNDARIES_X[0]:
        third = "D"
    elif x <= THIRD_BOUNDARIES_X[1]:
        third = "M"
    else:
        third = "A"
    # Vertical channel (1 = most negative y)
    if y < CHANNEL_BOUNDARIES_Y[0]:
        channel = "R"
    elif y < CHANNEL_BOUNDARIES_Y[1]:
        channel = "HR"
    elif y < CHANNEL_BOUNDARIES_Y[2]:
        channel = "C"
    elif y < CHANNEL_BOUNDARIES_Y[3]:
        channel = "HL"
    else:
        channel = "L"
    return third, channel


# ============================================================================
# Possession smoothing — pure NumPy, no intermediate DataFrame allocation
# ============================================================================

def _smooth_half(poss_arr: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth a possession array for ONE half (NumPy only).
    Returns: (poss_smoothed, phase_ids, phase_durations)
    Phase IDs are local to this half (start at 1).
    """
    poss = poss_arr.copy()
    n = len(poss)

    # Remove short phases (sequential — data dependency prevents vectorisation)
    i = 0
    while i < n:
        current_val = poss[i]
        j = i + 1
        while j < n and poss[j] == current_val:
            j += 1
        if (j - i) < cutoff and i > 0:
            poss[i:j] = poss[i - 1]
        i = j

    # Phase IDs and durations — vectorised
    change_pos = np.where(np.diff(poss) != 0)[0] + 1   # start indices of new phases
    boundaries = np.concatenate([[0], change_pos, [n]])  # (n_phases + 1,)
    phase_lengths = np.diff(boundaries)                  # duration of each phase

    new_id  = np.empty(n, dtype=np.int64)
    dur_arr = np.empty(n, dtype=np.int64)

    for k, (s, length) in enumerate(zip(boundaries[:-1], phase_lengths)):
        new_id[s:s + length]  = k + 1
        dur_arr[s:s + length] = length

    return poss, new_id, dur_arr


def smooth_possession(possession_array, half_array, cutoff=15):
    """
    Smooth possession over a full match (two halves).
    No intermediate DataFrame is created, avoiding repeated reallocation per group.
    """
    possession_array = np.asarray(possession_array).flatten().astype(np.int64)
    half_array       = np.asarray(half_array).flatten().astype(np.int64)
    n = len(possession_array)

    smoothed = np.empty(n, dtype=np.int64)
    new_id   = np.empty(n, dtype=np.int64)
    dur_arr  = np.empty(n, dtype=np.int64)

    global_id_offset = 0

    for half_val in np.unique(half_array):
        mask = half_array == half_val
        idx  = np.where(mask)[0]

        p_s, p_id, p_dur = _smooth_half(possession_array[idx], cutoff)

        smoothed[idx] = p_s
        new_id[idx]   = p_id + global_id_offset
        dur_arr[idx]  = p_dur

        global_id_offset += p_id.max()

    return smoothed, new_id, dur_arr


# ============================================================================
# Layer 2 — Metabolic power sub-segmentation
# Split candidates are pre-filtered with vectorised operations;
# only the `long_enough` criterion (depends on the last split) needs a loop.
# ============================================================================

def compute_metabolic_power(velocity_ms, framerate=25):
    dt    = 1.0 / framerate
    vel   = np.where(np.isnan(velocity_ms), 0.0, velocity_ms)
    accel = np.gradient(vel, dt)
    power = vel * accel

    g  = 9.81
    kt = 1.29
    i  = accel / g
    i_clipped = np.clip(i, -0.5, 0.5)
    ec = (155.4 * i_clipped**5 - 30.4 * i_clipped**4 - 43.3 * i_clipped**3
          + 46.3 * i_clipped**2 + 19.5 * i_clipped + 3.6) * kt

    return power, accel, ec


def _first_inflection_after(extremum_idx, dp, search_limit):
    """
    Return the index of the first local extremum of `power` strictly after
    `extremum_idx` and before `search_limit`.

    power = v * v'  (velocity times acceleration)
    dp    = d(power)/dt = (v')^2 + v * v''

    A sign change of dp  (dp[k-1] * dp[k] < 0)  marks a local extremum of
    power — i.e. a point where power switches from increasing to decreasing
    or vice versa.  This is NOT an inflection point of power (which would
    require d²p = 0) and NOT an inflection point of velocity (which would
    require v'' = 0).

    Heuristic rationale: cutting at a power extremum lands on a natural
    "shoulder" of the effort curve, avoiding cuts right in the middle of an
    acceleration or deceleration phase.

    If no such extremum is found before `search_limit`, the original
    `extremum_idx` is returned as fallback.
    """
    for k in range(extremum_idx + 1, search_limit):
        if dp[k - 1] * dp[k] < 0:
            return k
    return extremum_idx   # fallback: no inflection found → keep original cut


def split_segment_by_power(segment_df, framerate=25,
                            power_threshold=6.0, min_segment_frames=10):
    v             = segment_df['velocity'].values
    power, _, ec  = compute_metabolic_power(v, framerate=framerate)
    dp            = np.gradient(power)
   

    n = len(power)

    # Pre-filter candidate split positions vectorially.
    # `is_brutal`  : large instantaneous change in power (kept as-is).
    # `sign_change`: local extremum of the power curve (zero-crossing of power).
    #                These are now PROMOTED to the first inflection point of dp
    #                that follows the extremum, so the cut lands on a natural
    #                "shoulder" of the curve rather than right at the peak.
    inner = np.arange(min_segment_frames, n - min_segment_frames)
    is_brutal   = np.abs(dp[inner]) > power_threshold
    sign_change = power[inner - 1] * power[inner] < 0

    # For each extremum candidate, find the first inflection point of dp after it.
    # Inflection of dp  ≡  sign change of d2p  ≡  dp[k-1]*dp[k] < 0
    extremum_positions = inner[sign_change]
    inflection_map = {}   # extremum_idx → inflection_idx
    search_cap = n - min_segment_frames   # never cut too close to the end
    for ext in extremum_positions:
        inflection_map[int(ext)] = _first_inflection_after(int(ext), dp, search_cap)

    # Build the final candidate list:
    # - brutal-change positions are kept at their original index
    # - extremum positions are replaced by their downstream inflection point
    brutal_positions = set(inner[is_brutal].tolist())
    raw_candidates   = sorted(
        brutal_positions
        | set(inflection_map.values())
    )

    split_indices = [0]
    for i in raw_candidates:
        if (i - split_indices[-1]) >= min_segment_frames:
            split_indices.append(int(i))
    split_indices.append(len(segment_df))

    sub_segments = []
    for k in range(len(split_indices) - 1):
        seg = segment_df.iloc[split_indices[k]:split_indices[k + 1]].copy()
        if len(seg) >= 3:
            seg['ec'] = ec[split_indices[k]:split_indices[k + 1]]
            sub_segments.append(seg)
    return sub_segments


# ============================================================================
# Layer 1 — Velocity valley segmentation
# Valley filtering is fully vectorised (batch operations, reduced Python overhead).
# ============================================================================

def _compute_pre_run_window(player_velocity, start_frame, pre_run_window_s, framerate):
    """
    Compute the "pre-run window" kinematic indicators: the mean and peak
    speed of the player during the `pre_run_window_s` seconds immediately
    BEFORE `start_frame` (i.e. before the movement begins).

    Methodology
    -----------
    1. Convert `pre_run_window_s` to a number of frames
       (pre_frames = round(pre_run_window_s * framerate)).
    2. Take the window [start_frame - pre_frames, start_frame) on the
       player's FULL velocity time series for the half (clipped to 0 so it
       never looks before the start of the half's tracking data).
    3. Drop NaN samples (gaps in tracking).
    4. pre_run_mean_vel_kmh / pre_run_peak_vel_kmh = mean / max of the
       remaining valid samples, converted from m/s to km/h (*3.6) to stay
       consistent with the other speed columns in this pipeline.
    5. pre_run_window_s_used = the window duration actually available
       (may be shorter than the requested `pre_run_window_s` if the
       movement starts near the beginning of the half).

    Returns (pre_run_mean_vel_kmh, pre_run_peak_vel_kmh, pre_run_window_s_used),
    all NaN if the window is empty/degenerate or pre_run_window_s is None/<=0.
    """
    if pre_run_window_s is None or pre_run_window_s <= 0:
        return np.nan, np.nan, np.nan

    pre_frames = int(round(pre_run_window_s * framerate))
    pre_start  = max(0, start_frame - pre_frames)
    pre_end    = start_frame   # window ends right where the movement starts (exclusive)

    if pre_end <= pre_start:
        return np.nan, np.nan, np.nan

    v_pre = player_velocity[pre_start:pre_end]
    v_pre_valid = v_pre[~np.isnan(v_pre)]

    if v_pre_valid.size == 0:
        return np.nan, np.nan, round((pre_end - pre_start) / framerate, 2)

    pre_mean_kmh = float(np.mean(v_pre_valid)) * 3.6
    pre_peak_kmh = float(np.max(v_pre_valid)) * 3.6
    window_used_s = round((pre_end - pre_start) / framerate, 2)
    return pre_mean_kmh, pre_peak_kmh, window_used_s


def extract_movements_for_player(
        player_id, velocities, tracking_data, possession,
        min_valley_distance=15,
        valley_depth_abs=0.15,
        valley_depth_rel=0.05,
        power_threshold=6.0,
        min_segment_frames=10,
        framerate=25,
        pre_run_window_s=PRE_RUN_WINDOW_S,
):
    player_velocity = velocities[:, player_id]
    vel_for_peaks   = np.where(np.isnan(player_velocity), 0.0, player_velocity)

    raw_valleys, _ = find_peaks(-vel_for_peaks, distance=min_valley_distance)

    # Vectorised valley filtering — batch operations instead of scalar Python loop.
    if len(raw_valleys) > 0:
        valley_speeds = vel_for_peaks[raw_valleys]
        lefts  = np.maximum(0, raw_valleys - min_valley_distance)
        rights = np.minimum(len(vel_for_peaks), raw_valleys + min_valley_distance)

        # local_peak: .max() over variable-length slices is not vectorisable;
        # isolated in a list-comp rather than a full loop.
        local_peaks = np.fromiter(
            (vel_for_peaks[l:r].max() for l, r in zip(lefts, rights)),
            dtype=float, count=len(raw_valleys)
        )
        abs_drops = local_peaks - valley_speeds
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_drops = np.where(local_peaks > 0, abs_drops / local_peaks, 0.0)
        keep      = (abs_drops >= valley_depth_abs) & (rel_drops >= valley_depth_rel)
        valleys   = raw_valleys[keep]
    else:
        valleys = np.array([], dtype=int)

    boundaries = np.unique(
        np.concatenate([[0], valleys, [len(player_velocity) - 1]])
    )

    coarse_segments = []
    for i in range(len(boundaries) - 1):
        start  = int(boundaries[i])
        end    = int(boundaries[i + 1])
        if end - start < 3:
            continue
        frames = np.arange(start, end)
        v      = player_velocity[start:end]
        if np.all(np.isnan(v)):
            continue
        x = tracking_data[start:end, ::2][:, player_id]
        y = tracking_data[start:end, 1::2][:, player_id]
        p = possession[start:end]
        coarse_segments.append(pd.DataFrame({
            'frame': frames, 'x': x, 'y': y, 'velocity': v, 'possession': p,
        }))

    movements = []
    mov_idx   = 0
    for seg in coarse_segments:
        sub = split_segment_by_power(
            seg, framerate=framerate,
            power_threshold=power_threshold,
            min_segment_frames=min_segment_frames,
        )
        for s in sub:
            s = s.copy()
            s['movement_id'] = mov_idx

            # Pre-run window indicators, computed against the player's FULL
            # velocity series for the half (not just this sub-segment), so
            # the look-back can reach into whatever preceded the movement.
            pre_mean_kmh, pre_peak_kmh, pre_win_s = _compute_pre_run_window(
                player_velocity, int(s['frame'].iloc[0]),
                pre_run_window_s, framerate,
            )
            # Stored as constant columns (one value per movement, broadcast
            # over all its frames) so they survive through to
            # summarize_movements_to_dataframe the same way 'ec' does.
            s['pre_run_mean_vel_kmh'] = pre_mean_kmh
            s['pre_run_peak_vel_kmh'] = pre_peak_kmh
            s['pre_run_window_s']     = pre_win_s

            movements.append(s)
            mov_idx += 1

    return movements


def extract_discrete_movements_all_players(
        velocities, tracking_data, possession,
        min_valley_distance=15,
        valley_depth_abs=0.15,
        valley_depth_rel=0.05,
        power_threshold=6.0,
        min_segment_frames=10,
        framerate=25,
        pre_run_window_s=PRE_RUN_WINDOW_S,
):
    n_players = velocities.shape[1]
    return {
        player_id: extract_movements_for_player(
            player_id, velocities, tracking_data, possession,
            min_valley_distance=min_valley_distance,
            valley_depth_abs=valley_depth_abs,
            valley_depth_rel=valley_depth_rel,
            power_threshold=power_threshold,
            min_segment_frames=min_segment_frames,
            framerate=framerate,
            pre_run_window_s=pre_run_window_s,
        )
        for player_id in range(n_players)
    }


# ============================================================================
# Movement summary DataFrame
# Teamsheet pre-indexed by xID for O(1) lookup; array accesses are shared.
# ============================================================================

def summarize_movements_to_dataframe(movements_dict, teamsheet, framerate=25):
    ts_indexed = teamsheet.set_index('xID')

    rows = []
    dt   = 1.0 / framerate

    for xID, movements in movements_dict.items():
        if xID not in ts_indexed.index:
            continue
        info = ts_indexed.loc[xID]
        player_name = info['player']
        position    = info['position']
        team_name   = info['team']
        jersey_id   = info['jID']

        for df in movements:
            if df.empty or df['velocity'].isna().all():
                continue

            v_ms     = df['velocity'].values
            x_vals   = df['x'].values
            y_vals   = df['y'].values
            frames   = df['frame'].values

            # Peak speed — avoids np.argmax on a rebuilt masked array
            valid_v  = np.where(np.isnan(v_ms), -np.inf, v_ms)
            peak_idx = int(np.argmax(valid_v))
            peak_speed_kmh = (float(v_ms[peak_idx]) * 3.6
                              if not np.isnan(v_ms[peak_idx]) else 0.0)

            dx = np.diff(x_vals)
            dy = np.diff(y_vals)
            distance_m = float(np.sqrt(dx * dx + dy * dy).sum())

            # Midpoint position of the movement.
            # Methodology: the MEDIAN (not the mean) of x and y over all
            # valid frames of the movement, matching the convention used in
            # the manual-annotation enrichment tool. The median is robust
            # to a movement that lingers briefly at one extremity (e.g. a
            # player decelerating hard at the end), which would bias a
            # simple mean toward that extremity.
            x_mid = float(np.nanmedian(x_vals))
            y_mid = float(np.nanmedian(y_vals))

            # Pre-run window indicators — one constant value per movement,
            # attached upstream in extract_movements_for_player() via
            # _compute_pre_run_window(). Read the first row since the value
            # is broadcast identically across the whole movement.
            if 'pre_run_mean_vel_kmh' in df.columns:
                pre_run_mean_vel_kmh = float(df['pre_run_mean_vel_kmh'].iloc[0])
                pre_run_peak_vel_kmh = float(df['pre_run_peak_vel_kmh'].iloc[0])
                pre_run_window_s_used = df['pre_run_window_s'].iloc[0]
            else:
                pre_run_mean_vel_kmh = float('nan')
                pre_run_peak_vel_kmh = float('nan')
                pre_run_window_s_used = float('nan')

            if 'ec' in df.columns:
                ec_values = df['ec'].values
                v_safe    = np.where(np.isnan(v_ms), 0.0, v_ms)
                total_energy_cost = float(np.nansum(np.maximum(ec_values, 0) * v_safe * dt))
            else:
                total_energy_cost = float('nan')

            # Possession by majority vote over raw frames
            poss_vals  = df['possession'].values
            home_ratio = float((poss_vals == 1).mean())
            possession_label     = "Home" if home_ratio >= 0.5 else "Away"
            possession_contested = 0.30 < home_ratio < 0.70

            rows.append({
                'start_frame':            int(frames[0]),
                'end_frame':              int(frames[-1]),
                'peak_frame':             int(frames[peak_idx]),
                'xID':                    xID,
                'player':                 player_name,
                'position':               position,
                'team':                   team_name,
                'possession':             possession_label,
                'possession_ratio':       round(home_ratio, 3),
                'possession_contested':   possession_contested,
                'jID':                    jersey_id,
                'x_start':             float(x_vals[0]),
                'y_start':             float(y_vals[0]),
                'x_mid':               x_mid,
                'y_mid':               y_mid,
                'x_end':               float(x_vals[-1]),
                'y_end':               float(y_vals[-1]),
                'x_peak':              float(x_vals[peak_idx]),
                'y_peak':              float(y_vals[peak_idx]),
                'peak_speed_kmh':      peak_speed_kmh,
                'speed_category':      classify_speed_category(peak_speed_kmh),
                'avg_velocity_kmh':    float(np.nanmean(v_ms)) * 3.6,
                'distance_m':          distance_m,
                'duration_s':          len(df) / framerate,
                'total_energy_cost_J': total_energy_cost,
                'pre_run_mean_vel_kmh': pre_run_mean_vel_kmh,
                'pre_run_peak_vel_kmh': pre_run_peak_vel_kmh,
                'pre_run_window_s':     pre_run_window_s_used,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Centralised time anchors — events + matchinfo XML parsed only once
# ============================================================================

def _build_time_anchors(path_events: str, path_info: str,
                        framerate: int = 25,
                        info_root=None, ev_root=None) -> dict:
    """
    Parse events + matchinfo XML once.
    Returns kickoff_utc, second_half_start_utc, kickoff_frames.

    If `info_root` / `ev_root` (already-parsed ElementTree roots) are
    supplied, the corresponding XML file is NOT re-parsed from disk —
    lets callers share a single parse of each file across the pipeline.
    """
    if info_root is None:
        info_root = ET.parse(path_info).getroot()
    kickoff_str = info_root.find('.//General').attrib['KickoffTime']
    kickoff_utc = pd.Timestamp(kickoff_str).tz_convert('UTC')

    other_info = info_root.find('.//OtherGameInformation')
    if other_info is not None and 'TotalTimeFirstHalf' in other_info.attrib:
        h1_end_utc = kickoff_utc + pd.Timedelta(
            milliseconds=int(other_info.attrib['TotalTimeFirstHalf']))
    else:
        h1_end_utc = kickoff_utc + pd.Timedelta(minutes=50)

    if ev_root is None:
        ev_root = ET.parse(path_events).getroot()

    second_half_start_utc = None
    kickoff_frames        = {'firstHalf': None, 'secondHalf': None}

    for event in ev_root.findall('Event'):
        event_time_str = event.attrib.get('EventTime')
        if event_time_str is None:
            continue
        has_kickoff = any(child.tag == 'KickOff' for child in event)
        if not has_kickoff:
            continue

        event_utc = pd.Timestamp(event_time_str).tz_convert('UTC')
        elapsed_s = (event_utc - kickoff_utc).total_seconds()
        if elapsed_s < 0:
            continue

        kickoff_child = next(c for c in event if c.tag == 'KickOff')
        game_section  = kickoff_child.attrib.get('GameSection', '')

        if second_half_start_utc is None and event_utc > h1_end_utc:
            second_half_start_utc = event_utc
            label = "KickOff secondHalf" if game_section == 'secondHalf' else "first KickOff after H1 (fallback)"
            print(f"  [H2 anchor] {label} @ {event_utc} "
                  f"(break = {(event_utc - h1_end_utc).total_seconds()/60:.1f} min)")

        if event_utc < (second_half_start_utc or h1_end_utc):
            gs    = 'firstHalf'
            frame = int(round(elapsed_s * framerate))
        else:
            gs          = 'secondHalf'
            elapsed_2nd = (event_utc - second_half_start_utc).total_seconds()
            frame       = int(round(elapsed_2nd * framerate))

        if kickoff_frames[gs] is None:
            kickoff_frames[gs] = frame

        if (kickoff_frames['firstHalf'] is not None
                and kickoff_frames['secondHalf'] is not None
                and second_half_start_utc is not None):
            break

    if second_half_start_utc is None:
        print("[WARN] H2 KickOff not found — falling back to h1_end_utc")
        second_half_start_utc = h1_end_utc
    if kickoff_frames['firstHalf'] is None:
        print("[WARN] firstHalf KickOff not found — using frame 0")
        kickoff_frames['firstHalf'] = 0
    if kickoff_frames['secondHalf'] is None:
        print("[WARN] secondHalf KickOff not found — falling back to 67500")
        kickoff_frames['secondHalf'] = 67500

    print(f"[OK] kickoff frames — "
          f"H1: frame {kickoff_frames['firstHalf']} "
          f"({kickoff_frames['firstHalf']/framerate:.1f}s), "
          f"H2: frame {kickoff_frames['secondHalf']} "
          f"({kickoff_frames['secondHalf']/framerate:.1f}s into H2 tracking).")

    return {
        'kickoff_utc':           kickoff_utc,
        'second_half_start_utc': second_half_start_utc,
        'kickoff_frames':        kickoff_frames,
    }


# ============================================================================
# Parse contacts from EventData
# UTC conversion is batched with pd.to_datetime; game_section and frame are
# assigned with np.where / Series.map (no iterrows).
# ============================================================================

def _parse_contacts_from_event_data(event_data,
                                    anchors: dict,
                                    framerate: int = 25) -> pd.DataFrame:
    """
    Extract ball contacts from an already-loaded floodlight EventData object.
    Returns an empty DataFrame if event_data is missing or unusable
    (signals the caller to fall back to the XML parser).
    """
    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    EVENT_ROLE_MAP = {
        'Play':              'passer',
        'ThrowIn':           'passer',
        'FreeKick':          'passer',
        'GoalKick':          'passer',
        'KickOff':           'passer',
        'CornerKick':        'passer',
        'RefereeBall':       'passer',
        'OtherBallAction':   'ball_action',
        'BallClaiming':      'ball_claim',
        'ShotAtGoal':        'shooter',
        'SuccessfulShot':    'shooter',
        'ShotWide':          'shooter',
        'SavedShot':         'shooter',
        'BlockedShot':       'shooter',
        'ShotWoodWork':      'shooter',
        'ChanceWithoutShot': 'shooter',
        'Foul':              'fouled',
    }

    if event_data is None:
        return pd.DataFrame()

    try:
        if hasattr(event_data, 'events'):
            df_ev = event_data.events
        elif hasattr(event_data, '__iter__'):
            parts = list(event_data)
            df_ev = pd.concat(
                [p.events for p in parts if hasattr(p, 'events')],
                ignore_index=True
            )
        else:
            raise ValueError("Unknown EventData format")

        col_type = next((c for c in ['eID', 'event_type', 'type']    if c in df_ev.columns), None)
        col_pid  = next((c for c in ['player_id', 'pID', 'player']   if c in df_ev.columns), None)
        col_time = next((c for c in ['event_time', 'datetime', 'gameclock'] if c in df_ev.columns), None)

        if not (col_type and col_pid and col_time):
            raise ValueError(f"Missing columns (type={col_type}, pid={col_pid}, time={col_time})")

        # Batch UTC conversion
        times     = pd.to_datetime(df_ev[col_time], utc=True, errors='coerce')
        elapsed_s = (times - kickoff_utc).dt.total_seconds()

        valid_mask = df_ev[col_pid].notna() & (elapsed_s >= 0) & times.notna()
        df_v = df_ev[valid_mask].copy()

        if df_v.empty:
            return pd.DataFrame()

        t_v        = times[valid_mask]
        elapsed_v  = elapsed_s[valid_mask]
        is_h1      = t_v < second_half_start_utc
        elapsed_h2 = (t_v - second_half_start_utc).dt.total_seconds()

        df_v = df_v.assign(
            game_section=np.where(is_h1, 'firstHalf', 'secondHalf'),
            abs_frame=np.where(
                is_h1,
                (elapsed_v.values  * framerate).round().astype(int),
                (elapsed_h2.values * framerate).round().astype(int),
            )
        )

        # Role mapping
        df_v['role'] = df_v[col_type].map(EVENT_ROLE_MAP)
        active = df_v.dropna(subset=['role'])

        rows_list = active[['person_id' if 'person_id' in active.columns else col_pid,
                             'role', 'game_section', 'abs_frame', col_type]].copy()
        rows_list = rows_list.rename(columns={col_pid: 'person_id', col_type: 'event_tag'})

        # Optional recipient column
        rcol = next((c for c in ['recipient_id', 'recipient'] if c in df_v.columns), None)
        if rcol:
            rec = df_v[[rcol, 'game_section', 'abs_frame', col_type]].copy()
            rec = rec.dropna(subset=[rcol])
            rec = rec.rename(columns={rcol: 'person_id', col_type: 'event_tag'})
            rec['role'] = 'recipient'
            rows_list = pd.concat([rows_list, rec], ignore_index=True)

        if rows_list.empty:
            return pd.DataFrame()

        df_contacts = rows_list.drop_duplicates(
            subset=['person_id', 'game_section', 'abs_frame', 'role']
        ).reset_index(drop=True)

        print(f"  [contacts] {len(df_contacts)} events via floodlight EventData.")
        return df_contacts

    except Exception as e:
        print(f"  [contacts][WARN] floodlight EventData unusable ({e}).")
        return pd.DataFrame()


def _parse_contacts_xml_fallback(anchors: dict, framerate: int = 25,
                                  path_events: str = None,
                                  ev_root=None) -> pd.DataFrame:
    """
    XML fallback — uses pre-computed anchors (avoids re-parsing matchinfo).
    If `ev_root` (already-parsed root) is supplied, path_events is not
    re-read from disk.
    """
    if ev_root is None and path_events is None:
        raise ValueError("path_events or ev_root required for XML fallback")

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    PLAY_WRAPPERS = {'KickOff', 'ThrowIn', 'GoalKick', 'FreeKick',
                     'CornerKick', 'RefereeBall'}
    SHOT_TAGS     = {'ShotAtGoal', 'SuccessfulShot', 'ShotWide',
                     'SavedShot', 'BlockedShot', 'ShotWoodWork',
                     'ChanceWithoutShot'}

    if ev_root is None:
        ev_root = ET.parse(path_events).getroot()
    rows    = []

    def elapsed_to_frame_and_section(event_utc):
        elapsed_s = (event_utc - kickoff_utc).total_seconds()
        if elapsed_s < 0:
            return None, None
        if event_utc < second_half_start_utc:
            return 'firstHalf', int(round(elapsed_s * framerate))
        elapsed_2nd = (event_utc - second_half_start_utc).total_seconds()
        return 'secondHalf', int(round(elapsed_2nd * framerate))

    def add(person_id, role, gs, frame, tag):
        if person_id and gs:
            rows.append({'person_id': person_id, 'role': role,
                         'game_section': gs, 'abs_frame': frame, 'event_tag': tag})

    for event in ev_root.findall('Event'):
        t_str = event.attrib.get('EventTime')
        if not t_str:
            continue
        event_utc = pd.Timestamp(t_str).tz_convert('UTC')
        gs, frame = elapsed_to_frame_and_section(event_utc)
        if gs is None:
            continue

        for child in event:
            tag   = child.tag
            plays = []
            if tag == 'Play':
                plays = [child]
            elif tag in PLAY_WRAPPERS:
                plays = child.findall('Play')

            for play in plays:
                add(play.attrib.get('Player'),    'passer',    gs, frame, tag)
                add(play.attrib.get('Recipient'), 'recipient', gs, frame, tag)

            if tag == 'OtherBallAction':
                add(child.attrib.get('Player'), 'ball_action', gs, frame, tag)
            elif tag == 'BallClaiming':
                add(child.attrib.get('Player'), 'ball_claim', gs, frame, tag)
            elif tag == 'TacklingGame':
                if child.attrib.get('WinnerRole') == 'withBallControl':
                    add(child.attrib.get('Winner'), 'duel_winner', gs, frame, tag)
                if child.attrib.get('LoserRole') == 'withBallControl':
                    add(child.attrib.get('Loser'), 'duel_loser', gs, frame, tag)
            elif tag in SHOT_TAGS:
                add(child.attrib.get('Player'), 'shooter', gs, frame, tag)
            elif tag == 'Foul':
                add(child.attrib.get('Fouled'), 'fouled', gs, frame, tag)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=['person_id', 'game_section', 'abs_frame', 'role']
        ).reset_index(drop=True)
    print(f"  [contacts] {len(df)} events via fallback XML parser.")
    return df


# ============================================================================
# Possession intervals
# frame_end lookup is vectorised with np.searchsorted over the whole group.
# ============================================================================

def build_possession_intervals(contacts, framerate=25,
                               recipient_delay_s=RECIPIENT_DELAY_S,
                               role_delays_s=None):
    CARRIER_ROLES = {'recipient', 'ball_action', 'ball_claim',
                     'duel_winner', 'duel_loser', 'shooter', 'fouled'}

    if role_delays_s is not None:
        delay_frames = {
            role: int(round(role_delays_s.get(role, 0.0) * framerate))
            for role in CARRIER_ROLES
        }
    else:
        delay_frames = {role: 0 for role in CARRIER_ROLES}
        delay_frames['recipient'] = int(round(recipient_delay_s * framerate))

    intervals = {}
    for gs, group in contacts.groupby('game_section'):
        carriers = (
            group[group['role'].isin(CARRIER_ROLES)]
            .sort_values('abs_frame')
            .reset_index(drop=True)
        )
        if carriers.empty:
            intervals[gs] = []
            continue

        all_frames  = carriers['abs_frame'].values.astype(np.int64)
        delay_vals  = carriers['role'].map(delay_frames).fillna(0).astype(np.int64).values
        frame_starts = all_frames + delay_vals

        # frame_end = next event frame, or +250 if last.
        # np.where evaluates both branches eagerly, so we clip the index first
        # to avoid IndexError on the last element; the mask ensures the clipped
        # value is never actually used in the output.
        next_idx      = np.searchsorted(all_frames, all_frames, side='right')
        safe_next_idx = np.minimum(next_idx, len(all_frames) - 1)
        frame_ends    = np.where(
            next_idx < len(all_frames),
            all_frames[safe_next_idx],
            all_frames + 250,
        )

        valid_mask = frame_starts < frame_ends
        result = list(zip(
            carriers.loc[valid_mask, 'person_id'].values,
            frame_starts[valid_mask].tolist(),
            frame_ends[valid_mask].tolist(),
        ))
        intervals[gs] = result

    return intervals


# ============================================================================
# apply_possession_intervals_to_array — vectorised pid_to_team lookup
# ============================================================================

def apply_possession_intervals_to_array(possession_array, intervals, teamsheets, half_name):
    pid_to_team = {}
    for team_name, ts_obj in teamsheets.items():
        ts       = ts_obj.teamsheet
        col_pid  = next((c for c in ['player_id', 'pID'] if c in ts.columns), None)
        if col_pid is None:
            continue
        team_code = 1 if team_name == 'Home' else 0
        valid = ts[col_pid].dropna()
        for pid in valid:
            pid_to_team[pid] = team_code

    new_poss = possession_array.copy()
    for pid, start, end in intervals.get(half_name, []):
        team = pid_to_team.get(pid)
        if team is not None:
            new_poss[start:end] = team
    return new_poss


# ============================================================================
# Setpiece blackout
# Returns a dict of sorted np.ndarray (N×2) intervals instead of Python sets.
# Vectorised filter applied in process_match via _apply_blackout_mask.
# Each row is an interval [start (inclusive), end (exclusive)].
# ============================================================================

def build_setpiece_blackout_frames(path_events, path_info,
                                   framerate=25, anchors=None,
                                   ev_root=None):
    """
    Return blackout intervals per half as a sorted np.ndarray (N, 2),
    compatible with np.searchsorted.
    If `ev_root` (already-parsed root) is supplied, path_events is not
    re-read from disk.
    """
    if anchors is None:
        anchors = _build_time_anchors(path_events, path_info, framerate, ev_root=ev_root)

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    if ev_root is None:
        ev_root = ET.parse(path_events).getroot()
    intervals = {'firstHalf': [], 'secondHalf': []}
    n_windows = 0

    first_kickoff_seen = {'firstHalf': False, 'secondHalf': False}
    kickoff_count      = {'firstHalf': 0,     'secondHalf': 0}

    for event in ev_root.findall('Event'):
        event_time_str = event.attrib.get('EventTime')
        if event_time_str is None:
            continue

        event_utc = pd.Timestamp(event_time_str).tz_convert('UTC')

        for child in event:
            tag = child.tag
            if tag not in SETPIECE_OFFSETS:
                continue

            offset_frames = int(round(SETPIECE_OFFSETS[tag] * framerate))
            elapsed_s     = (event_utc - kickoff_utc).total_seconds()
            if elapsed_s < 0:
                continue

            if event_utc < second_half_start_utc:
                gs          = 'firstHalf'
                event_frame = int(round(elapsed_s * framerate))
            else:
                gs          = 'secondHalf'
                event_frame = int(round(
                    (event_utc - second_half_start_utc).total_seconds() * framerate))

            if tag == 'KickOff':
                kickoff_count[gs] += 1
                if not first_kickoff_seen[gs]:
                    blackout_start = 0
                    first_kickoff_seen[gs] = True
                    print(f"  [blackout] {gs} KickOff #1: event_frame={event_frame} "
                          f"(gap={event_frame/framerate:.1f}s) → blackout extended to frame 0")
                else:
                    blackout_start = event_frame
                    print(f"  [blackout] {gs} KickOff #{kickoff_count[gs]} (after goal): "
                          f"frame={event_frame} ({frame_to_minute(event_frame)}) "
                          f"→ [{blackout_start}, {event_frame + offset_frames})")
            else:
                blackout_start = event_frame

            intervals[gs].append((blackout_start, event_frame + offset_frames))
            n_windows += 1
            break  # one setpiece per Event

    # Convert to sorted NumPy arrays (required for searchsorted)
    result = {}
    for gs, ivs in intervals.items():
        if ivs:
            arr = np.array(ivs, dtype=np.int64)
            arr = arr[arr[:, 0].argsort()]
            result[gs] = arr
        else:
            result[gs] = np.empty((0, 2), dtype=np.int64)

    n_h1 = len(result.get('firstHalf',  []))
    n_h2 = len(result.get('secondHalf', []))
    print(f"[OK] setpiece blackout — {n_windows} windows "
          f"(H1 KickOff: {kickoff_count['firstHalf']}, H2: {kickoff_count['secondHalf']}) | "
          f"{n_h1} intervals H1, {n_h2} intervals H2.")
    return result


def _apply_blackout_mask(df_movements: pd.DataFrame,
                         blackout_intervals: dict) -> np.ndarray:
    """
    Return a boolean mask: True if a movement starts inside a blackout interval.
    Uses np.searchsorted (O(n log m)) — replaces row-by-row apply and avoids
    allocating a set of individual frames.

    blackout_intervals: dict {'firstHalf': ndarray(N,2), 'secondHalf': ...}
    """
    mask = np.zeros(len(df_movements), dtype=bool)

    for half, ivs in blackout_intervals.items():
        if len(ivs) == 0:
            continue
        half_mask = (df_movements['half'] == half).values
        if not half_mask.any():
            continue

        starts = df_movements.loc[half_mask, 'start_frame'].values.astype(np.int64)
        # Find the interval whose start is the largest value <= starts[i]
        idx        = np.searchsorted(ivs[:, 0], starts, side='right') - 1
        valid      = idx >= 0
        in_blackout = np.zeros(len(starts), dtype=bool)
        if valid.any():
            # Movement is inside the blackout if starts[i] < end of the matched interval
            in_blackout[valid] = starts[valid] < ivs[idx[valid], 1]
        mask[half_mask] = in_blackout

    return mask


# ============================================================================
# Goals, shots, scoreline and shot-follow-up flag
# ============================================================================

def parse_goals_and_shots(path_events, path_info, anchors, framerate=25,
                          info_root=None, ev_root=None):
    """
    Parse the events XML for goals and shots.
    If `info_root` / `ev_root` (already-parsed roots) are supplied, the
    corresponding XML file is not re-read from disk.

    Elapsed times are expressed in seconds SINCE THE KICKOFF OF THE
    RESPECTIVE HALF (0 at kickoff of either half) — the same "per_half"
    convention as `_compute_elapsed_seconds_vectorized(..., per_half=True)`
    in `process_match`, so movements and events can be compared directly.

    Reuses the already-computed `anchors` (kickoff_utc, second_half_start_utc)
    instead of re-deriving them, so the half boundary is guaranteed
    consistent with the rest of the pipeline (kickoff/blackout detection).

    Returns (df_goals, df_shots):
      df_goals: one row per goal, columns
                [half, elapsed_s, scoring_team, home_score, away_score]
                home_score/away_score = CUMULATIVE score right after this goal.
      df_shots: one row per shot attempt (goals included), columns
                [half, elapsed_s, team, shot_type]
    """
    kickoff_utc            = anchors['kickoff_utc']
    second_half_start_utc  = anchors['second_half_start_utc']

    if info_root is None:
        info_root = ET.parse(path_info).getroot()
    home_team_id = away_team_id = None
    for el in info_root.iter():
        role = el.get('Role', '')
        tid  = el.get('TeamId') or el.get('TeamID', '')
        if not tid:
            continue
        if role == 'home':
            home_team_id = tid
        elif role in ('guest', 'away'):
            away_team_id = tid

    SHOT_TAGS = {'ShotAtGoal', 'ShotOnTarget', 'ShotOffTarget',
                 'SavedShot', 'ShotWide', 'ShotHigh', 'BlockedShot'}

    if ev_root is None:
        ev_root = ET.parse(path_events).getroot()

    goal_rows, shot_rows = [], []
    home_score = away_score = 0

    for event in ev_root.findall('Event'):
        t_str = event.attrib.get('EventTime')
        if not t_str:
            continue
        event_utc = pd.Timestamp(t_str).tz_convert('UTC')

        if event_utc < second_half_start_utc:
            half      = 'firstHalf'
            elapsed_s = (event_utc - kickoff_utc).total_seconds()
        else:
            half      = 'secondHalf'
            elapsed_s = (event_utc - second_half_start_utc).total_seconds()
        if elapsed_s < 0:
            continue

        # Descend into all descendants: a shot tag can be a direct child of
        # <Event> or wrapped inside <Penalty>, <FreeKick>, etc.
        for shot_el in event.iter():
            if shot_el.tag not in SHOT_TAGS:
                continue

            team_id   = shot_el.get('Team', '')
            shot_team = 'Home' if team_id == home_team_id else 'Away'

            shot_rows.append({'half': half, 'elapsed_s': elapsed_s,
                              'team': shot_team, 'shot_type': shot_el.tag})

            # A goal is a <SuccessfulShot> direct child of the shot tag.
            goal_el = shot_el.find('SuccessfulShot')
            if goal_el is not None:
                if shot_team == 'Home':
                    home_score += 1
                else:
                    away_score += 1
                goal_rows.append({'half': half, 'elapsed_s': elapsed_s,
                                  'scoring_team': shot_team,
                                  'home_score': home_score, 'away_score': away_score})

    df_goals = (pd.DataFrame(goal_rows) if goal_rows else
                pd.DataFrame(columns=['half', 'elapsed_s', 'scoring_team',
                                      'home_score', 'away_score']))
    df_shots = (pd.DataFrame(shot_rows) if shot_rows else
                pd.DataFrame(columns=['half', 'elapsed_s', 'team', 'shot_type']))

    print(f"[OK] events for scoreline/shots — {len(df_goals)} goal(s), {len(df_shots)} shot(s).")
    return df_goals, df_shots


def compute_scoreline_vectorized(elapsed_s_arr, half_arr, df_goals):
    """
    Vectorised lookup of the scoreline in effect at the moment (half,
    elapsed_s) of each movement's start.

    Methodology
    -----------
    Processed one half at a time (goals in the other half are irrelevant):
      1. Goals of that half are sorted by `elapsed_s`.
      2. For every movement, `np.searchsorted(goal_elapsed_sorted,
         elapsed_s, side='right') - 1` gives the index of the LAST goal
         that occurred at or before the movement's `elapsed_s` — i.e. the
         goal that set the scoreline currently in effect.
      3. If no such goal exists (movement occurs before the half's first
         goal, or the half has no goals at all), the scoreline defaults to
         "0-0" (home_score = away_score = 0).

    Returns (scoreline_str_array, home_score_array, away_score_array), all
    aligned with `elapsed_s_arr` / `half_arr`.
    """
    n = len(elapsed_s_arr)
    home_out = np.zeros(n, dtype=int)
    away_out = np.zeros(n, dtype=int)

    for half_val in np.unique(half_arr):
        mask = half_arr == half_val
        goals_half = df_goals[df_goals['half'] == half_val].sort_values('elapsed_s')
        if goals_half.empty:
            continue

        g_elapsed = goals_half['elapsed_s'].values
        g_home    = goals_half['home_score'].values
        g_away    = goals_half['away_score'].values

        idx   = np.searchsorted(g_elapsed, elapsed_s_arr[mask], side='right') - 1
        valid = idx >= 0

        sub_home = np.zeros(int(mask.sum()), dtype=int)
        sub_away = np.zeros(int(mask.sum()), dtype=int)
        sub_home[valid] = g_home[idx[valid]]
        sub_away[valid] = g_away[idx[valid]]

        home_out[mask] = sub_home
        away_out[mask] = sub_away

    scoreline_arr = np.array([f"{h}-{a}" for h, a in zip(home_out, away_out)], dtype=object)
    return scoreline_arr, home_out, away_out


def compute_goal_indicator_vectorized(home_score_arr, away_score_arr, location_arr):
    """
    Signed goal-deficit indicator for each movement, from the perspective
    of the team performing it (`location_arr`, 'Home'/'Away').

    diff = away_score - home_score  (positive => Away is leading)
      - movement performed by Away team -> indicator =  diff
      - movement performed by Home team -> indicator = -diff

    So the indicator is always POSITIVE when the performing team is
    leading, NEGATIVE when trailing, and 0 when level — regardless of
    which side (Home/Away) the team is on.
    """
    diff    = away_score_arr - home_score_arr
    is_away = np.asarray(location_arr) == 'Away'
    return np.where(is_away, diff, -diff)


def flag_shot_within_window_vectorized(half_arr, location_arr, end_elapsed_s_arr,
                                       df_shots, window_s):
    """
    Vectorised version of "did this movement's team take a shot within
    `window_s` seconds AFTER the movement ended (same half)?".

    Methodology
    -----------
    Merge-based approach (avoiding a per-row Python loop):
      1. Build a small per-movement frame with (half, team, end_elapsed_s).
      2. Left-merge it against all shots on (half, team) — this creates
         one row per (movement, candidate shot) pair sharing the same half
         and team.
      3. Keep only pairs where the shot's `elapsed_s` falls in
         [end_elapsed_s, end_elapsed_s + window_s].
      4. A movement is flagged True if AT LEAST ONE such pair survives.

    Returns a boolean np.ndarray aligned with the inputs.
    """
    n = len(end_elapsed_s_arr)
    if df_shots.empty or window_s is None:
        return np.zeros(n, dtype=bool)

    df_m = pd.DataFrame({
        'half':          np.asarray(half_arr),
        'team':          np.asarray(location_arr),
        'end_elapsed_s': np.asarray(end_elapsed_s_arr),
        '_idx':          np.arange(n),
    })
    shots = (df_shots[['half', 'team', 'elapsed_s']]
             .rename(columns={'elapsed_s': 'shot_elapsed_s'}))

    merged = df_m.merge(shots, on=['half', 'team'], how='left')
    in_window = (
        merged['shot_elapsed_s'].notna()
        & (merged['shot_elapsed_s'] >= merged['end_elapsed_s'])
        & (merged['shot_elapsed_s'] <= merged['end_elapsed_s'] + window_s)
    )
    flagged_idx = merged.loc[in_window, '_idx'].unique()

    flag = np.zeros(n, dtype=bool)
    if len(flagged_idx) > 0:
        flag[flagged_idx] = True
    return flag


# ============================================================================
# Main pipeline
# ============================================================================

def process_match(
        match_id, DATA_DIR, dict_direction,
        butterworth_Wn      = 0.5,
        butterworth_order   = 1,
        min_valley_distance = 15,
        valley_depth_abs    = 0.15,
        valley_depth_rel    = 0.05,
        power_threshold     = 6.0,
        min_segment_frames  = 10,
        framerate           = 25,
        speed_thresholds    = None,
        pre_run_window_s    = PRE_RUN_WINDOW_S,
        shot_window_s       = SHOT_WINDOW_S,
):
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
                f"No file matching '{match_id}' + '{keyword}' in {DATA_DIR}")
        return os.path.join(DATA_DIR, matches[0])

    path_positions = find_file('positions')
    path_events    = find_file('events')
    path_info      = find_file('matchinformation')

    # ------------------------------------------------------------------
    # 2. Time anchors — computed only once
    #
    # events.xml and matchinformation.xml are each parsed with ET.parse()
    # EXACTLY ONCE here (info_root / ev_root). Every downstream function
    # that used to re-parse these files on its own (_build_time_anchors,
    # build_setpiece_blackout_frames, parse_goals_and_shots,
    # the contacts XML fallback) now accepts the
    # already-parsed root and skips its own ET.parse() call.
    # ------------------------------------------------------------------
    print("Computing time anchors…")
    info_root = ET.parse(path_info).getroot()
    ev_root   = ET.parse(path_events).getroot()

    anchors               = _build_time_anchors(path_events, path_info, framerate,
                                                 info_root=info_root, ev_root=ev_root)
    second_half_start_utc = anchors['second_half_start_utc']  # noqa: F841
    h1_kickoff            = anchors['kickoff_frames']['firstHalf']
    h2_kickoff            = anchors['kickoff_frames']['secondHalf']

    # ------------------------------------------------------------------
    # 3. Load tracking + events (floodlight — single parse)
    # ------------------------------------------------------------------
    xy, possession, ballstatus, teamsheets, pitch = dfl.read_position_data_xml(
        path_positions, path_info,
        teamsheet_home=None, teamsheet_away=None,
    )

    # Snapshot of raw possession from the positions file,
    # before any rewrite by XML contacts or smoothing.
    possession_raw = {
        half_name: np.array(possession[half_name]).flatten().astype(int)
        for half_name in ['firstHalf', 'secondHalf']
    }

    _events, _teamsheets_ev, _pitch = dfl.read_event_data_xml(path_events, path_info)
    pitch.sport = "football"

    # ------------------------------------------------------------------
    # 3b. Contacts + possession intervals (no re-parse)
    # ------------------------------------------------------------------
    contacts = _parse_contacts_from_event_data(_events, anchors, framerate=framerate)
    if contacts.empty:
        print("  [contacts] EventData empty — falling back to XML.")
        contacts = _parse_contacts_xml_fallback(anchors, framerate,
                                                path_events=path_events, ev_root=ev_root)

    possession_intervals = build_possession_intervals(
        contacts, framerate=framerate, recipient_delay_s=RECIPIENT_DELAY_S)

    for half_name in ['firstHalf', 'secondHalf']:
        raw_poss = np.array(possession[half_name]).flatten().astype(int)
        possession[half_name] = apply_possession_intervals_to_array(
            raw_poss, possession_intervals, teamsheets, half_name)

    xy_all = {
        'firstHalf':  {'Home': xy['firstHalf']['Home'],  'Away': xy['firstHalf']['Away']},
        'secondHalf': {'Home': xy['secondHalf']['Home'], 'Away': xy['secondHalf']['Away']},
    }

    # ------------------------------------------------------------------
    # 4. Filtered positions + velocities
    #
    # butterworth_lowpass() is run EXACTLY ONCE per (half, team) here and
    # the filtered XY object is kept in `xy_filtered_dict`. It used to be
    # re-run identically in step 6 (movement extraction) and step 7
    # (total distance) — same input, same Wn/order — tripling the cost of
    # the (expensive) filtfilt pass over the whole match for no benefit.
    # Both steps below now reuse `xy_filtered_dict` instead of refiltering.
    # ------------------------------------------------------------------
    xy_filtered_dict       = {half: {} for half in ['firstHalf', 'secondHalf']}
    velocity_filtered_dict = {half: {} for half in ['firstHalf', 'secondHalf']}
    for half, teams in xy_all.items():
        for team, xy_data in teams.items():
            xy_f = butterworth_lowpass(xy_data, remove_short_seqs=True,
                                       Wn=butterworth_Wn, order=butterworth_order)
            xy_filtered_dict[half][team] = xy_f

            vm_f = VelocityModel()
            vm_f.fit(xy_f)
            velocity_filtered_dict[half][team] = vm_f.velocity().property

    # ------------------------------------------------------------------
    # 5. Light possession smoothing (cutoff=10 frames = 0.4 s)
    # Absorbs micro-changes from noisy isolated defensive touches
    # without overwriting the real possession from the positions file.
    # ------------------------------------------------------------------
    for half_name, half_num in [('firstHalf', 1), ('secondHalf', 2)]:
        half_arr = np.full(len(possession_raw[half_name]), half_num, dtype=int)
        smoothed, _, _ = smooth_possession(possession_raw[half_name], half_arr, cutoff=10)
        possession_raw[half_name] = smoothed

    # ------------------------------------------------------------------
    # 6. Discrete movement extraction
    # ------------------------------------------------------------------
    H2_BASE_SECONDS = 45 * 60

    # Vectorised elapsed-time computation, shared by the timecode formatter
    # below and by the scoreline / shot-flag lookups further down the
    # pipeline (both need elapsed seconds since kickoff).
    def _compute_elapsed_seconds_vectorized(frames_arr, half_labels, kickoff_h1, kickoff_h2,
                                             per_half=False):
        """
        frames_arr  : np.ndarray (int) — raw tracking-array frame indices
        half_labels : pd.Series or np.ndarray of str ('firstHalf'/'secondHalf')
        per_half    : if False (default), returns CONTINUOUS match seconds
                      (0 at H1 kickoff, 45*60+x during H2) — used for display
                      timecodes. If True, returns seconds elapsed SINCE THE
                      KICKOFF OF THE RESPECTIVE HALF (0 at kickoff of either
                      half) — used to compare against goal/shot timestamps,
                      which are themselves stored per-half.
        """
        is_h1 = np.asarray(half_labels) == 'firstHalf'
        if per_half:
            elapsed_s = np.where(
                is_h1,
                (frames_arr - kickoff_h1) / framerate,
                (frames_arr - kickoff_h2) / framerate,
            )
        else:
            elapsed_s = np.where(
                is_h1,
                (frames_arr - kickoff_h1) / framerate,
                (frames_arr - kickoff_h2) / framerate + H2_BASE_SECONDS,
            )
        return np.maximum(elapsed_s, 0.0)

    def _compute_timecodes_vectorized(frames_arr, half_labels, kickoff_h1, kickoff_h2):
        """
        frames_arr  : np.ndarray (int)
        half_labels : pd.Series or np.ndarray of str
        """
        video_s = _compute_elapsed_seconds_vectorized(
            frames_arr, half_labels, kickoff_h1, kickoff_h2, per_half=False)
        # sec_to_minute is scalar (string formatting) — one pass is unavoidable,
        # but it is called only once per column.
        return np.fromiter((sec_to_minute(s) for s in video_s),
                           dtype=object, count=len(video_s))

    ls_movement_dfs   = []
    dict_trajectories = {}

    for team in ['Home', 'Away']:
        ls_movements_team = []

        for half in ['firstHalf', 'secondHalf']:
            teamsheets[team].add_xIDs()

            xy_f = xy_filtered_dict[half][team].xy

            movements_dict = extract_discrete_movements_all_players(
                velocity_filtered_dict[half][team],
                xy_f,
                possession_raw[half],
                min_valley_distance=min_valley_distance,
                valley_depth_abs=valley_depth_abs,
                valley_depth_rel=valley_depth_rel,
                power_threshold=power_threshold,
                min_segment_frames=min_segment_frames,
                framerate=framerate,
                pre_run_window_s=pre_run_window_s,
            )

            summary_df = summarize_movements_to_dataframe(
                movements_dict, teamsheets[team].teamsheet, framerate=framerate)

            summary_df['half']     = half
            summary_df['location'] = team

            summary_df['start_timecode'] = _compute_timecodes_vectorized(
                summary_df['start_frame'].values, summary_df['half'].values,
                h1_kickoff, h2_kickoff)
            summary_df['end_timecode'] = _compute_timecodes_vectorized(
                summary_df['end_frame'].values, summary_df['half'].values,
                h1_kickoff, h2_kickoff)

            ls_movement_dfs.append(summary_df)
            ls_movements_team.append(movements_dict)

        dict_trajectories[team] = {
            k: ls_movements_team[0][k] + ls_movements_team[1][k]
            for k in ls_movements_team[0]
        }

    df_movements = pd.concat(ls_movement_dfs, ignore_index=True)
    df_movements = df_movements[df_movements['distance_m'] > 0]

    # ------------------------------------------------------------------
    # 7. Total distance per player
    # ------------------------------------------------------------------
    ls_dfs_distance = []
    for half, teams in xy_filtered_dict.items():
        for team, xy_f in teams.items():
            dm = DistanceModel()
            dm.fit(xy_f)
            df_players = teamsheets[team].teamsheet.copy()
            df_players['distance_covered'] = np.nansum(dm.distance_covered().property, axis=0)
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
    # 8. Play direction (vectorised)
    # ------------------------------------------------------------------
    if match_id not in dict_direction:
        print(f"[WARN] '{match_id}' not found in direction JSON — 'direction' = NaN.")

    dx = df_movements["x_end"] - df_movements["x_start"]
    dy = df_movements["y_end"] - df_movements["y_start"]
    base_direction = dx.div(dx.abs().add(dy.abs())).fillna(0)
    home_by_half   = dict_direction.get(match_id, {}).get("Home", {})
    home_direction = df_movements["half"].map(home_by_half)
    player_direction = home_direction.where(
        df_movements["location"].ne("Away"),
        home_direction.map({
            "left_to_right": "right_to_left",
            "right_to_left": "left_to_right",
        }),
    )
    multiplier = player_direction.map(
        {"left_to_right": 1, "right_to_left": -1}).fillna(1)

    df_movements["direction"]   = base_direction * multiplier
    df_movements["attack_sign"] = multiplier
    df_movements["match_id"]    = match_id

    # ------------------------------------------------------------------
    # 8b. Pitch zones (D/M/A x 1-5) — start / mid / end
    # Reuses `player_direction` (the same per-movement playing-direction
    # string computed just above for the continuous `direction` metric),
    # so zone assignment and direction metrics are always consistent with
    # each other. See assign_zone() for the full methodology.
    # ------------------------------------------------------------------
    
    tuples_start = [
    assign_zone(x, y, direction=d)
    for x, y, d in zip(df_movements["x_start"], df_movements["y_start"], player_direction)
    ]

    df_movements["third_start"], df_movements["lane_start"] = zip(*tuples_start)
    df_movements["zone_start"] = [f"{x}{y}" if isinstance(x, str) else x for x, y in tuples_start]

    tuples_mid = [
        assign_zone(x, y, direction=d)
        for x, y, d in zip(df_movements["x_mid"], df_movements["y_mid"], player_direction)
    ]
    df_movements["third_mid"], df_movements["lane_mid"] = zip(*tuples_mid)
    df_movements["zone_mid"] = [f"{x}{y}" if isinstance(x, str) else x for x, y in tuples_mid]

    tuples_end = [
        assign_zone(x, y, direction=d)
        for x, y, d in zip(df_movements["x_end"], df_movements["y_end"], player_direction)
    ]   
    df_movements["third_end"], df_movements["lane_end"] = zip(*tuples_end)
    df_movements["zone_end"] = [f"{x}{y}" if isinstance(x, str) else x for x, y in tuples_end]
    

    # ------------------------------------------------------------------
    # 8c. Scoreline, goal indicator, and shot-within-window flag
    # Reuses `anchors` (kickoff_utc / second_half_start_utc) so the half
    # boundary used for goal/shot timestamps matches the rest of the
    # pipeline exactly (kickoff detection, blackout windows, etc.).
    # ------------------------------------------------------------------
    df_goals, df_shots = parse_goals_and_shots(
        path_events, path_info, anchors, framerate=framerate,
        info_root=info_root, ev_root=ev_root)

    # Elapsed seconds SINCE THE KICKOFF OF THE RESPECTIVE HALF (per_half=True),
    # matching the convention used when parsing goals/shots above.
    start_elapsed_s = _compute_elapsed_seconds_vectorized(
        df_movements["start_frame"].values, df_movements["half"].values,
        h1_kickoff, h2_kickoff, per_half=True)
    end_elapsed_s = _compute_elapsed_seconds_vectorized(
        df_movements["end_frame"].values, df_movements["half"].values,
        h1_kickoff, h2_kickoff, per_half=True)

    scoreline_arr, home_score_arr, away_score_arr = compute_scoreline_vectorized(
        start_elapsed_s, df_movements["half"].values, df_goals)
    df_movements["scoreline_at_run_start"] = scoreline_arr
    df_movements["goal_indicator"] = compute_goal_indicator_vectorized(
        home_score_arr, away_score_arr, df_movements["location"].values)

    shot_col = f"shot_within_{int(shot_window_s)}s" if shot_window_s is not None else "shot_within_Ns"
    df_movements[shot_col] = flag_shot_within_window_vectorized(
        df_movements["half"].values, df_movements["location"].values,
        end_elapsed_s, df_shots, shot_window_s)

    # ------------------------------------------------------------------
    # 9. Setpiece blackout exclusion — vectorised via searchsorted
    # ------------------------------------------------------------------
    blackout_intervals = build_setpiece_blackout_frames(
        path_events, path_info,
        framerate=framerate, anchors=anchors,
        ev_root=ev_root,
    )

    mask_blackout = _apply_blackout_mask(df_movements, blackout_intervals)
    n_excluded    = int(mask_blackout.sum())
    df_movements  = df_movements[~mask_blackout].reset_index(drop=True)
    print(f"[OK] {n_excluded} movements excluded (setpiece blackout).")

    for half in ['firstHalf', 'secondHalf']:
        sub = df_movements[df_movements['half'] == half]
        if not sub.empty:
            min_frame = sub['start_frame'].min()
            min_tc    = sub.loc[sub['start_frame'].idxmin(), 'start_timecode']
            print(f"    earliest surviving movement in {half}: frame {min_frame} ({min_tc})")

    # ------------------------------------------------------------------
    # 10. Final column ordering
    # Group columns by theme (identifiers -> timing -> kinematics ->
    # pre-run window -> spatial/zones -> possession -> match context)
    # so the exported CSV reads logically from left to right.
    # Any column not explicitly listed (e.g. future additions) is appended
    # at the end automatically, so this never silently drops data.
    # ------------------------------------------------------------------
    preferred_order = [
        # Identifiers
        'match_id', 'half', 'location', 'team', 'player', 'jID', 'xID', 'position',
        # Timing
        'start_frame', 'end_frame', 'peak_frame',
        'start_timecode', 'end_timecode', 'duration_s',
        # Kinematics
        'peak_speed_kmh', 'avg_velocity_kmh', 'speed_category',
        'distance_m', 'total_energy_cost_J',
        # Pre-run window (see _compute_pre_run_window)
        'pre_run_mean_vel_kmh', 'pre_run_peak_vel_kmh', 'pre_run_window_s',
        # Spatial (see summarize_movements_to_dataframe / assign_zone)
        'x_start', 'y_start', 'zone_start', 'third_start', 'lane_start',
        'x_mid', 'y_mid', 'zone_mid', 'third_mid', 'lane_mid',
        'x_end', 'y_end', 'zone_end', 'third_end', 'lane_end',
        'direction', 'attack_sign',
        # Possession
        'possession', 'possession_ratio', 'possession_contested',
        # Match context
        'scoreline_at_run_start', 'goal_indicator', shot_col,
    ]
    ordered_cols   = [c for c in preferred_order if c in df_movements.columns]
    remaining_cols = [c for c in df_movements.columns if c not in ordered_cols]
    df_movements   = df_movements[ordered_cols + remaining_cols]

    print(f"[OK] {match_id} — {len(df_movements)} discrete movements extracted.")
    return df_movements, df_distances, dict_trajectories