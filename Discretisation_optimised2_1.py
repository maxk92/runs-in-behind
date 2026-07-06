"""
Discretisation pipeline
=======================
Optimised version — pure NumPy possession smoothing, vectorised valley
filtering, merge-based ball-touch tagging, NumPy interval blackout filter.
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
    'ThrowIn':    1.0,
    'FreeKick':   2.0,
    'GoalKick':   2.0,
    'KickOff':    6.0,
    'CornerKick': 2.0,
}

RECIPIENT_DELAY_S = 0.0


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

def extract_movements_for_player(
        player_id, velocities, tracking_data, possession,
        min_valley_distance=15,
        valley_depth_abs=0.15,
        valley_depth_rel=0.05,
        power_threshold=6.0,
        min_segment_frames=10,
        framerate=25,
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
            })

    return pd.DataFrame(rows)


# ============================================================================
# Centralised time anchors — events + matchinfo XML parsed only once
# ============================================================================

def _build_time_anchors(path_events: str, path_info: str,
                        framerate: int = 25) -> dict:
    """
    Parse events + matchinfo XML once.
    Returns kickoff_utc, second_half_start_utc, kickoff_frames.
    """
    info_root   = ET.parse(path_info).getroot()
    kickoff_str = info_root.find('.//General').attrib['KickoffTime']
    kickoff_utc = pd.Timestamp(kickoff_str).tz_convert('UTC')

    other_info = info_root.find('.//OtherGameInformation')
    if other_info is not None and 'TotalTimeFirstHalf' in other_info.attrib:
        h1_end_utc = kickoff_utc + pd.Timedelta(
            milliseconds=int(other_info.attrib['TotalTimeFirstHalf']))
    else:
        h1_end_utc = kickoff_utc + pd.Timedelta(minutes=50)

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
# Shortname → PersonId map — vectorised via set_index, no iterrows
# ============================================================================

def _build_shortname_map_from_teamsheets(teamsheets: dict) -> dict:
    """
    Build {Shortname → PersonId} from floodlight teamsheets.
    No XML parsing, no iterrows.
    """
    mapping = {}
    for team_ts in teamsheets.values():
        ts      = team_ts.teamsheet
        col_pid = next((c for c in ['player_id', 'pID'] if c in ts.columns), None)
        if col_pid is None or 'player' not in ts.columns:
            continue
        valid = ts[['player', col_pid]].dropna()
        mapping.update(valid.set_index('player')[col_pid].to_dict())
    return mapping


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
                                  path_events: str = None) -> pd.DataFrame:
    """
    XML fallback — uses pre-computed anchors (avoids re-parsing matchinfo).
    """
    if path_events is None:
        raise ValueError("path_events required for XML fallback")

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    PLAY_WRAPPERS = {'KickOff', 'ThrowIn', 'GoalKick', 'FreeKick',
                     'CornerKick', 'RefereeBall'}
    SHOT_TAGS     = {'ShotAtGoal', 'SuccessfulShot', 'ShotWide',
                     'SavedShot', 'BlockedShot', 'ShotWoodWork',
                     'ChanceWithoutShot'}

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
# tag_ball_touches — merge-based, O(n log n) instead of row-by-row apply
# ============================================================================

def tag_ball_touches(df_movements, path_events, path_info,
                     framerate=25, anchors=None,
                     teamsheets=None, event_data=None):
    """
    Add a 'has_ball_touch' column to df_movements.
    No XML re-parse when anchors, teamsheets and event_data are provided.
    """
    TOUCH_BUFFER_FRAMES = 1 * framerate
    ACTIVE_TOUCH_ROLES  = {'ball_action', 'passer', 'shooter',
                           'duel_winner', 'duel_loser', 'fouled'}

    if anchors is None:
        anchors = _build_time_anchors(path_events, path_info, framerate)

    contacts = _parse_contacts_from_event_data(event_data, anchors, framerate=framerate)
    if contacts.empty:
        print("  [contacts] EventData empty — falling back to XML.")
        contacts = _parse_contacts_xml_fallback(anchors, framerate, path_events=path_events)

    if contacts.empty:
        print("[WARN] No ball contact events — 'has_ball_touch' set to False.")
        df_movements['has_ball_touch'] = False
        return df_movements

    # Player name → PersonId mapping
    if teamsheets is not None:
        shortname_to_pid = _build_shortname_map_from_teamsheets(teamsheets)
    else:
        info_root = ET.parse(path_info).getroot()
        shortname_to_pid = {
            p.attrib['Shortname']: p.attrib['PersonId']
            for p in info_root.findall('.//Player')
            if 'Shortname' in p.attrib and 'PersonId' in p.attrib
        }

    unknown = set(df_movements['player'].unique()) - set(shortname_to_pid.keys())
    if unknown:
        print(f"[WARN] {len(unknown)} player name(s) not found: {unknown}")

    # Vectorised merge approach — join movements with contact events, filter by time window.
    active_contacts = (
        contacts[contacts['role'].isin(ACTIVE_TOUCH_ROLES)]
        [['person_id', 'game_section', 'abs_frame']]
        .rename(columns={'person_id': '_pid', 'game_section': 'half',
                         'abs_frame': 'touch_frame'})
    )

    df_m = df_movements.copy()
    df_m['_pid']       = df_m['player'].map(shortname_to_pid)
    df_m['_win_start'] = df_m['start_frame'].astype(int) + TOUCH_BUFFER_FRAMES
    df_m['_win_end']   = df_m['end_frame'].astype(int)   - TOUCH_BUFFER_FRAMES
    df_m['_idx']       = np.arange(len(df_m))

    # Only merge rows with a known pid and a valid window
    valid_m = df_m.dropna(subset=['_pid'])
    valid_m = valid_m[valid_m['_win_end'] > valid_m['_win_start']]

    if valid_m.empty:
        df_movements['has_ball_touch'] = False
        return df_movements

    merged = valid_m[['_pid', 'half', '_win_start', '_win_end', '_idx']].merge(
        active_contacts, on=['_pid', 'half'], how='left'
    )

    # Keep only contacts within [win_start, win_end]
    in_window = (
        merged['touch_frame'].notna()
        & (merged['touch_frame'] >= merged['_win_start'])
        & (merged['touch_frame'] <= merged['_win_end'])
    )
    touched_indices = merged.loc[in_window, '_idx'].unique()

    touch_mask = np.zeros(len(df_movements), dtype=bool)
    if len(touched_indices) > 0:
        touch_mask[touched_indices] = True
    df_movements['has_ball_touch'] = touch_mask

    n_touch = int(touch_mask.sum())
    n_total = len(df_movements)
    by_role_str = contacts[contacts['role'].isin(ACTIVE_TOUCH_ROLES)]['role'].value_counts().to_dict()
    print(f"[OK] ball touches tagged — {n_touch}/{n_total} movements.")
    print(f"     active-touch roles : {by_role_str}")
    print(f"     touch buffer (each side) : {TOUCH_BUFFER_FRAMES/framerate:.1f}s")

    # Drop temporary columns
    df_movements.drop(columns=['_pid', '_win_start', '_win_end', '_idx'],
                      errors='ignore', inplace=True)
    return df_movements


# ============================================================================
# Setpiece blackout
# Returns a dict of sorted np.ndarray (N×2) intervals instead of Python sets.
# Vectorised filter applied in process_match via _apply_blackout_mask.
# Each row is an interval [start (inclusive), end (exclusive)].
# ============================================================================

def build_setpiece_blackout_frames(path_events, path_info,
                                   framerate=25, anchors=None):
    """
    Return blackout intervals per half as a sorted np.ndarray (N, 2),
    compatible with np.searchsorted.
    """
    if anchors is None:
        anchors = _build_time_anchors(path_events, path_info, framerate)

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    ev_root   = ET.parse(path_events).getroot()
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
    # ------------------------------------------------------------------
    print("Computing time anchors…")
    anchors               = _build_time_anchors(path_events, path_info, framerate)
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
        contacts = _parse_contacts_xml_fallback(anchors, framerate, path_events=path_events)

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
    # 4. Filtered velocities
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

    # Vectorised timecode computation — replaces two lambda apply() calls
    def _compute_timecodes_vectorized(frames_arr, half_labels, kickoff_h1, kickoff_h2):
        """
        frames_arr  : np.ndarray (int)
        half_labels : pd.Series or np.ndarray of str
        """
        is_h1    = np.asarray(half_labels) == 'firstHalf'
        video_s  = np.where(
            is_h1,
            (frames_arr - kickoff_h1) / framerate,
            (frames_arr - kickoff_h2) / framerate + H2_BASE_SECONDS,
        )
        video_s  = np.maximum(video_s, 0.0)
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

            xy_f = butterworth_lowpass(
                xy[half][team], remove_short_seqs=True,
                Wn=butterworth_Wn, order=butterworth_order,
            ).xy

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
    for half, teams in xy_all.items():
        for team, xy_data in teams.items():
            xy_f = butterworth_lowpass(xy_data, remove_short_seqs=True,
                                       Wn=butterworth_Wn, order=butterworth_order)
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
    # 9. Tag ball touches (no re-parse)
    # ------------------------------------------------------------------
    df_movements = tag_ball_touches(
        df_movements, path_events, path_info,
        framerate=framerate,
        anchors=anchors,
        teamsheets=teamsheets,
        event_data=_events,
    )

    # ------------------------------------------------------------------
    # 10. Setpiece blackout exclusion — vectorised via searchsorted
    # ------------------------------------------------------------------
    blackout_intervals = build_setpiece_blackout_frames(
        path_events, path_info,
        framerate=framerate, anchors=anchors,
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

    print(f"[OK] {match_id} — {len(df_movements)} discrete movements extracted.")
    return df_movements, df_distances, dict_trajectories