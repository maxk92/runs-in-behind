"""
Discrete_efforts_Ilana.py
--------------------------
Processing module for DFL tracking data (single match).
Contains all helper functions and the main process_match() function
designed to be imported and called in a loop by Loop_Ilana.py.

Segmentation pipeline (hybrid approach)
----------------------------------------
Layer 1 — Velocity valleys   : detect major effort boundaries
                                (full stops, major speed drops)
Layer 2 — Metabolic power    : sub-segment within each valley-segment
                                using P = v * a to catch progressive
                                run→sprint transitions and direction changes
                                that produce no clear velocity valley
"""

import os
import xml.etree.ElementTree as ET
from datetime import timezone
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
# Possession smoothing
# ============================================================================

def rewrite_poss(group, cutoff=50):
    """
    Helper for smooth_possession: reassigns short possession phases (< cutoff frames)
    to the preceding possession team, then recomputes possession IDs and durations.

    Parameters
    ----------
    group   : pd.DataFrame — slice for one half, must contain 'possession' column
    cutoff  : int          — minimum number of frames a possession phase must last

    Returns
    -------
    pd.DataFrame with added columns: 'new' (smoothed possession),
                                     'newID' (smoothed possession ID),
                                     'dur'   (duration of each phase in frames)
    """
    poss = group['possession'].values.copy()
    n    = len(poss)

    # ── Pass 1: merge phases shorter than cutoff into the previous phase ──────
    i = 0
    while i < n:
        current_val = poss[i]
        j = i
        while j < n and poss[j] == current_val:
            j += 1
        phase_len = j - i
        if phase_len < cutoff and i > 0:
            poss[i:j] = poss[i - 1]   # absorb into preceding phase
        i = j

    # ── Pass 2: compute new possessionID and duration ─────────────────────────
    new_id  = np.zeros(n, dtype=int)
    dur_arr = np.zeros(n, dtype=int)

    current_id  = 1
    phase_start = 0
    for k in range(1, n):
        if poss[k] != poss[k - 1]:
            phase_len = k - phase_start
            new_id[phase_start:k]  = current_id
            dur_arr[phase_start:k] = phase_len
            current_id  += 1
            phase_start  = k
    # Last phase
    phase_len = n - phase_start
    new_id[phase_start:]  = current_id
    dur_arr[phase_start:] = phase_len

    group = group.copy()
    group['new']   = poss
    group['newID'] = new_id
    group['dur']   = dur_arr
    return group


def smooth_possession(possession_array, half_array, cutoff=50):
    """
    Smooth a possession signal by dropping phases shorter than `cutoff` frames.

    Parameters
    ----------
    possession_array : np.ndarray (n_frames,) — raw possession values (e.g. 1=Home, 2=Away)
    half_array       : np.ndarray (n_frames,) — half indicator (1 or 2) for each frame
    cutoff           : int — minimum phase length in frames (default 50 = 2 s at 25 fps)

    Returns
    -------
    smoothed   : np.ndarray (n_frames,) — smoothed possession values
    new_ids    : np.ndarray (n_frames,) — new sequential possession-phase IDs
    durations  : np.ndarray (n_frames,) — duration (frames) of each possession phase
    """
    possession_array = np.array(possession_array).flatten().astype(int)
    half_array       = np.array(half_array).flatten().astype(int)
    df = pd.DataFrame({'possession': possession_array, 'half': half_array})
    result = df.groupby('half', group_keys=False).apply(rewrite_poss, cutoff=cutoff)

    # Ensure possessionIDs are globally unique across halves
    mask_h2 = result['half'] == 2
    if mask_h2.any():
        max_id_h1 = result.loc[~mask_h2, 'newID'].max()
        result.loc[mask_h2, 'newID'] += max_id_h1

    return result['new'].values, result['newID'].values, result['dur'].values


# ============================================================================
# Layer 2 — Metabolic power sub-segmentation
# ============================================================================

def compute_metabolic_power(velocity_ms, framerate=25):
    """
    Compute approximate metabolic power P = v * a for each frame.

    Acceleration is estimated via central finite differences on the
    velocity signal (np.gradient), which is robust to NaN-free signals.
    NaNs are replaced with 0 before differentiation (player not tracked
    = no active effort).

    Parameters
    ----------
    velocity_ms : np.ndarray (n_frames,) — speed in m/s
    framerate   : int — acquisition frame rate (fps)

    Returns
    -------
    power : np.ndarray (n_frames,) — metabolic power in W/kg (approx)
    accel : np.ndarray (n_frames,) — acceleration in m/s²
    """
    dt  = 1.0 / framerate
    vel = np.where(np.isnan(velocity_ms), 0.0, velocity_ms)

    accel = np.gradient(vel, dt)          # m/s²
    power = vel * accel                   # W/kg (approx)

    return power, accel


def split_segment_by_power(segment_df, framerate=25,
                            power_threshold=6.0,
                            min_segment_frames=10):
    """
    Sub-segment a single movement DataFrame using metabolic power ruptures.

    A rupture is detected when:
      - the derivative of power exceeds `power_threshold` (W/kg/s), OR
      - the power signal changes sign (braking → acceleration or vice-versa)

    Only ruptures that separate segments of at least `min_segment_frames`
    are kept, preventing micro-fragments from sensor noise.

    Parameters
    ----------
    segment_df           : pd.DataFrame — one movement (output of valley layer)
                           must contain 'velocity' column
    framerate            : int
    power_threshold      : float — minimum |dP/dt| to trigger a split (W/kg/s)
                           Recommended: 3.0. Lower → more splits (more sensitive).
    min_segment_frames   : int — minimum length of a sub-segment in frames
                           At 25 fps: 10 frames = 0.4 s

    Returns
    -------
    list[pd.DataFrame] — sub-segments (may be a single-element list if no
                         rupture is detected)
    """
    v      = segment_df['velocity'].values
    power, _ = compute_metabolic_power(v, framerate=framerate)

    # Derivative of power (used to detect abrupt regime changes)
    dp = np.gradient(power)

    split_indices = [0]
    for i in range(min_segment_frames, len(power) - min_segment_frames):
        is_brutal   = abs(dp[i]) > power_threshold
        sign_change = (power[i - 1] * power[i] < 0)
        long_enough = (i - split_indices[-1]) >= min_segment_frames

        if (is_brutal or sign_change) and long_enough:
            split_indices.append(i)

    split_indices.append(len(segment_df))

    sub_segments = []
    for k in range(len(split_indices) - 1):
        seg = segment_df.iloc[split_indices[k]:split_indices[k + 1]].copy()
        if len(seg) >= 3:
            sub_segments.append(seg)

    return sub_segments


# ============================================================================
# Layer 1 — Velocity valley segmentation  +  pipeline assembly
# ============================================================================

def extract_movements_for_player(
        player_id, velocities, tracking_data, possession,
        # ── Layer 1: valley parameters ──────────────────────────────────────
        min_valley_distance = 15,
        valley_depth_abs    = 0.3,    # m/s  — minimum absolute speed drop
        valley_depth_rel    = 0.10,   # ratio — minimum relative speed drop
        # ── Layer 2: metabolic power parameters ─────────────────────────────
        power_threshold     = 6.0,    # W/kg/s — rupture sensitivity
        min_segment_frames  = 10,     # frames — minimum sub-segment length
        framerate           = 25,
):
    """
    Segment a player's speed signal into discrete movements.

    Pipeline
    --------
    1. Valley detection on the velocity signal → coarse boundaries
       (full stops, major speed drops)
    2. Metabolic power sub-segmentation within each valley-segment
       → catches progressive run→sprint transitions and direction changes
       that produce no clear velocity valley

    All speed categories are kept (walking included).
    Filtering by speed category is left to the caller.

    Parameters
    ----------
    player_id            : int   — column index in `velocities`
    velocities           : np.ndarray (n_frames, n_players) — speed in m/s
    tracking_data        : np.ndarray (n_frames, n_players*2) — x,y interleaved
    possession           : np.ndarray (n_frames,) — 1=Home, 2=Away
    min_valley_distance  : int   — minimum frames between two valleys (25 fps → 15 = 0.6 s)
    valley_depth_abs     : float — minimum absolute speed drop to accept a valley (m/s)
    valley_depth_rel     : float — minimum relative speed drop to accept a valley (ratio)
    power_threshold      : float — |dP/dt| threshold for a power rupture (W/kg/s)
    min_segment_frames   : int   — minimum sub-segment length after power split (frames)
    framerate            : int   — acquisition frame rate (fps)

    Returns
    -------
    list[pd.DataFrame] : one DataFrame per discrete movement
    """
    player_velocity = velocities[:, player_id]      # m/s

    # ── Layer 1a: detect raw valleys ─────────────────────────────────────────
    # Replace NaN with 0 for valley detection (NaN = player not tracked = stationary)
    vel_for_peaks = np.where(np.isnan(player_velocity), 0.0, player_velocity)

    raw_valleys, _ = find_peaks(-vel_for_peaks, distance=min_valley_distance)

    # ── Layer 1b: filter out shallow valleys ─────────────────────────────────
    significant_valleys = []
    for v in raw_valleys:
        valley_speed = vel_for_peaks[v]
        left       = max(0, v - min_valley_distance)
        right      = min(len(vel_for_peaks), v + min_valley_distance)
        local_peak = vel_for_peaks[left:right].max()

        abs_drop = local_peak - valley_speed
        rel_drop = abs_drop / local_peak if local_peak > 0 else 0.0

        if abs_drop >= valley_depth_abs and rel_drop >= valley_depth_rel:
            significant_valleys.append(v)

    valleys    = np.array(significant_valleys, dtype=int)
    boundaries = np.unique(
        np.concatenate([[0], valleys, [len(player_velocity) - 1]])
    )

    # ── Layer 1c: build coarse valley-segments ────────────────────────────────
    coarse_segments = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end   = int(boundaries[i + 1])

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
            'frame':      frames,
            'x':          x,
            'y':          y,
            'velocity':   v,
            'possession': p,
        }))

    # ── Layer 2: metabolic power sub-segmentation ─────────────────────────────
    movements = []
    mov_idx   = 0
    for seg in coarse_segments:
        sub = split_segment_by_power(
            seg,
            framerate          = framerate,
            power_threshold    = power_threshold,
            min_segment_frames = min_segment_frames,
        )
        for s in sub:
            s = s.copy()
            s['movement_id'] = mov_idx
            movements.append(s)
            mov_idx += 1

    return movements


def extract_discrete_movements_all_players(
        velocities, tracking_data, possession,
        min_valley_distance = 15,
        valley_depth_abs    = 0.3,
        valley_depth_rel    = 0.10,
        power_threshold     = 6.0,
        min_segment_frames  = 10,
        framerate           = 25,
):
    """
    Extract discrete movements for every player in a tracking data array.

    Parameters
    ----------
    velocities           : np.ndarray (n_frames, n_players)
    tracking_data        : np.ndarray (n_frames, n_players*2)
    possession           : np.ndarray (n_frames,)
    min_valley_distance  : int
    valley_depth_abs     : float
    valley_depth_rel     : float
    power_threshold      : float
    min_segment_frames   : int
    framerate            : int

    Returns
    -------
    dict {player_id: list[pd.DataFrame]}
    """
    n_players = velocities.shape[1]
    return {
        player_id: extract_movements_for_player(
            player_id, velocities, tracking_data, possession,
            min_valley_distance = min_valley_distance,
            valley_depth_abs    = valley_depth_abs,
            valley_depth_rel    = valley_depth_rel,
            power_threshold     = power_threshold,
            min_segment_frames  = min_segment_frames,
            framerate           = framerate,
        )
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
                'x_peak':           df['x'].iloc[peak_idx],
                'y_peak':           df['y'].iloc[peak_idx],
                'peak_speed_kmh':   peak_speed_kmh,
                'speed_category':   classify_speed_category(peak_speed_kmh),
                'avg_velocity_kmh': np.nanmean(v_ms) * 3.6,
                'distance_m':       distance_m,
                'duration_s':       len(df) / framerate,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Ball touch tagging
# ============================================================================

def parse_ball_contact_events(path_events, path_info, framerate=25):
    """
    Parse the DFL events XML and return a DataFrame of all events where a
    specific player can be identified as having physical contact with the ball.

    Events covered and how the player is extracted
    -----------------------------------------------
    Play (+ any wrapper: KickOff, ThrowIn, GoalKick, FreeKick, CornerKick)
        → Player (ball sender) + Recipient if present (ball receiver)
          covers: passes, crosses, shots from open play, set pieces

    OtherBallAction
        → Player  (control, clearance, headed touch, etc.)

    BallClaiming
        → Player  (interception, recovery)

    TacklingGame
        → Winner  if WinnerRole == 'withBallControl'  (won the duel WITH ball)
        → Loser   if LoserRole  == 'withBallControl'  (had the ball, lost duel)
        Both roles indicate the player physically possessed the ball.

    ShotAtGoal / SuccessfulShot / ShotWide / SavedShot / BlockedShot
        → Player  (the shooter)

    Foul
        → Fouled  (the player who had the ball and was fouled)

    Events deliberately excluded
    ----------------------------
    Delete, Substitution, Offside, Caution, VideoAssistantAction, etc.
    — these carry no ball-contact information.

    Timezone handling
    -----------------
    EventTime uses local time (+02:00 CEST); KickoffTime uses UTC (+00:00).
    Both are normalised to UTC before computing elapsed time → frame number,
    which fully neutralises the apparent 2-hour offset.

    Parameters
    ----------
    path_events  : str — path to the DFL events XML file
    path_info    : str — path to the DFL matchinformation XML file
    framerate    : int — acquisition frame rate (default 25 fps)

    Returns
    -------
    pd.DataFrame with columns:
        person_id    : str  — DFL PersonId of the player with ball contact
        role         : str  — semantic role (see mapping below)
        game_section : str  — 'firstHalf' or 'secondHalf'
        abs_frame    : int  — frame number relative to kickoff of that half
        event_tag    : str  — XML tag of the event element (e.g. 'Play', 'TacklingGame')
    """

    # ── Role labels ───────────────────────────────────────────────────────────
    # Kept descriptive to allow downstream filtering if needed.
    ROLE_PASSER       = 'passer'
    ROLE_RECIPIENT    = 'recipient'
    ROLE_BALL_ACTION  = 'ball_action'    # OtherBallAction
    ROLE_BALL_CLAIM   = 'ball_claim'     # BallClaiming (interception/recovery)
    ROLE_DUEL_WINNER  = 'duel_winner'    # TacklingGame winner with ball
    ROLE_DUEL_LOSER   = 'duel_loser'     # TacklingGame loser with ball
    ROLE_SHOOTER      = 'shooter'
    ROLE_FOULED       = 'fouled'         # had the ball, was fouled

    # Tags that wrap a <Play> child (set pieces / restarts)
    PLAY_WRAPPERS = {'KickOff', 'ThrowIn', 'GoalKick', 'FreeKick',
                     'CornerKick', 'RefereeBall'}

    # Shot-type tags that carry a Player attribute directly on the child
    SHOT_TAGS = {'ShotAtGoal', 'SuccessfulShot', 'ShotWide',
                 'SavedShot', 'BlockedShot', 'ShotWoodWork',
                 'ChanceWithoutShot'}

    # ── 1. Kickoff time (UTC) from matchinformation ───────────────────────────
    info_root   = ET.parse(path_info).getroot()
    kickoff_str = info_root.find('.//General').attrib['KickoffTime']
    kickoff_utc = pd.Timestamp(kickoff_str).tz_convert('UTC')

    # ── 2. Second-half kickoff: needed to assign game_section correctly.
    #    We derive it from TotalTimeFirstHalf (milliseconds) in matchinformation.
    other_info = info_root.find('.//OtherGameInformation')
    if other_info is not None and 'TotalTimeFirstHalf' in other_info.attrib:
        first_half_ms  = int(other_info.attrib['TotalTimeFirstHalf'])
        second_half_start_utc = kickoff_utc + pd.Timedelta(milliseconds=first_half_ms)
    else:
        # Fallback: assume 50 minutes for the first half (incl. stoppage)
        second_half_start_utc = kickoff_utc + pd.Timedelta(minutes=50)

    # ── 3. Parse events XML ───────────────────────────────────────────────────
    ev_root = ET.parse(path_events).getroot()

    rows = []

    def add(person_id, role, game_section, abs_frame, event_tag):
        """Append a contact record if person_id and game_section are known."""
        if person_id and game_section:
            rows.append({
                'person_id':    person_id,
                'role':         role,
                'game_section': game_section,
                'abs_frame':    abs_frame,
                'event_tag':    event_tag,
            })

    def elapsed_to_frame_and_section(event_utc):
        """
        Return (game_section, abs_frame) for a given UTC timestamp.
        abs_frame is relative to the start of the respective half.
        Returns (None, None) if the event is before kickoff.
        """
        elapsed_s = (event_utc - kickoff_utc).total_seconds()
        if elapsed_s < 0:
            return None, None   # pre-match artefact

        if event_utc < second_half_start_utc:
            gs    = 'firstHalf'
            frame = int(round(elapsed_s * framerate))
        else:
            gs    = 'secondHalf'
            elapsed_2nd = (event_utc - second_half_start_utc).total_seconds()
            frame = int(round(elapsed_2nd * framerate))

        return gs, frame

    for event in ev_root.findall('Event'):
        event_time_str = event.attrib.get('EventTime')
        if event_time_str is None:
            continue

        event_utc         = pd.Timestamp(event_time_str).tz_convert('UTC')
        game_section, frame = elapsed_to_frame_and_section(event_utc)
        if game_section is None:
            continue

        for child in event:
            tag = child.tag

            # ── Play (direct or wrapped inside a set-piece element) ───────────
            plays = []
            if tag == 'Play':
                plays = [child]
            elif tag in PLAY_WRAPPERS:
                plays = child.findall('Play')

            for play in plays:
                player    = play.attrib.get('Player')
                recipient = play.attrib.get('Recipient')
                add(player,    ROLE_PASSER,    game_section, frame, tag)
                add(recipient, ROLE_RECIPIENT, game_section, frame, tag)

            # ── OtherBallAction ───────────────────────────────────────────────
            if tag == 'OtherBallAction':
                add(child.attrib.get('Player'), ROLE_BALL_ACTION,
                    game_section, frame, tag)

            # ── BallClaiming (interception, recovery) ─────────────────────────
            elif tag == 'BallClaiming':
                add(child.attrib.get('Player'), ROLE_BALL_CLAIM,
                    game_section, frame, tag)

            # ── TacklingGame ──────────────────────────────────────────────────
            elif tag == 'TacklingGame':
                winner_role = child.attrib.get('WinnerRole', '')
                loser_role  = child.attrib.get('LoserRole',  '')
                if winner_role == 'withBallControl':
                    add(child.attrib.get('Winner'), ROLE_DUEL_WINNER,
                        game_section, frame, tag)
                if loser_role == 'withBallControl':
                    add(child.attrib.get('Loser'), ROLE_DUEL_LOSER,
                        game_section, frame, tag)

            # ── Shots ─────────────────────────────────────────────────────────
            elif tag in SHOT_TAGS:
                add(child.attrib.get('Player'), ROLE_SHOOTER,
                    game_section, frame, tag)

            # ── Foul: the fouled player had the ball ──────────────────────────
            elif tag == 'Foul':
                add(child.attrib.get('Fouled'), ROLE_FOULED,
                    game_section, frame, tag)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Remove duplicates that can arise from nested Play elements
        df = df.drop_duplicates(
            subset=['person_id', 'game_section', 'abs_frame', 'role']
        ).reset_index(drop=True)

    return df


def build_shortname_to_person_id(path_info):
    """
    Build a mapping {Shortname → PersonId} from the DFL matchinformation XML.

    Floodlight stores the DFL Shortname (e.g. 'F. Kainz') in the 'player'
    column of df_movements, while events use the PersonId (e.g. 'DFL-OBJ-0027AX').
    This mapping bridges the two.

    Parameters
    ----------
    path_info : str — path to the DFL matchinformation XML file

    Returns
    -------
    dict {shortname: person_id}
    """
    root    = ET.parse(path_info).getroot()
    mapping = {}
    for player in root.findall('.//Player'):
        shortname = player.attrib.get('Shortname')
        person_id = player.attrib.get('PersonId')
        if shortname and person_id:
            mapping[shortname] = person_id
    return mapping


def build_possession_intervals(contacts):
    """
    From a contacts DataFrame, reconstruct continuous ball-possession intervals
    per player: a player holds the ball from the frame he receives/recovers it
    until the next event occurs globally (regardless of who it belongs to).

    The reasoning: events are logged at the moment of each ball contact.
    Between two consecutive events, the ball belongs to the player identified
    in the first event. The next event (any player) marks the end of that
    possession.

    Parameters
    ----------
    contacts : pd.DataFrame — output of parse_ball_contact_events()

    Returns
    -------
    dict { game_section → list of (person_id, frame_start, frame_end) }
    """
    # Roles that GIVE the ball to the named player (they become the carrier)
    CARRIER_ROLES = {'recipient', 'ball_action', 'ball_claim',
                     'duel_winner', 'duel_loser', 'shooter', 'fouled'}

    intervals = {}

    for gs, group in contacts.groupby('game_section'):
        # Sort all events by frame within this half
        group_sorted = group.sort_values('abs_frame').reset_index(drop=True)

        # All distinct frame timestamps (= moments where possession may change)
        all_frames = group_sorted['abs_frame'].values

        possession_intervals = []
        for i, row in group_sorted.iterrows():
            if row['role'] not in CARRIER_ROLES:
                continue

            frame_start = row['abs_frame']

            # Possession ends at the next event after this one
            later = all_frames[all_frames > frame_start]
            frame_end = int(later[0]) if len(later) > 0 else frame_start + 250  # fallback ~10s

            possession_intervals.append(
                (row['person_id'], frame_start, frame_end)
            )

        intervals[gs] = possession_intervals

    return intervals


def tag_ball_touches(df_movements, path_events, path_info,
                     framerate=25):
    """
    Add a boolean column 'has_ball_touch' to df_movements.

    A movement is tagged True if the player was carrying the ball at any
    point in [start_frame, end_frame - 25 frames].

    Possession model: a player carries the ball from the frame he receives
    or recovers it (recipient, ball_action, ball_claim, duel_winner/loser,
    shooter, fouled) until the very next ball-contact event in the match
    (regardless of who it is). This means we don't just check isolated event
    frames — we check entire possession intervals.

    The last 25 frames (1 second at 25 fps) of each movement are excluded
    from the search window: a ball contact in that window is interpreted as
    the player receiving the ball at the END of his run (arrival), not as
    him carrying it during the run.

    Player matching: df_movements['player'] holds the DFL Shortname
    (e.g. 'F. Kainz'); events use PersonId (e.g. 'DFL-OBJ-0027AX').
    The matchinformation file bridges the two.

    Parameters
    ----------
    df_movements : pd.DataFrame — output of process_match()
    path_events  : str          — path to the DFL events XML file
    path_info    : str          — path to the DFL matchinformation XML file
    framerate    : int          — frame rate (default 25 fps)

    Returns
    -------
    pd.DataFrame — df_movements with added column 'has_ball_touch' (bool)
    """
    contacts = parse_ball_contact_events(path_events, path_info, framerate=framerate)

    if contacts.empty:
        print("[WARN] No ball contact events parsed — 'has_ball_touch' set to False.")
        df_movements['has_ball_touch'] = False
        return df_movements

    # ── Shortname → PersonId mapping ─────────────────────────────────────────
    shortname_to_pid = build_shortname_to_person_id(path_info)
    unknown = set(df_movements['player'].unique()) - set(shortname_to_pid.keys())
    if unknown:
        print(f"[WARN] {len(unknown)} player name(s) not found in matchinformation: {unknown}")

    # ── Build possession intervals per half ───────────────────────────────────
    possession_intervals = build_possession_intervals(contacts)

    # Index intervals by (person_id, game_section) for fast lookup
    # Structure: { (person_id, game_section): [(frame_start, frame_end), ...] }
    interval_index = {}
    for gs, intervals in possession_intervals.items():
        for person_id, f_start, f_end in intervals:
            key = (person_id, gs)
            interval_index.setdefault(key, []).append((f_start, f_end))

    # ── Tag each movement ─────────────────────────────────────────────────────
    def movement_has_touch(row):
        gs  = row['half']
        pid = shortname_to_pid.get(row['player'])
        if pid is None:
            return False

        # Search window: [start_frame, end_frame - 25 frames]
        # The last second is excluded: a touch there means arrival, not carrying.
        win_start = int(row['start_frame'])
        win_end   = int(row['end_frame']) - framerate
        if win_end < win_start:
            return False  # movement shorter than 1 second — nothing to check

        key = (pid, gs)
        for (f_start, f_end) in interval_index.get(key, []):
            # Overlap between [win_start, win_end] and [f_start, f_end]
            if f_start <= win_end and f_end >= win_start:
                return True
        return False

    df_movements['has_ball_touch'] = df_movements.apply(movement_has_touch, axis=1)

    n_touch = df_movements['has_ball_touch'].sum()
    n_total = len(df_movements)
    by_role = contacts['role'].value_counts().to_dict()
    print(f"[OK] ball touches tagged — {n_touch}/{n_total} movements contain a ball possession.")
    print(f"     contacts by role: {by_role}")
    return df_movements


# ============================================================================
# Main pipeline function
# ============================================================================

def process_match(
        match_id, DATA_DIR, dict_direction,
        # ── Signal filtering ─────────────────────────────────────────────────
        butterworth_Wn    = 0.5,
        butterworth_order = 1,
        # ── Layer 1: valley segmentation ─────────────────────────────────────
        min_valley_distance = 15,     # frames — min distance between valleys
        valley_depth_abs    = 0.3,    # m/s    — min absolute speed drop
        valley_depth_rel    = 0.10,   # ratio  — min relative speed drop
        # ── Layer 2: metabolic power sub-segmentation ─────────────────────────
        power_threshold    = 6.0,     # W/kg/s — rupture sensitivity
        min_segment_frames = 10,      # frames — min sub-segment length (0.4 s)
        # ── General ──────────────────────────────────────────────────────────
        framerate        = 25,
        speed_thresholds = None,
):
    """
    Full processing pipeline for a single match:
    loading → filtering → discrete movements → distances → playing direction.

    Segmentation uses a 2-layer hybrid approach:
      Layer 1 — velocity valleys   (coarse boundaries: full stops, major speed drops)
      Layer 2 — metabolic power    (fine sub-segmentation: run→sprint, direction changes)

    All speed categories are preserved in the output. Filtering by speed
    category (e.g. keeping only jogging/running/sprinting) is done in
    Loop_Ilana.py using the 'speed_category' column.

    Parameters
    ----------
    match_id            : str   — match identifier (used to filter filenames)
    DATA_DIR            : str   — directory containing DFL data files
    dict_direction      : dict  — playing directions loaded from JSON
                          format: {match_id: {"Home": {"firstHalf": "left_to_right", ...}}}
    butterworth_Wn      : float — cutoff frequency for the low-pass filter
    butterworth_order   : int   — Butterworth filter order
    min_valley_distance : int   — minimum frames between two valleys
    valley_depth_abs    : float — minimum absolute speed drop to keep a valley (m/s)
    valley_depth_rel    : float — minimum relative speed drop to keep a valley (ratio)
    power_threshold     : float — metabolic power derivative threshold for a split (W/kg/s)
    min_segment_frames  : int   — minimum sub-segment length after power split (frames)
    framerate           : int   — acquisition frame rate (fps)
    speed_thresholds    : dict  — speed category thresholds (falls back to SPEED_THRESHOLDS)

    Returns
    -------
    df_movements      : pd.DataFrame  — one discrete movement per row (all speed categories)
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
    # 4. Smooth possession signal (drop phases shorter than cutoff)
    # ------------------------------------------------------------------
    possession_smoothed = {}
    for half_name, half_num in [('firstHalf', 1), ('secondHalf', 2)]:
        raw_poss = np.array(possession[half_name]).flatten().astype(int)
        half_arr = np.full(len(raw_poss), half_num, dtype=int)
        smoothed, _, _ = smooth_possession(raw_poss, half_arr, cutoff=50)
        possession_smoothed[half_name] = smoothed

    # ------------------------------------------------------------------
    # 5. Extract discrete movements (3-layer pipeline)
    # ------------------------------------------------------------------
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
                possession_smoothed[half],
                min_valley_distance = min_valley_distance,
                valley_depth_abs    = valley_depth_abs,
                valley_depth_rel    = valley_depth_rel,
                power_threshold     = power_threshold,
                min_segment_frames  = min_segment_frames,
                framerate           = framerate,
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
    # 6. Total distance covered per player
    # ------------------------------------------------------------------
    ls_dfs_distance = []

    for half, teams in xy_all.items():
        for team, xy_data in teams.items():
            xy_f = butterworth_lowpass(xy_data, remove_short_seqs=True,
                                       Wn=butterworth_Wn, order=butterworth_order)
            dm = DistanceModel()
            dm.fit(xy_f)

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
    # 7. Playing direction
    # ------------------------------------------------------------------
    if match_id not in dict_direction:
        print(f"[WARN] '{match_id}' not found in direction JSON — "
              f"'direction' column will be NaN.")

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

    df_movements["direction"]    = base_direction * multiplier
    df_movements["attack_sign"]  = multiplier
    df_movements["match_id"]     = match_id

    # ------------------------------------------------------------------
    # 8. Tag ball touches (pass sent or received during the movement)
    # ------------------------------------------------------------------
    df_movements = tag_ball_touches(
        df_movements, path_events, path_info, framerate=framerate
    )

    print(f"[OK] {match_id} — {len(df_movements)} discrete movements extracted.")
    return df_movements, df_distances, dict_trajectories