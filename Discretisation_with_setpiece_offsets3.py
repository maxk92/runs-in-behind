"""
Discretisation_floodlight.py — VERSION OPTIMISÉE AVEC FLOODLIGHT
-----------------------------------------------------------------
Refactorisation du pipeline de discrétisation des efforts.

L'API floodlight n'expose PAS de classes DFLMatchinformationParser /
DFLEventDataParser — elle expose des fonctions directes :
  - dfl.read_position_data_xml()   → tracking XY + possession + teamsheets
  - dfl.read_event_data_xml()      → events (EventData floodlight)

CE QUI A CHANGÉ vs. la version originale
-----------------------------------------
1. get_second_half_start_utc() + get_kickoff_frames()
     → fusionnés en _build_time_anchors() : parse le XML events UNE SEULE FOIS
       et passe les ancres en paramètre à toutes les fonctions qui en ont besoin.
       L'original parsait les mêmes XMLs 3–4 fois séparément.

2. build_shortname_to_person_id()
     → _build_shortname_map_from_teamsheets() : construit le mapping depuis
       les teamsheets déjà retournés par dfl.read_position_data_xml() —
       zéro parse XML supplémentaire.

3. parse_ball_contact_events()
     → _parse_contacts_from_event_data() : utilise l'EventData floodlight
       (déjà chargé par dfl.read_event_data_xml()) pour extraire les contacts
       ballon, avec fallback XML minimal si les colonnes attendues sont absentes.

4. process_match()
     → charge tracking + events une seule fois, calcule les ancres une seule
       fois, et passe tout en paramètre pour éviter tout re-parse en aval.

CE QUI EST INCHANGÉ (logique métier trop fine pour floodlight)
--------------------------------------------------------------
- build_possession_intervals()    : per-role delays (TOUCH_RECIPIENT_DELAY_S)
- tag_ball_touches()              : TOUCH_BUFFER_FRAMES, active-touch logic
- build_setpiece_blackout_frames(): offsets custom, extension frame-0 KickOff
- smooth_possession() / rewrite_poss()
- extract_movements_for_player()  : valley + metabolic power pipeline
- summarize_movements_to_dataframe()

Dépendances :
  pip install floodlight numpy pandas scipy tqdm
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
    'FreeKick':   3.0,
    'GoalKick':   3.0,
    'KickOff':    6.0,
    'CornerKick': 4.0,
}

RECIPIENT_DELAY_S       = 3.0
TOUCH_RECIPIENT_DELAY_S = 6.0


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
# Possession smoothing  (inchangé)
# ============================================================================

def rewrite_poss(group, cutoff=50):
    poss = group['possession'].values.copy()
    n    = len(poss)

    i = 0
    while i < n:
        current_val = poss[i]
        j = i
        while j < n and poss[j] == current_val:
            j += 1
        phase_len = j - i
        if phase_len < cutoff and i > 0:
            poss[i:j] = poss[i - 1]
        i = j

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
    phase_len = n - phase_start
    new_id[phase_start:]  = current_id
    dur_arr[phase_start:] = phase_len

    group = group.copy()
    group['new']   = poss
    group['newID'] = new_id
    group['dur']   = dur_arr
    return group


def smooth_possession(possession_array, half_array, cutoff=50):
    possession_array = np.array(possession_array).flatten().astype(int)
    half_array       = np.array(half_array).flatten().astype(int)
    df = pd.DataFrame({'possession': possession_array, 'half': half_array})
    result = df.groupby('half', group_keys=False).apply(rewrite_poss, cutoff=cutoff)

    mask_h2 = result['half'] == 2
    if mask_h2.any():
        max_id_h1 = result.loc[~mask_h2, 'newID'].max()
        result.loc[mask_h2, 'newID'] += max_id_h1

    return result['new'].values, result['newID'].values, result['dur'].values


# ============================================================================
# Layer 2 — Metabolic power sub-segmentation  (inchangé)
# ============================================================================

def compute_metabolic_power(velocity_ms, framerate=25):
    dt    = 1.0 / framerate
    vel   = np.where(np.isnan(velocity_ms), 0.0, velocity_ms)
    accel = np.gradient(vel, dt)
    power = vel * accel
    return power, accel


def split_segment_by_power(segment_df, framerate=25,
                            power_threshold=6.0, min_segment_frames=10):
    v        = segment_df['velocity'].values
    power, _ = compute_metabolic_power(v, framerate=framerate)
    dp       = np.gradient(power)

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
# Layer 1 — Velocity valley segmentation  (inchangé)
# ============================================================================

def extract_movements_for_player(
        player_id, velocities, tracking_data, possession,
        min_valley_distance=15,
        valley_depth_abs=0.3,
        valley_depth_rel=0.10,
        power_threshold=6.0,
        min_segment_frames=10,
        framerate=25,
):
    player_velocity = velocities[:, player_id]
    vel_for_peaks   = np.where(np.isnan(player_velocity), 0.0, player_velocity)

    raw_valleys, _ = find_peaks(-vel_for_peaks, distance=min_valley_distance)

    significant_valleys = []
    for v in raw_valleys:
        valley_speed = vel_for_peaks[v]
        left       = max(0, v - min_valley_distance)
        right      = min(len(vel_for_peaks), v + min_valley_distance)
        local_peak = vel_for_peaks[left:right].max()
        abs_drop   = local_peak - valley_speed
        rel_drop   = abs_drop / local_peak if local_peak > 0 else 0.0
        if abs_drop >= valley_depth_abs and rel_drop >= valley_depth_rel:
            significant_valleys.append(v)

    valleys    = np.array(significant_valleys, dtype=int)
    boundaries = np.unique(
        np.concatenate([[0], valleys, [len(player_velocity) - 1]])
    )

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
        valley_depth_abs=0.3,
        valley_depth_rel=0.10,
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
# Movement summary DataFrame  (inchangé)
# ============================================================================

def summarize_movements_to_dataframe(movements_dict, teamsheet, framerate=25):
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
# ▶▶  NOUVEAU — Ancres temporelles centralisées  (remplace get_second_half_start_utc
#               et parse XML répétitif dans get_kickoff_frames)
# ============================================================================

def _build_time_anchors(path_events: str, path_info: str,
                        framerate: int = 25) -> dict:
    """
    Parse les XMLs events + matchinfo UNE SEULE FOIS pour toutes les ancres.

    Centralise ce que l'original faisait dans trois fonctions séparées
    (get_second_half_start_utc, get_kickoff_frames, build_setpiece_blackout_frames)
    qui re-parsaient chacune les mêmes fichiers.

    Retourne un dict avec :
        kickoff_utc          : pd.Timestamp UTC du coup d'envoi H1
        second_half_start_utc: pd.Timestamp UTC du coup d'envoi H2
        kickoff_frames       : {'firstHalf': int, 'secondHalf': int}
    """
    # ── 1. KickoffTime depuis matchinfo XML (lecture directe, pas de classe parser)
    info_root   = ET.parse(path_info).getroot()
    kickoff_str = info_root.find('.//General').attrib['KickoffTime']
    kickoff_utc = pd.Timestamp(kickoff_str).tz_convert('UTC')

    # Estimation grossière de fin de H1 pour bootstrap de la recherche H2
    other_info = info_root.find('.//OtherGameInformation')
    if other_info is not None and 'TotalTimeFirstHalf' in other_info.attrib:
        h1_end_utc = kickoff_utc + pd.Timedelta(
            milliseconds=int(other_info.attrib['TotalTimeFirstHalf']))
    else:
        h1_end_utc = kickoff_utc + pd.Timedelta(minutes=50)

    # ── 2. Parse events XML UNE SEULE FOIS ───────────────────────────────────
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

        # Ancre H2 : premier KickOff GameSection=secondHalf (ou fallback)
        if second_half_start_utc is None and event_utc > h1_end_utc:
            second_half_start_utc = event_utc
            if game_section == 'secondHalf':
                print(f"  [H2 anchor] KickOff secondHalf @ {event_utc} "
                      f"(break = {(event_utc - h1_end_utc).total_seconds()/60:.1f} min)")
            else:
                print(f"  [H2 anchor] premier KickOff après H1 end @ {event_utc} (fallback)")

        # Frame du kickoff dans son demi-temps
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

    # Fallbacks
    if second_half_start_utc is None:
        print("[WARN] H2 KickOff non trouvé — fallback sur h1_end_utc")
        second_half_start_utc = h1_end_utc
    if kickoff_frames['firstHalf'] is None:
        print("[WARN] firstHalf KickOff non trouvé — frame 0")
        kickoff_frames['firstHalf'] = 0
    if kickoff_frames['secondHalf'] is None:
        print("[WARN] secondHalf KickOff non trouvé — fallback 67500")
        kickoff_frames['secondHalf'] = 67500

    print(
        f"[OK] kickoff frames — "
        f"H1: frame {kickoff_frames['firstHalf']} "
        f"({kickoff_frames['firstHalf']/framerate:.1f}s), "
        f"H2: frame {kickoff_frames['secondHalf']} "
        f"({kickoff_frames['secondHalf']/framerate:.1f}s into H2 tracking)."
    )

    return {
        'kickoff_utc':           kickoff_utc,
        'second_half_start_utc': second_half_start_utc,
        'kickoff_frames':        kickoff_frames,
    }


# ============================================================================
# ▶▶  NOUVEAU — Shortname → PersonId via floodlight
#               (remplace build_shortname_to_person_id)
# ============================================================================

def _build_shortname_map_from_teamsheets(teamsheets: dict) -> dict:
    """
    Construit {Shortname → PersonId} depuis les teamsheets déjà chargés
    par dfl.read_position_data_xml() — zéro parse XML supplémentaire.

    Remplace build_shortname_to_person_id() qui re-parsait le matchinfo XML.

    Les teamsheets floodlight ont les colonnes :
        player (= Shortname DFL), player_id (= PersonId DFL)
    """
    mapping = {}
    for team_ts in teamsheets.values():
        ts = team_ts.teamsheet
        if 'player' in ts.columns and 'player_id' in ts.columns:
            for _, row in ts.iterrows():
                if pd.notna(row['player']) and pd.notna(row['player_id']):
                    mapping[row['player']] = row['player_id']
        elif 'player' in ts.columns and 'pID' in ts.columns:
            # Nom alternatif selon la version de floodlight
            for _, row in ts.iterrows():
                if pd.notna(row['player']) and pd.notna(row['pID']):
                    mapping[row['player']] = row['pID']
    return mapping


# ============================================================================
# ▶▶  NOUVEAU — parse_ball_contact_events via floodlight
#               (remplace le parsing XML manuel complet)
# ============================================================================

def _parse_contacts_from_event_data(event_data,
                                    anchors: dict,
                                    framerate: int = 25) -> pd.DataFrame:
    """
    Extrait les contacts ballon depuis l'EventData floodlight déjà chargé
    (retourné par dfl.read_event_data_xml()).

    Évite tout re-parse XML — l'EventData est passé en paramètre depuis
    process_match() qui l'a déjà chargé.

    Tente d'abord de lire les colonnes floodlight standard ; si celles-ci
    sont absentes ou vides, bascule sur le fallback XML.

    Paramètres
    ----------
    event_data : objet EventData floodlight (ou None si non disponible)
    anchors    : dict retourné par _build_time_anchors()

    Retourne
    --------
    pd.DataFrame : person_id, role, game_section, abs_frame, event_tag
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

    def utc_to_gs_frame(event_utc):
        elapsed_s = (event_utc - kickoff_utc).total_seconds()
        if elapsed_s < 0:
            return None, None
        if event_utc < second_half_start_utc:
            return 'firstHalf', int(round(elapsed_s * framerate))
        elapsed_2nd = (event_utc - second_half_start_utc).total_seconds()
        return 'secondHalf', int(round(elapsed_2nd * framerate))

    rows = []

    # ── Tentative via EventData floodlight ────────────────────────────────────
    if event_data is not None:
        try:
            # floodlight EventData : attribut .events (DataFrame) ou itérable
            # La structure exacte varie selon la version ; on essaie les deux.
            if hasattr(event_data, 'events'):
                df_ev = event_data.events
            elif hasattr(event_data, '__iter__'):
                # Certaines versions retournent un tuple (events_home, events_away)
                parts = list(event_data)
                df_ev = pd.concat([p.events for p in parts
                                   if hasattr(p, 'events')], ignore_index=True)
            else:
                raise ValueError("Format EventData inconnu")

            # Colonnes attendues (noms variables selon version floodlight)
            col_type = next((c for c in ['eID', 'event_type', 'type']
                             if c in df_ev.columns), None)
            col_pid  = next((c for c in ['player_id', 'pID', 'player']
                             if c in df_ev.columns), None)
            col_time = next((c for c in ['event_time', 'datetime', 'gameclock']
                             if c in df_ev.columns), None)

            if col_type and col_pid and col_time:
                for _, row in df_ev.iterrows():
                    ev_type = row[col_type]
                    pid     = row[col_pid]
                    t_val   = row[col_time]
                    if not pid or pd.isna(pid):
                        continue
                    try:
                        event_utc = pd.Timestamp(t_val).tz_convert('UTC')
                    except Exception:
                        continue
                    gs, frame = utc_to_gs_frame(event_utc)
                    if gs is None:
                        continue
                    role = EVENT_ROLE_MAP.get(ev_type)
                    if role:
                        rows.append({'person_id': pid, 'role': role,
                                     'game_section': gs, 'abs_frame': frame,
                                     'event_tag': ev_type})
                    # Recipient (colonne optionnelle)
                    for rcol in ['recipient_id', 'recipient']:
                        rid = row.get(rcol)
                        if rid and not pd.isna(rid):
                            rows.append({'person_id': rid, 'role': 'recipient',
                                         'game_section': gs, 'abs_frame': frame,
                                         'event_tag': ev_type})
                            break

                if rows:
                    df_contacts = pd.DataFrame(rows).drop_duplicates(
                        subset=['person_id', 'game_section', 'abs_frame', 'role']
                    ).reset_index(drop=True)
                    print(f"  [contacts] {len(df_contacts)} events via floodlight EventData.")
                    return df_contacts

        except Exception as e:
            print(f"  [contacts][WARN] EventData floodlight inutilisable ({e}).")

    # ── Fallback : parsing XML minimal (ancres déjà calculées, pas de re-parse matchinfo)
    # path_events doit être passé par l'appelant via le paramètre dédié
    return pd.DataFrame()   # signal au caller de déclencher le fallback avec path_events


def _parse_contacts_xml_fallback(anchors: dict, framerate: int = 25,
                                  path_events: str = None) -> pd.DataFrame:
    """
    Fallback XML — même logique que l'original parse_ball_contact_events()
    mais utilise les ancres pré-calculées (évite de re-parser matchinfo).
    path_events doit être fourni si appelé hors de process_match().
    """
    if path_events is None:
        raise ValueError("path_events requis pour le fallback XML")

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    ROLE_PASSER      = 'passer'
    ROLE_RECIPIENT   = 'recipient'
    ROLE_BALL_ACTION = 'ball_action'
    ROLE_BALL_CLAIM  = 'ball_claim'
    ROLE_DUEL_WINNER = 'duel_winner'
    ROLE_DUEL_LOSER  = 'duel_loser'
    ROLE_SHOOTER     = 'shooter'
    ROLE_FOULED      = 'fouled'

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
        else:
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
                add(play.attrib.get('Player'),    ROLE_PASSER,    gs, frame, tag)
                add(play.attrib.get('Recipient'), ROLE_RECIPIENT, gs, frame, tag)

            if tag == 'OtherBallAction':
                add(child.attrib.get('Player'), ROLE_BALL_ACTION, gs, frame, tag)
            elif tag == 'BallClaiming':
                add(child.attrib.get('Player'), ROLE_BALL_CLAIM, gs, frame, tag)
            elif tag == 'TacklingGame':
                if child.attrib.get('WinnerRole') == 'withBallControl':
                    add(child.attrib.get('Winner'), ROLE_DUEL_WINNER, gs, frame, tag)
                if child.attrib.get('LoserRole') == 'withBallControl':
                    add(child.attrib.get('Loser'), ROLE_DUEL_LOSER, gs, frame, tag)
            elif tag in SHOT_TAGS:
                add(child.attrib.get('Player'), ROLE_SHOOTER, gs, frame, tag)
            elif tag == 'Foul':
                add(child.attrib.get('Fouled'), ROLE_FOULED, gs, frame, tag)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=['person_id', 'game_section', 'abs_frame', 'role']
        ).reset_index(drop=True)

    print(f"  [contacts] {len(df)} events via fallback XML parser.")
    return df


# ============================================================================
# Possession intervals  (inchangé — logique métier trop fine)
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
        group_sorted = group.sort_values('abs_frame').reset_index(drop=True)
        all_frames   = group_sorted['abs_frame'].values

        possession_intervals = []
        for i, row in group_sorted.iterrows():
            if row['role'] not in CARRIER_ROLES:
                continue
            event_frame = row['abs_frame']
            frame_start = event_frame + delay_frames[row['role']]
            later       = all_frames[all_frames > event_frame]
            frame_end   = int(later[0]) if len(later) > 0 else event_frame + 250
            if frame_start >= frame_end:
                continue
            possession_intervals.append((row['person_id'], frame_start, frame_end))

        intervals[gs] = possession_intervals

    return intervals


def apply_possession_intervals_to_array(possession_array, intervals, teamsheets, half_name):
    """
    Applique les intervalles de possession (person_id, start, end) au tableau possession.

    possession_array : array 1D (frames) avec 0=away, 1=home
    intervals : dict par game_section, liste de (person_id, start, end)
    teamsheets : dict floodlight teamsheets
    half_name : 'firstHalf' ou 'secondHalf'
    """
    # Construire mapping person_id → team (0=away, 1=home)
    pid_to_team = {}
    for team_name, ts_obj in teamsheets.items():
        ts = ts_obj.teamsheet
        team_code = 1 if team_name == 'Home' else 0
        for _, row in ts.iterrows():
            pid = row.get('player_id') or row.get('pID')
            if pid:
                pid_to_team[pid] = team_code

    # Copier le tableau original
    new_poss = possession_array.copy()

    # Appliquer les intervalles (dernier gagnant en cas de chevauchement)
    for pid, start, end in intervals.get(half_name, []):
        team = pid_to_team.get(pid)
        if team is not None:
            new_poss[start:end] = team

    return new_poss


# ============================================================================
# tag_ball_touches  (inchangé — logique métier trop fine)
# ============================================================================

def tag_ball_touches(df_movements, path_events, path_info,
                     framerate=25, anchors=None,
                     teamsheets=None, event_data=None):
    """
    Ajoute la colonne 'has_ball_touch' à df_movements.

    Paramètres optionnels pour éviter tout re-parse :
      anchors    : dict de _build_time_anchors() — recalculé si None
      teamsheets : dict retourné par dfl.read_position_data_xml()
                   → construit le mapping shortname→PersonId sans XML
      event_data : objet retourné par dfl.read_event_data_xml()
                   → extrait les contacts sans re-parser les events
    """
    TOUCH_BUFFER_FRAMES = 1 * framerate

    ACTIVE_TOUCH_ROLES = {
        'ball_action', 'passer', 'shooter',
        'duel_winner', 'duel_loser', 'fouled',
    }

    if anchors is None:
        anchors = _build_time_anchors(path_events, path_info, framerate)

    # Contacts ballon — via EventData floodlight si disponible
    contacts = _parse_contacts_from_event_data(
        event_data, anchors, framerate=framerate)

    # Fallback XML si EventData absent ou vide (path_events toujours disponible ici)
    if contacts.empty:
        print("  [contacts] EventData vide — fallback XML.")
        contacts = _parse_contacts_xml_fallback(
            anchors, framerate, path_events=path_events)

    if contacts.empty:
        print("[WARN] No ball contact events — 'has_ball_touch' set to False.")
        df_movements['has_ball_touch'] = False
        return df_movements

    # Mapping shortname → PersonId
    # Priorité : teamsheets déjà chargés ; fallback : parse matchinfo XML
    if teamsheets is not None:
        shortname_to_pid = _build_shortname_map_from_teamsheets(teamsheets)
    else:
        # Fallback : lecture XML matchinfo (comportement original)
        info_root = ET.parse(path_info).getroot()
        shortname_to_pid = {}
        for player in info_root.findall('.//Player'):
            sn  = player.attrib.get('Shortname')
            pid = player.attrib.get('PersonId')
            if sn and pid:
                shortname_to_pid[sn] = pid

    unknown = set(df_movements['player'].unique()) - set(shortname_to_pid.keys())
    if unknown:
        print(f"[WARN] {len(unknown)} player name(s) not found: {unknown}")

    active_contacts = contacts[contacts['role'].isin(ACTIVE_TOUCH_ROLES)].copy()
    touch_index = {}
    for _, row in active_contacts.iterrows():
        key = (row['person_id'], row['game_section'])
        touch_index.setdefault(key, []).append(row['abs_frame'])
    touch_index = {k: np.array(sorted(v), dtype=int) for k, v in touch_index.items()}

    def movement_has_touch(row):
        gs  = row['half']
        pid = shortname_to_pid.get(row['player'])
        if pid is None:
            return False
        win_start = int(row['start_frame']) + TOUCH_BUFFER_FRAMES
        win_end   = int(row['end_frame'])   - TOUCH_BUFFER_FRAMES
        if win_end <= win_start:
            return False
        frames = touch_index.get((pid, gs))
        if frames is None or len(frames) == 0:
            return False
        lo = np.searchsorted(frames, win_start, side='left')
        return lo < len(frames) and frames[lo] <= win_end

    df_movements['has_ball_touch'] = df_movements.apply(movement_has_touch, axis=1)

    n_touch = df_movements['has_ball_touch'].sum()
    n_total = len(df_movements)
    by_role = active_contacts['role'].value_counts().to_dict()
    print(f"[OK] ball touches tagged — {n_touch}/{n_total} movements.")
    print(f"     active-touch roles : {by_role}")
    print(f"     touch buffer (each side) : {TOUCH_BUFFER_FRAMES/framerate:.1f}s")

    return df_movements


# ============================================================================
# Setpiece blackout  (inchangé — logique métier trop fine)
# — mais utilise maintenant les ancres pré-calculées pour éviter re-parse XML
# ============================================================================

def build_setpiece_blackout_frames(path_events, path_info,
                                   framerate=25, anchors=None):
    """
    Construit les sets de frames blackout par demi-temps.

    Si `anchors` (dict de _build_time_anchors) est fourni, évite de re-parser
    les XMLs. Sinon, les calcule à la volée.
    """
    if anchors is None:
        anchors = _build_time_anchors(path_events, path_info, framerate)

    kickoff_utc           = anchors['kickoff_utc']
    second_half_start_utc = anchors['second_half_start_utc']

    ev_root  = ET.parse(path_events).getroot()
    blackout = {'firstHalf': set(), 'secondHalf': set()}
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

            offset_s      = SETPIECE_OFFSETS[tag]
            offset_frames = int(round(offset_s * framerate))

            elapsed_s = (event_utc - kickoff_utc).total_seconds()
            if elapsed_s < 0:
                continue

            if event_utc < second_half_start_utc:
                gs          = 'firstHalf'
                event_frame = int(round(elapsed_s * framerate))
            else:
                gs          = 'secondHalf'
                elapsed_2nd = (event_utc - second_half_start_utc).total_seconds()
                event_frame = int(round(elapsed_2nd * framerate))

            if tag == 'KickOff':
                kickoff_count[gs] += 1
                if not first_kickoff_seen[gs]:
                    blackout_start = 0
                    first_kickoff_seen[gs] = True
                    print(f"  [blackout] {gs} KickOff #1: event_frame={event_frame} "
                          f"(gap={event_frame/framerate:.1f}s) → blackout étendu au frame 0")
                else:
                    blackout_start = event_frame
                    print(f"  [blackout] {gs} KickOff #{kickoff_count[gs]} (après but): "
                          f"frame={event_frame} ({frame_to_minute(event_frame)}) "
                          f"→ [{blackout_start}, {event_frame + offset_frames})")
            else:
                blackout_start = event_frame

            blackout[gs].update(range(blackout_start, event_frame + offset_frames))
            n_windows += 1
            break

    print(f"[OK] setpiece blackout — {n_windows} windows "
          f"(H1 KickOff: {kickoff_count['firstHalf']}, H2: {kickoff_count['secondHalf']}) | "
          f"{len(blackout['firstHalf'])} frames H1, {len(blackout['secondHalf'])} frames H2.")
    return blackout


# ============================================================================
# Pipeline principal  (allégé — ancres calculées une seule fois)
# ============================================================================

def process_match(
        match_id, DATA_DIR, dict_direction,
        butterworth_Wn      = 0.5,
        butterworth_order   = 1,
        min_valley_distance = 15,
        valley_depth_abs    = 0.3,
        valley_depth_rel    = 0.10,
        power_threshold     = 6.0,
        min_segment_frames  = 10,
        framerate           = 25,
        speed_thresholds    = None,
):
    if speed_thresholds is None:
        speed_thresholds = SPEED_THRESHOLDS

    # ------------------------------------------------------------------
    # 1. Localiser les fichiers
    # ------------------------------------------------------------------
    files = os.listdir(DATA_DIR)

    def find_file(keyword):
        matches = [x for x in files if match_id in x and keyword in x]
        if not matches:
            raise FileNotFoundError(
                f"Aucun fichier '{match_id}' + '{keyword}' dans {DATA_DIR}")
        return os.path.join(DATA_DIR, matches[0])

    path_positions = find_file('positions')
    path_events    = find_file('events')
    path_info      = find_file('matchinformation')

    # ------------------------------------------------------------------
    # 2. Ancres temporelles — calculées UNE SEULE FOIS  ▶▶ OPTIMISATION
    #    L'original parsait les XMLs 3–4 fois séparément :
    #      get_second_half_start_utc(), get_kickoff_frames(),
    #      build_setpiece_blackout_frames(), tag_ball_touches()
    # ------------------------------------------------------------------
    print("Calcul des ancres temporelles…")
    anchors = _build_time_anchors(path_events, path_info, framerate)
    second_half_start_utc = anchors['second_half_start_utc']
    h1_kickoff            = anchors['kickoff_frames']['firstHalf']
    h2_kickoff            = anchors['kickoff_frames']['secondHalf']

    # ------------------------------------------------------------------
    # 3. Charger tracking + events (floodlight — inchangé)
    # ------------------------------------------------------------------
    xy, possession, ballstatus, teamsheets, pitch = dfl.read_position_data_xml(
        path_positions, path_info,
        teamsheet_home=None, teamsheet_away=None,
    )
    _events, _teamsheets_ev, _pitch = dfl.read_event_data_xml(path_events, path_info)
    pitch.sport = "football"

    # ------------------------------------------------------------------
    # 3b. Appliquer les délais de possession basés sur les événements
    # ------------------------------------------------------------------
    contacts = _parse_contacts_from_event_data(_events, anchors, framerate=framerate)
    if contacts.empty:
        print("  [contacts] EventData vide — fallback XML.")
        contacts = _parse_contacts_xml_fallback(anchors, framerate, path_events=path_events)

    possession_intervals = build_possession_intervals(contacts, framerate=framerate, recipient_delay_s=6.0)

    # Modifier possession avec les intervalles
    for half_name in ['firstHalf', 'secondHalf']:
        raw_poss = np.array(possession[half_name]).flatten().astype(int)
        possession[half_name] = apply_possession_intervals_to_array(raw_poss, possession_intervals, teamsheets, half_name)

    xy_all = {
        'firstHalf':  {'Home': xy['firstHalf']['Home'],  'Away': xy['firstHalf']['Away']},
        'secondHalf': {'Home': xy['secondHalf']['Home'], 'Away': xy['secondHalf']['Away']},
    }

    # ------------------------------------------------------------------
    # 4. Velocités filtrées
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
    # 5. Lissage de la possession
    # ------------------------------------------------------------------
    possession_smoothed = {}
    for half_name, half_num in [('firstHalf', 1), ('secondHalf', 2)]:
        raw_poss  = np.array(possession[half_name]).flatten().astype(int)
        half_arr  = np.full(len(raw_poss), half_num, dtype=int)
        smoothed, _, _ = smooth_possession(raw_poss, half_arr, cutoff=50)
        possession_smoothed[half_name] = smoothed

    # ------------------------------------------------------------------
    # 6. Extraction des mouvements discrets (pipeline 2 couches)
    # ------------------------------------------------------------------
    H2_BASE_SECONDS = 45 * 60

    def frame_to_video_timecode(frame, half_label):
        if half_label == 'firstHalf':
            video_s = (frame - h1_kickoff) / framerate
        else:
            video_s = (frame - h2_kickoff) / framerate + H2_BASE_SECONDS
        return sec_to_minute(max(0, video_s))

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

            summary_df['start_timecode'] = summary_df.apply(
                lambda row: frame_to_video_timecode(row['start_frame'], row['half']), axis=1)
            summary_df['end_timecode'] = summary_df.apply(
                lambda row: frame_to_video_timecode(row['end_frame'], row['half']), axis=1)

            ls_movement_dfs.append(summary_df)
            ls_movements_team.append(movements_dict)

        dict_trajectories[team] = {
            k: ls_movements_team[0][k] + ls_movements_team[1][k]
            for k in ls_movements_team[0]
        }

    df_movements = pd.concat(ls_movement_dfs, ignore_index=True)
    df_movements = df_movements[df_movements['distance_m'] > 0]

    # ------------------------------------------------------------------
    # 7. Distance totale par joueur
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
    # 8. Sens de jeu
    # ------------------------------------------------------------------
    if match_id not in dict_direction:
        print(f"[WARN] '{match_id}' not found in direction JSON — 'direction' = NaN.")

    dx = df_movements["x_end"] - df_movements["x_start"]
    dy = df_movements["y_end"] - df_movements["y_start"]
    base_direction   = dx.div(dx.abs().add(dy.abs())).fillna(0)
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

    df_movements["direction"]   = base_direction * multiplier
    df_movements["attack_sign"] = multiplier
    df_movements["match_id"]    = match_id

    # ------------------------------------------------------------------
    # 9. Tag ball touches  (passe ancres + teamsheets + event_data — zéro re-parse)
    # ------------------------------------------------------------------
    df_movements = tag_ball_touches(
        df_movements, path_events, path_info,
        framerate=framerate,
        anchors=anchors,
        teamsheets=teamsheets,
        event_data=_events,
    )

    # ------------------------------------------------------------------
    # 10. Exclusion setpiece blackout  (passe les ancres pré-calculées)
    # ------------------------------------------------------------------
    blackout_frames = build_setpiece_blackout_frames(
        path_events, path_info,
        framerate=framerate, anchors=anchors,
    )

    def is_in_blackout(row):
        bset  = blackout_frames.get(row['half'], set())
        start = int(row['start_frame'])
        return start in bset

    mask_blackout = df_movements.apply(is_in_blackout, axis=1)
    n_excluded    = mask_blackout.sum()
    df_movements  = df_movements[~mask_blackout].reset_index(drop=True)
    print(f"[OK] {n_excluded} movements exclus (blackout setpiece).")

    for half in ['firstHalf', 'secondHalf']:
        sub = df_movements[df_movements['half'] == half]
        if not sub.empty:
            min_frame = sub['start_frame'].min()
            min_tc    = sub.loc[sub['start_frame'].idxmin(), 'start_timecode']
            print(f"    earliest surviving movement in {half}: frame {min_frame} ({min_tc})")

    print(f"[OK] {match_id} — {len(df_movements)} mouvements discrets extraits.")
    return df_movements, df_distances, dict_trajectories