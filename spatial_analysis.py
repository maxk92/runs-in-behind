"""
spatial_analysis.py
===================
Enriches runs-in-behind with Space Control metrics (Floodlight 1.1.0 / DFL).

For each run (CSV row), computes 8 metrics at 3 key instants
(start_frame, mid_frame, end_frame):

  1. sc_runner_area_m2         – area (m²) controlled by the runner
  2. sc_interligne_pct         – % of attacking team control in the zone
                                 between the opponent's defensive and midfield lines
  3. sc_ball_carrier_pct       – % of ball-carrier team control within a radius
                                 around the ball
  4. inter_line_dist           – distance (m) between defensive and midfield lines
  5. defensive_line_width      – lateral width (m) of the defensive block on the Y axis
                                 (leftmost to rightmost outfield defender)
  6. defensive_line_spread     – longitudinal spread (m) of the defensive block on the X axis
                                 (front-to-back compactness, GK excluded)
  7. runner_to_def_dist        – distance (m) from the runner to the nearest opponent defender
  8. ball_carrier_to_def_dist  – distance (m) from the ball to the nearest opponent defender

Output: CSV in OUTPUT_DIR with all original columns + new spatial columns
        (metrics × 3 instants: _start, _mid, _end) + deltas and composite score.

Usage:
    python spatial_analysis.py
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import zscore

import floodlight.io.dfl as dfl
from floodlight.core.xy import XY as FLXY
from floodlight.models.space import DiscreteVoronoiModel


# =============================================================================
# Paths
# =============================================================================

DATA_DIR   = r"C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223"
RUNS_DIR   = r"C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_loop_with_offsets"
OUTPUT_DIR = r"C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_spatial"

FRAMERATE       = 25    # Hz
VORONOI_XPOINTS = 200   # grid resolution (increase up to 200 for higher precision)

# =============================================================================
# Match list
# =============================================================================

LS_MATCH_IDS = [
    'J03WMX', 'J03WN1', 'J03WPY', 'J03WOH','J03WQQ', 'J03WOY', 'J03WR9',
]

# =============================================================================
# DFL position codes → functional category (used for inter-line zone)
# =============================================================================

DEFENDER_POSITIONS   = {'TW', 'IVL', 'IVR', 'LV', 'RV', 'IV'}
MIDFIELDER_POSITIONS = {
    'DML', 'DMR', 'DMZ',
    'ZM', 'ZO', 'ZU',
    'OLM', 'ORM',
    'DM', 'OM',
}
POSITIONS = {
    'IVL', 'IVR', 'LV', 'RV',        # Defence
    'DML', 'DMR', 'DMZ', 'LM', 'RM', # Midfield
    'OLM', 'ORM', 'ZO',              # Attacking midfield
    'STL', 'STR', 'STZ'               # Attack
}

# =============================================================================
# DFL data loading
# =============================================================================

def _build_file_paths(match_id: str) -> dict:
    """
    Resolve paths to the DFL files for a given match.

    Accepted keywords per file type:
      positions : positions_raw, tracking, position
      events    : events_raw, eventdata, event
      info      : matchinformation, matchinfo
    """
    POSITION_KEYWORDS = ('positions_raw', 'tracking', 'position')
    EVENT_KEYWORDS    = ('events_raw', 'eventdata', 'event')
    INFO_KEYWORDS     = ('matchinformation', 'matchinfo')

    path_info      = None
    path_positions = None
    path_events    = None

    for fname in os.listdir(DATA_DIR):
        if match_id not in fname or not fname.endswith('.xml'):
            continue
        flower = fname.lower()
        if path_info is None and any(kw in flower for kw in INFO_KEYWORDS):
            path_info = os.path.join(DATA_DIR, fname)
        elif path_positions is None and any(kw in flower for kw in POSITION_KEYWORDS):
            path_positions = os.path.join(DATA_DIR, fname)
        elif path_events is None and any(kw in flower for kw in EVENT_KEYWORDS):
            path_events = os.path.join(DATA_DIR, fname)

    if path_info is None or path_positions is None or path_events is None:
        all_xml = [f for f in os.listdir(DATA_DIR)
                   if match_id in f and f.endswith('.xml')]
        print(f"  [DEBUG] XML files found for {match_id}: {all_xml}")
        print(f"  [DEBUG] info={path_info}, positions={path_positions}, events={path_events}")

    return {'info': path_info, 'positions': path_positions, 'events': path_events}


def _load_match(match_id: str) -> dict:
    """
    Load DFL tracking data via floodlight.
    Returns a dict with xy, teamsheets and pitch.
    """
    paths = _build_file_paths(match_id)

    if paths['info'] is None:
        raise FileNotFoundError(
            f"[{match_id}] matchinformation not found in {DATA_DIR}"
        )
    if paths['positions'] is None:
        raise FileNotFoundError(
            f"[{match_id}] tracking file not found in {DATA_DIR}"
        )

    print(f"  Loading tracking: {os.path.basename(paths['positions'])}")
    xy, possession, ballstatus, teamsheets, pitch = dfl.read_position_data_xml(
        paths['positions'],
        paths['info'],
        teamsheet_home=None,
        teamsheet_away=None,
    )
    pitch.sport = "football"

    for ts in teamsheets.values():
        ts.add_xIDs()

    return {'xy': xy, 'teamsheets': teamsheets, 'pitch': pitch}


# =============================================================================
# Helpers: single-frame snapshots
# =============================================================================

def _safe_row(xy_data: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
    """Return row frame_idx from an XY numpy array, or None if out of bounds."""
    if frame_idx < 0 or frame_idx >= xy_data.shape[0]:
        return None
    return xy_data[frame_idx]


def _extract_ball_xy(home_row: np.ndarray,
                     xy_half: Optional[dict] = None,
                     frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Look up ball coordinates using the following priority:
      1. 'Ball' key in xy_half (separate XY object, present in some DFL exports)
      2. Last 2 columns of home_row (standard DFL open-data convention)
    Returns np.array([bx, by]) or None if all sources are NaN.
    """
    # 1. Separate Ball key
    if xy_half is not None and frame_idx is not None:
        for key in ('Ball', 'ball', 'BALL'):
            if key in xy_half:
                ball_arr = xy_half[key].xy
                if frame_idx < ball_arr.shape[0]:
                    row = ball_arr[frame_idx]
                    if row.shape[0] >= 2:
                        bx, by = row[0], row[1]
                        if not (np.isnan(bx) or np.isnan(by)):
                            return np.array([bx, by])

    # 2. Last 2 columns of home_row
    if home_row is None or home_row.shape[0] < 2:
        return None
    bx, by = home_row[-2], home_row[-1]
    if np.isnan(bx) or np.isnan(by):
        return None
    return np.array([bx, by])


def _build_flxy_objects(home_row: np.ndarray, away_row: np.ndarray) -> tuple:
    """
    Build two single-frame FLXY objects for Home and Away.
    Strips the last 2 columns from home_row (ball coordinates).
    """
    home_players = home_row[:-2]
    xy1 = FLXY(home_players.reshape(1, -1), framerate=FRAMERATE)
    xy2 = FLXY(away_row.reshape(1, -1),     framerate=FRAMERATE)
    return xy1, xy2


# =============================================================================
# Helpers: opponent defensive and midfield lines
# =============================================================================

def _opponent_line_x(opp_row: np.ndarray, opp_teamsheet) -> tuple:
    """
    Compute the mean X position of opponent defenders and midfielders.
    Returns (x_def_line, x_mid_line) — np.nan if a category is absent.
    If one line is missing, it falls back to the other.
    """
    ts   = opp_teamsheet.teamsheet
    xs   = opp_row[0::2]
    def_x, mid_x = [], []

    for _, row in ts.iterrows():
        xid = row.get('xID')
        pos = str(row.get('position', ''))
        if xid is None:
            continue
        try:
            xid = int(xid)
        except (ValueError, TypeError):
            continue
        if xid >= len(xs):
            continue
        x_val = xs[xid]
        if np.isnan(x_val):
            continue
        if pos in DEFENDER_POSITIONS:
            def_x.append(x_val)
        elif pos in MIDFIELDER_POSITIONS:
            mid_x.append(x_val)

    x_def = float(np.mean(def_x)) if def_x else np.nan
    x_mid = float(np.mean(mid_x)) if mid_x else np.nan

    if np.isnan(x_def) and not np.isnan(x_mid):
        x_def = x_mid
    if np.isnan(x_mid) and not np.isnan(x_def):
        x_mid = x_def

    return x_def, x_mid


# =============================================================================
# Space Control metrics (based on DiscreteVoronoiModel._cell_controls_)
#
# Global xID convention in _cell_controls_ (shape T x ny x nx):
#   xID ∈ [0,       N1-1]   → Home player (xy1), local index = xID
#   xID ∈ [N1, N1+N2-1]     → Away player (xy2), local index = xID - N1
# =============================================================================

def _sc_runner_m2(model: DiscreteVoronoiModel,
                  runner_xid: int, runner_team: str) -> float:
    """Area (m²) controlled by the runner."""
    try:
        global_xid = runner_xid if runner_team == 'Home' else (model._N1_ + runner_xid)
        cell_area  = model._xpolysize_ * model._ypolysize_
        count      = int(np.sum(model._cell_controls_[0] == global_xid))
        return float(count * cell_area)
    except Exception as exc:
        print(f"    [WARN] sc_runner_m2: {exc}")
        return np.nan


def _sc_interligne_pct(model: DiscreteVoronoiModel,
                       att_team: str,
                       x_def_line: float, x_mid_line: float) -> float:
    """
    % of cells controlled by the attacking team in the inter-line zone
    [min(x_def, x_mid), max(x_def, x_mid)].
    Returns np.nan if the zone is narrower than 0.5 m.
    """
    try:
        if np.isnan(x_def_line) or np.isnan(x_mid_line):
            return np.nan

        x_lo, x_hi = min(x_def_line, x_mid_line), max(x_def_line, x_mid_line)
        if abs(x_hi - x_lo) < 0.5:
            return np.nan

        # model._meshx_ shape = (ny, nx): X coordinate of each cell centre
        col_mask   = (model._meshx_[0] >= x_lo) & (model._meshx_[0] <= x_hi)
        if col_mask.sum() == 0:
            return np.nan

        zone_cells = model._cell_controls_[0][:, col_mask]   # (ny, n_cols)
        total_zone = zone_cells.size

        att_ids = (range(model._N1_) if att_team == 'Home'
                   else range(model._N1_, model._N1_ + model._N2_))
        att_count = int(np.sum(np.isin(zone_cells, list(att_ids))))
        return round(100.0 * att_count / total_zone, 2)

    except Exception as exc:
        print(f"    [WARN] sc_interligne_pct: {exc}")
        return np.nan


def _sc_ball_carrier_pct(model: DiscreteVoronoiModel,
                          home_row: np.ndarray, away_row: np.ndarray,
                          ball_xy: Optional[np.ndarray],
                          runner_team: str, runner_xid: int,
                          ball_radius_m: float = 8.0) -> float:
    """
    % of Voronoi cells controlled by the team in possession within
    a radius of `ball_radius_m` around the ball.
    Returns np.nan if ball_xy is None.
    """
    try:
        if ball_xy is None:
            return np.nan

        # Cell distance mask around the ball
        # model._meshx_ and model._meshy_: shape (ny, nx)
        dist_grid = np.hypot(model._meshx_ - ball_xy[0],
                             model._meshy_ - ball_xy[1])
        ball_mask = dist_grid <= ball_radius_m          # (ny, nx) boolean

        zone_cells = model._cell_controls_[0][ball_mask]   # global IDs
        total_zone = zone_cells.size
        if total_zone == 0:
            return np.nan

        # Global IDs belonging to the team in possession
        if runner_team == 'Home':
            att_ids = list(range(model._N1_))
        else:
            att_ids = list(range(model._N1_, model._N1_ + model._N2_))

        att_count = int(np.isin(zone_cells, att_ids).sum())
        return round(100.0 * att_count / total_zone, 2)

    except Exception as exc:
        print(f"    [WARN] sc_ball_carrier_pct: {exc}")
        return np.nan


# =============================================================================
# Structural metrics (position-based, no Voronoi required)
# =============================================================================

def _inter_line_distance(x_def_line: float, x_mid_line: float) -> float:
    """
    Distance (m) between the opponent defensive line and midfield line.
    = |mean_X_defenders − mean_X_midfielders|
    """
    if np.isnan(x_def_line) or np.isnan(x_mid_line):
        return np.nan
    return round(abs(x_def_line - x_mid_line), 3)


def _defensive_line_spread(opp_row: np.ndarray, opp_teamsheet) -> float:
    """
    Longitudinal spread (m) of the defensive line along the X axis.
    = max_X_outfield_defenders − min_X_outfield_defenders  (GK excluded)

    Captures how deep/compact the defensive block is from front to back.
    """
    OUTFIELD_DEFENDERS = DEFENDER_POSITIONS - {'TW'}
    ts = opp_teamsheet.teamsheet
    xs = opp_row[0::2]

    def_x = []
    for _, row in ts.iterrows():
        xid = row.get('xID')
        pos = str(row.get('position', ''))
        if xid is None:
            continue
        try:
            xid = int(xid)
        except (ValueError, TypeError):
            continue
        if xid >= len(xs):
            continue
        x_val = xs[xid]
        if np.isnan(x_val):
            continue
        if pos in OUTFIELD_DEFENDERS:
            def_x.append(x_val)

    if len(def_x) < 2:
        return np.nan
    return round(float(max(def_x) - min(def_x)), 3)


def _defensive_line_width(opp_row: np.ndarray, opp_teamsheet) -> float:
    """
    Lateral width (m) of the defensive line along the Y axis.
    = max_Y_outfield_defenders − min_Y_outfield_defenders  (GK excluded)

    Captures how wide the defensive block stretches across the pitch.
    """
    OUTFIELD_DEFENDERS = DEFENDER_POSITIONS - {'TW'}
    ts = opp_teamsheet.teamsheet
    ys = opp_row[1::2]

    def_y = []
    for _, row in ts.iterrows():
        xid = row.get('xID')
        pos = str(row.get('position', ''))
        if xid is None:
            continue
        try:
            xid = int(xid)
        except (ValueError, TypeError):
            continue
        if xid >= len(ys):
            continue
        y_val = ys[xid]
        if np.isnan(y_val):
            continue
        if pos in OUTFIELD_DEFENDERS:
            def_y.append(y_val)

    if len(def_y) < 2:
        return np.nan
    return round(float(max(def_y) - min(def_y)), 3)


# =============================================================================
# Distance to nearest defender
# =============================================================================

def _dist_to_nearest_defender(
    player_x: float, player_y: float,
    opp_row: np.ndarray,
    opp_teamsheet,
) -> float:
    """
    Distance (m) from a player (player_x, player_y) to the nearest outfield
    opponent defender (GK excluded).
    """
    OUTFIELD_DEFENDERS = POSITIONS - {'TW'}
    ts = opp_teamsheet.teamsheet
    xs = opp_row[0::2]
    ys = opp_row[1::2]

    best_dist = np.inf
    for _, row in ts.iterrows():
        xid = row.get('xID')
        pos = str(row.get('position', ''))
        if xid is None or pos not in OUTFIELD_DEFENDERS:
            continue
        try:
            xid = int(xid)
        except (ValueError, TypeError):
            continue
        if xid >= len(xs):
            continue
        dx, dy = xs[xid], ys[xid]
        if np.isnan(dx) or np.isnan(dy):
            continue
        d = np.hypot(player_x - dx, player_y - dy)
        if d < best_dist:
            best_dist = d

    return round(best_dist, 3) if best_dist < np.inf else np.nan


# =============================================================================
# All 8 metrics at a single frame
# =============================================================================

def _metrics_at_frame(frame_idx: int, half: str,
                       runner_xid: int, runner_team: str,
                       match_data: dict,
                       voronoi_model: DiscreteVoronoiModel) -> dict:
    """
    Compute all 8 metrics for runner_xid at frame_idx.
    Returns a dict (np.nan for each key if computation fails):
        sc_runner, sc_interligne, sc_ball_carrier,
        inter_line_dist, defensive_line_width, defensive_line_spread,
        runner_to_def_dist, ball_carrier_to_def_dist
    """
    xy         = match_data['xy']
    teamsheets = match_data['teamsheets']
    nan_r      = dict(sc_runner=np.nan, sc_interligne=np.nan, sc_ball_carrier=np.nan,
                      inter_line_dist=np.nan, defensive_line_width=np.nan,
                      defensive_line_spread=np.nan,
                      runner_to_def_dist=np.nan, ball_carrier_to_def_dist=np.nan)

    home_row = _safe_row(xy[half]['Home'].xy, frame_idx)
    away_row = _safe_row(xy[half]['Away'].xy, frame_idx)
    if home_row is None or away_row is None:
        return nan_r

    opp_team = 'Away' if runner_team == 'Home' else 'Home'
    opp_row  = away_row if opp_team == 'Away' else home_row
    x_def, x_mid = _opponent_line_x(opp_row, teamsheets[opp_team])

    # Structural metrics (no Voronoi needed)
    inter_line_dist  = _inter_line_distance(x_def, x_mid)
    def_line_width   = _defensive_line_width(opp_row, teamsheets[opp_team])
    def_line_spread  = _defensive_line_spread(opp_row, teamsheets[opp_team])

    # Runner position at this frame
    runner_row = home_row[:-2] if runner_team == 'Home' else away_row
    if runner_xid * 2 + 1 >= len(runner_row):
        # Player slot exists in teamsheet but has no position data at this frame
        # (e.g. substitution window, momentarily missing tracking data)
        return nan_r
    runner_x   = runner_row[runner_xid * 2]
    runner_y   = runner_row[runner_xid * 2 + 1]
    runner_to_def = _dist_to_nearest_defender(
        runner_x, runner_y, opp_row, teamsheets[opp_team])

    # Distance from ball to nearest opponent defender
    ball_xy = _extract_ball_xy(home_row, xy_half=xy[half], frame_idx=frame_idx)
    bc_to_def = np.nan
    if ball_xy is not None:
        bc_to_def = _dist_to_nearest_defender(
            ball_xy[0], ball_xy[1], opp_row, teamsheets[opp_team])

    # Space Control metrics (Voronoi)
    xy1, xy2 = _build_flxy_objects(home_row, away_row)

    try:
        voronoi_model.fit(xy1, xy2)
    except Exception as exc:
        print(f"    [WARN] Voronoi fit failed at frame {frame_idx}: {exc}")
        return dict(sc_runner=np.nan, sc_interligne=np.nan, sc_ball_carrier=np.nan,
                    inter_line_dist=inter_line_dist,
                    defensive_line_width=def_line_width,
                    defensive_line_spread=def_line_spread,
                    runner_to_def_dist=runner_to_def, ball_carrier_to_def_dist=bc_to_def)

    return dict(
        sc_runner               = _sc_runner_m2(voronoi_model, runner_xid, runner_team),
        sc_interligne           = _sc_interligne_pct(voronoi_model, runner_team, x_def, x_mid),
        sc_ball_carrier         = _sc_ball_carrier_pct(
            voronoi_model, home_row, away_row, ball_xy, runner_team, runner_xid),
        inter_line_dist         = inter_line_dist,
        defensive_line_width    = def_line_width,
        defensive_line_spread   = def_line_spread,
        runner_to_def_dist      = runner_to_def,
        ball_carrier_to_def_dist= bc_to_def,
    )


# =============================================================================
# Feature engineering — deltas and composite metrics
# =============================================================================

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived metrics (deltas, composite scores) from the
    _start / _end columns produced by the tracking loop.

    Added columns
    -------------
    delta_sc_runner                  : runner's space control gain (end − start)
    delta_sc_interligne              : opponent inter-line expansion (end − start)
    delta_inter_line_dist            : defensive compression (end − start, Δ<0 = compression)
    delta_sc_ball_carrier            : ball-carrier impact (diversion run?)
    delta_defensive_line_width       : lateral defensive stretch on Y axis (end − start)
    delta_defensive_line_spread      : longitudinal defensive compactness on X axis (end − start)
    delta_dist_runner_defender       : change in runner ↔ nearest defender distance
    delta_ball_carrier_defender      : change in ball ↔ nearest defender distance
    space_per_meter                  : delta_sc_runner per metre covered
    defender_width_stretch_index     : delta_defensive_line_width / delta_sc_runner
    defender_stretch_ratio           : runner_to_def_dist_end / runner_to_def_dist_start
    total_perturbation_score         : sum of z-scores (composite disruption indicator)
                                       includes: delta_sc_runner, delta_sc_interligne,
                                       delta_inter_line_dist, delta_defensive_line_width,
                                       delta_defensive_line_spread, delta_dist_runner_defender,
                                       delta_ball_carrier_defender, space_per_meter,
                                       defender_width_stretch_index, defender_stretch_ratio
    """

    df['delta_sc_runner'] = df['sc_runner_end'] - df['sc_runner_start']
    df['delta_sc_interligne'] = df['sc_interligne_end'] - df['sc_interligne_start']
    df['delta_inter_line_dist'] = df['inter_line_dist_end'] - df['inter_line_dist_start']
    df['delta_sc_ball_carrier'] = df['sc_ball_carrier_end'] - df['sc_ball_carrier_start']

    # Lateral width of the defensive block (Y axis): how wide defenders spread across the pitch
    df['delta_defensive_line_width'] = (
        df['defensive_line_width_end'] - df['defensive_line_width_start']
    )
    # Longitudinal spread of the defensive block (X axis): depth/compactness front-to-back
    df['delta_defensive_line_spread'] = (
        df['defensive_line_spread_end'] - df['defensive_line_spread_start']
    )
    df['delta_dist_runner_defender'] = (
        df['runner_to_def_dist_end'] - df['runner_to_def_dist_start']
    )
    df['delta_ball_carrier_defender'] = (
        df['ball_carrier_to_def_dist_end'] - df['ball_carrier_to_def_dist_start']
    )
    df['space_per_meter'] = (
        (df['delta_sc_runner'] + df['delta_sc_interligne']) / df['distance_m'].replace(0, np.nan)
    )
    df['defender_width_stretch_index'] = (
        df['delta_defensive_line_width'] / df['delta_sc_runner'].replace(0, np.nan)
    )
    df['defender_stretch_ratio'] = (
        df['runner_to_def_dist_end'] / df['runner_to_def_dist_start'].clip(lower=0.01)
    )

    _zs_cols = [
        'delta_sc_runner',
        'delta_sc_interligne',
        'delta_inter_line_dist',
        'delta_defensive_line_width',   # lateral stretch of the defensive block (Y axis)
        'delta_defensive_line_spread',  # longitudinal compactness of the defensive block (X axis)
        'delta_dist_runner_defender',
        'delta_ball_carrier_defender',
        'space_per_meter',
        'defender_width_stretch_index',
        'defender_stretch_ratio',
    ]
    df['total_perturbation_score'] = sum(
        zscore(df[col], nan_policy='omit') for col in _zs_cols
    )

    return df


# =============================================================================
# Per-match processing
# =============================================================================

def process_match(match_id: str) -> pd.DataFrame:
    """Enrich the runs CSV for one match with the 8 spatial metrics."""
    print(f"\n{'='*60}")
    print(f"[{match_id}] Spatial processing")
    print(f"{'='*60}")

    runs_file = os.path.join(RUNS_DIR, f"runs_behind_{match_id}.csv")
    if not os.path.isfile(runs_file):
        print(f"  [SKIP] CSV not found: {runs_file}")
        return pd.DataFrame()

    df_runs = pd.read_csv(runs_file)
    print(f"  {len(df_runs)} runs loaded")

    try:
        match_data = _load_match(match_id)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return pd.DataFrame()

    pitch = match_data['pitch']

    # Instantiate Voronoi model once — the mesh does not change between frames
    voronoi_model = DiscreteVoronoiModel(pitch=pitch, mesh='square',
                                         xpoints=VORONOI_XPOINTS)
    print(f"  Voronoi mesh: {VORONOI_XPOINTS} pts on X  "
          f"(cell ≈ {voronoi_model._xpolysize_:.2f}m × {voronoi_model._ypolysize_:.2f}m)")

    for col in ['sc_runner_start',       'sc_interligne_start',    'sc_ball_carrier_start',
                'sc_runner_mid',         'sc_interligne_mid',      'sc_ball_carrier_mid',
                'sc_runner_end',         'sc_interligne_end',      'sc_ball_carrier_end',
                'inter_line_dist_start',             'inter_line_dist_mid',              'inter_line_dist_end',
                'defensive_line_width_start',        'defensive_line_width_mid',         'defensive_line_width_end',
                'defensive_line_spread_start',       'defensive_line_spread_mid',        'defensive_line_spread_end',
                'runner_to_def_dist_start',          'runner_to_def_dist_mid',           'runner_to_def_dist_end',
                'ball_carrier_to_def_dist_start',    'ball_carrier_to_def_dist_mid',     'ball_carrier_to_def_dist_end']:
        df_runs[col] = np.nan

    total = len(df_runs)
    for idx, run in df_runs.iterrows():
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  run {idx+1}/{total} …")

        half     = run['half']
        xid      = int(run['xID'])
        location = run['location']
        start_f  = int(run['start_frame'])
        end_f    = int(run['end_frame'])
        mid_f    = int((start_f + end_f) / 2)

        for suffix, frame in [('start', start_f), ('mid', mid_f), ('end', end_f)]:
            m = _metrics_at_frame(frame, half, xid, location, match_data, voronoi_model)
            df_runs.at[idx, f'sc_runner_{suffix}']                  = m['sc_runner']
            df_runs.at[idx, f'sc_interligne_{suffix}']              = m['sc_interligne']
            df_runs.at[idx, f'sc_ball_carrier_{suffix}']            = m['sc_ball_carrier']
            df_runs.at[idx, f'inter_line_dist_{suffix}']            = m['inter_line_dist']
            df_runs.at[idx, f'defensive_line_width_{suffix}']       = m['defensive_line_width']
            df_runs.at[idx, f'defensive_line_spread_{suffix}']      = m['defensive_line_spread']
            df_runs.at[idx, f'runner_to_def_dist_{suffix}']         = m['runner_to_def_dist']
            df_runs.at[idx, f'ball_carrier_to_def_dist_{suffix}']   = m['ball_carrier_to_def_dist']

    print(f"  [OK] {match_id} — {total} runs enriched.")

    df_runs = _compute_features(df_runs)
    print(f"  [OK] {match_id} — derived features computed.")

    return df_runs


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: list[pd.DataFrame] = []

    for match_id in LS_MATCH_IDS:
        df_enriched = process_match(match_id)
        if df_enriched.empty:
            continue
        out_file = os.path.join(OUTPUT_DIR, f"runs_spatial_{match_id}.csv")
        df_enriched.to_csv(out_file, index=False)
        print(f"  → {out_file}")
        all_results.append(df_enriched)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        out_all = os.path.join(OUTPUT_DIR, "runs_spatial_ALL.csv")
        df_all.to_csv(out_all, index=False)
        print(f"\n[OK] Aggregated: {out_all}  ({len(df_all)} runs)")
    else:
        print("\n[WARN] No results produced.")


if __name__ == "__main__":
    main()