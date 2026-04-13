"""
spatial_analysis.py
===================
Enrichissement des runs-in-behind avec des métriques de Space Control (Floodlight 1.1.0 / DFL).

Pour chaque run (ligne du CSV), calcule 3 métriques à 3 instants clés
(start_frame, mid_frame, end_frame) :

  1. sc_runner_area_m2  – surface (m²) contrôlée par le joueur qui fait l'appel
  2. sc_interligne_pct  – % de contrôle de l'équipe attaquante dans la zone
                          entre la ligne défensive et la ligne de milieu adverses
  3. sc_ball_carrier_pct – % de contrôle du porteur de balle (joueur le plus
                           proche du ballon à cet instant)

Sortie : CSV dans OUTPUT_DIR avec toutes les colonnes d'origine + 9 nouvelles colonnes
         (3 métriques × 3 instants : _start, _mid, _end).

Usage :
    python spatial_analysis.py
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# ── Floodlight ────────────────────────────────────────────────────────────────
import floodlight.io.dfl as dfl
from floodlight.core.xy import XY as FLXY
from floodlight.models.space import DiscreteVoronoiModel


# =============================================================================
# Chemins
# =============================================================================

DATA_DIR   = r"C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223"
RUNS_DIR   = r"C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_loop_with_offsets"
OUTPUT_DIR = r"C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_spatial"

FRAMERATE       = 25    # Hz
VORONOI_XPOINTS = 100   # résolution de la grille (augmenter jusqu'à 200 pour + de précision)

# =============================================================================
# Liste des matchs
# =============================================================================

LS_MATCH_IDS = [
    'J03WMX', 'J03WN1', 'J03WPY', 'J03WOH',
    'J03WQQ', 'J03WOY', 'J03WR9',
]

# =============================================================================
# Codes de position DFL → catégorie fonctionnelle (pour l'interligne)
# =============================================================================

DEFENDER_POSITIONS   = {'TW', 'IVL', 'IVR', 'LV', 'RV', 'IV'}
MIDFIELDER_POSITIONS = {
    'DML', 'DMR', 'DMZ',
    'ZM', 'ZO', 'ZU',
    'OLM', 'ORM',
    'DM', 'OM',
}
POSITIONS = {
    'IVL', 'IVR', 'LV', 'RV',        # Défense
    'DML', 'DMR', 'DMZ', 'LM', 'RM', # Milieu
    'OLM', 'ORM', 'ZO',              # Milieu offensif
    'STL', 'STR', 'STZ'               # Attaque
}

# =============================================================================
# Chargement des données DFL
# =============================================================================

def _build_file_paths(match_id: str) -> dict:
    """
    Reconstruit les chemins vers les fichiers DFL d'un match.

    Patterns DFL connus :
      positions : DFL_04_03_positions_raw_observed_...  /  DFL_04_04_tracking_...
      events    : DFL_03_02_events_raw_...              /  DFL_02_02_eventdata_...
      info      : DFL_02_01_matchinformation_...
    """
    # Mots-clés acceptés pour chaque type de fichier
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
        print(f"  [DEBUG] Fichiers XML trouvés pour {match_id} : {all_xml}")
        print(f"  [DEBUG] info={path_info}, positions={path_positions}, events={path_events}")

    return {'info': path_info, 'positions': path_positions, 'events': path_events}


def _load_match(match_id: str) -> dict:
    """
    Charge les données de tracking DFL via floodlight.
    Retourne un dict avec xy, teamsheets et pitch.
    """
    paths = _build_file_paths(match_id)

    if paths['info'] is None:
        raise FileNotFoundError(
            f"[{match_id}] matchinformation introuvable dans {DATA_DIR}"
        )
    if paths['positions'] is None:
        raise FileNotFoundError(
            f"[{match_id}] Fichier tracking introuvable dans {DATA_DIR}"
        )

    print(f"  Chargement tracking : {os.path.basename(paths['positions'])}")
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
# Helpers : snapshot à une frame donnée
# =============================================================================

def _safe_row(xy_data: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
    """Retourne la ligne frame_idx d'un tableau XY numpy, ou None si hors bornes."""
    if frame_idx < 0 or frame_idx >= xy_data.shape[0]:
        return None
    return xy_data[frame_idx]


def _extract_ball_xy(home_row: np.ndarray,
                     xy_half: Optional[dict] = None,
                     frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Cherche les coordonnées du ballon dans cet ordre de priorité :
      1. Clé 'Ball' dans xy_half (objet XY séparé, présent dans certains exports DFL)
      2. 2 dernières colonnes de home_row (convention open-data DFL standard)
    Retourne np.array([bx, by]) ou None si toutes les sources sont NaN.
    """
    # 1. Clé 'Ball' séparée
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

    # 2. 2 dernières colonnes de home_row
    if home_row is None or home_row.shape[0] < 2:
        return None
    bx, by = home_row[-2], home_row[-1]
    if np.isnan(bx) or np.isnan(by):
        return None
    return np.array([bx, by])


def _build_flxy_objects(home_row: np.ndarray, away_row: np.ndarray) -> tuple:
    """
    Construit deux objets FLXY (1 frame) pour Home et Away.
    Retire les 2 dernières colonnes de Home (= ballon).
    """
    home_players = home_row[:-2]
    xy1 = FLXY(home_players.reshape(1, -1), framerate=FRAMERATE)
    xy2 = FLXY(away_row.reshape(1, -1),     framerate=FRAMERATE)
    return xy1, xy2


# =============================================================================
# Helpers : lignes défensive et de milieu adverses
# =============================================================================

def _opponent_line_x(opp_row: np.ndarray, opp_teamsheet) -> tuple:
    """
    Calcule la position X moyenne des défenseurs et des milieux adverses.
    Retourne (x_def_line, x_mid_line) — np.nan si catégorie absente.
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
# Métriques Space Control (sur _cell_controls_ du DiscreteVoronoiModel)
#
# Convention xID global dans _cell_controls_ (shape T x ny x nx) :
#   xID ∈ [0,       N1-1]        → joueur Home (xy1), index local = xID
#   xID ∈ [N1, N1+N2-1]          → joueur Away (xy2), index local = xID - N1
# =============================================================================

def _sc_runner_m2(model: DiscreteVoronoiModel,
                  runner_xid: int, runner_team: str) -> float:
    """Surface (m²) contrôlée par le runner."""
    try:
        global_xid = runner_xid if runner_team == 'Home' else (model._N1_ + runner_xid)
        cell_area  = model._xpolysize_ * model._ypolysize_
        count      = int(np.sum(model._cell_controls_[0] == global_xid))
        return float(count * cell_area)
    except Exception as exc:
        print(f"    [WARN] sc_runner_m2 : {exc}")
        return np.nan


def _sc_interligne_pct(model: DiscreteVoronoiModel,
                       att_team: str,
                       x_def_line: float, x_mid_line: float) -> float:
    """
    % de cellules contrôlées par l'équipe attaquante dans la zone
    [min(x_def, x_mid), max(x_def, x_mid)].
    """
    try:
        if np.isnan(x_def_line) or np.isnan(x_mid_line):
            return np.nan

        x_lo, x_hi = min(x_def_line, x_mid_line), max(x_def_line, x_mid_line)
        if abs(x_hi - x_lo) < 0.5:
            return np.nan

        # model._meshx_ shape = (ny, nx) : coordonnées X du centre de chaque cellule
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
        print(f"    [WARN] sc_interligne_pct : {exc}")
        return np.nan


def _sc_ball_carrier_pct(model: DiscreteVoronoiModel,
                          home_row: np.ndarray, away_row: np.ndarray,
                          ball_xy: Optional[np.ndarray],
                          runner_team: str, runner_xid: int) -> float:
    """
    % de surface totale contrôlée par le porteur de balle
    (joueur le plus proche du ballon). Utilise player_controls().
    """
    try:
        carrier_team      = runner_team
        carrier_local_xid = runner_xid

        if ball_xy is not None:
            best_dist = np.inf
            home_players = home_row[:-2]   # sans la colonne ballon

            for xid in range(model._N1_):
                px, py = home_players[xid * 2], home_players[xid * 2 + 1]
                if np.isnan(px) or np.isnan(py):
                    continue
                d = np.hypot(px - ball_xy[0], py - ball_xy[1])
                if d < best_dist:
                    best_dist, carrier_team, carrier_local_xid = d, 'Home', xid

            for xid in range(model._N2_):
                px, py = away_row[xid * 2], away_row[xid * 2 + 1]
                if np.isnan(px) or np.isnan(py):
                    continue
                d = np.hypot(px - ball_xy[0], py - ball_xy[1])
                if d < best_dist:
                    best_dist, carrier_team, carrier_local_xid = d, 'Away', xid

        pc1, pc2 = model.player_controls()   # (T=1, N) en %
        if carrier_team == 'Home':
            return float(pc1.property[0, carrier_local_xid])
        else:
            return float(pc2.property[0, carrier_local_xid])

    except Exception as exc:
        print(f"    [WARN] sc_ball_carrier_pct : {exc}")
        return np.nan


# =============================================================================
# Métriques structurelles (sans Voronoi — calcul direct sur positions)
# =============================================================================

def _inter_line_distance(x_def_line: float, x_mid_line: float) -> float:
    """
    Distance (m) entre la ligne défensive adverse et la ligne de milieu adverse.
    = |moy_X_défenseurs  −  moy_X_milieux|
    """
    if np.isnan(x_def_line) or np.isnan(x_mid_line):
        return np.nan
    return round(abs(x_def_line - x_mid_line), 3)


def _team_length(opp_row: np.ndarray, opp_teamsheet) -> float:
    """
    Longueur (m) de l'équipe défensive le long de l'axe X.
    = max_X_défenseurs_outfield  −  min_X_défenseurs_outfield  (GK exclu)
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


# =============================================================================
# Métriques de distance au défenseur le plus proche
# =============================================================================

def _dist_to_nearest_defender(
    player_x: float, player_y: float,
    opp_row: np.ndarray,
    opp_teamsheet,
) -> float:
    """
    Distance (m) entre un joueur (player_x, player_y) et le défenseur adverse
    outfield le plus proche (GK exclu).
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
# Calcul des 7 métriques à une frame donnée
# =============================================================================

def _metrics_at_frame(frame_idx: int, half: str,
                       runner_xid: int, runner_team: str,
                       match_data: dict,
                       voronoi_model: DiscreteVoronoiModel) -> dict:
    """
    Calcule les 7 métriques pour runner_xid à frame_idx.
    Retourne un dict (np.nan pour chaque clé si calcul impossible) :
        sc_runner, sc_interligne, sc_ball_carrier,
        inter_line_dist, team_length,
        runner_to_def_dist, ball_carrier_to_def_dist
    """
    xy         = match_data['xy']
    teamsheets = match_data['teamsheets']
    nan_r      = dict(sc_runner=np.nan, sc_interligne=np.nan, sc_ball_carrier=np.nan,
                      inter_line_dist=np.nan, team_length=np.nan,
                      runner_to_def_dist=np.nan, ball_carrier_to_def_dist=np.nan)

    home_row = _safe_row(xy[half]['Home'].xy, frame_idx)
    away_row = _safe_row(xy[half]['Away'].xy, frame_idx)
    if home_row is None or away_row is None:
        return nan_r

    opp_team = 'Away' if runner_team == 'Home' else 'Home'
    opp_row  = away_row if opp_team == 'Away' else home_row
    x_def, x_mid = _opponent_line_x(opp_row, teamsheets[opp_team])

    # Métriques structurelles (pas besoin de Voronoi)
    inter_line_dist = _inter_line_distance(x_def, x_mid)
    team_len        = _team_length(opp_row, teamsheets[opp_team])

    # Position du runner à cette frame
    runner_row = home_row[:-2] if runner_team == 'Home' else away_row
    if runner_xid * 2 + 1 >= len(runner_row):
        # Player slot exists in teamsheet but has no position data at this frame
        # (e.g. substitution window, momentarily missing tracking data)
        return nan_r
    runner_x   = runner_row[runner_xid * 2]
    runner_y   = runner_row[runner_xid * 2 + 1]
    runner_to_def = _dist_to_nearest_defender(
        runner_x, runner_y, opp_row, teamsheets[opp_team])

    # Distance entre la balle et le défenseur adverse le plus proche
    ball_xy = _extract_ball_xy(home_row, xy_half=xy[half], frame_idx=frame_idx)
    bc_to_def = np.nan
    if ball_xy is not None:
        bc_to_def = _dist_to_nearest_defender(
            ball_xy[0], ball_xy[1], opp_row, teamsheets[opp_team])

    # Métriques Space Control (Voronoi)
    xy1, xy2 = _build_flxy_objects(home_row, away_row)

    try:
        voronoi_model.fit(xy1, xy2)
    except Exception as exc:
        print(f"    [WARN] Voronoi fit échoué frame {frame_idx} : {exc}")
        return dict(sc_runner=np.nan, sc_interligne=np.nan, sc_ball_carrier=np.nan,
                    inter_line_dist=inter_line_dist, team_length=team_len,
                    runner_to_def_dist=runner_to_def, ball_carrier_to_def_dist=bc_to_def)

    return dict(
        sc_runner               = _sc_runner_m2(voronoi_model, runner_xid, runner_team),
        sc_interligne           = _sc_interligne_pct(voronoi_model, runner_team, x_def, x_mid),
        sc_ball_carrier         = _sc_ball_carrier_pct(
            voronoi_model, home_row, away_row, ball_xy, runner_team, runner_xid),
        inter_line_dist         = inter_line_dist,
        team_length             = team_len,
        runner_to_def_dist      = runner_to_def,
        ball_carrier_to_def_dist= bc_to_def,
    )


# =============================================================================
# Traitement d'un match
# =============================================================================

def process_match(match_id: str) -> pd.DataFrame:
    """Enrichit le CSV des runs d'un match avec les 9 métriques spatiales."""
    print(f"\n{'='*60}")
    print(f"[{match_id}] Traitement spatial")
    print(f"{'='*60}")

    runs_file = os.path.join(RUNS_DIR, f"runs_behind_{match_id}.csv")
    if not os.path.isfile(runs_file):
        print(f"  [SKIP] CSV introuvable : {runs_file}")
        return pd.DataFrame()

    df_runs = pd.read_csv(runs_file)
    print(f"  {len(df_runs)} runs chargés")

    try:
        match_data = _load_match(match_id)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return pd.DataFrame()

    pitch = match_data['pitch']

    # Instancier le modèle Voronoi UNE FOIS : le mesh ne change pas entre frames
    voronoi_model = DiscreteVoronoiModel(pitch=pitch, mesh='square',
                                         xpoints=VORONOI_XPOINTS)
    print(f"  Voronoi mesh : {VORONOI_XPOINTS} pts en X  "
          f"(cellule ≈ {voronoi_model._xpolysize_:.2f}m × {voronoi_model._ypolysize_:.2f}m)")

    for col in ['sc_runner_start',       'sc_interligne_start',    'sc_ball_carrier_start',
                'sc_runner_mid',         'sc_interligne_mid',      'sc_ball_carrier_mid',
                'sc_runner_end',         'sc_interligne_end',      'sc_ball_carrier_end',
                'inter_line_dist_start',        'inter_line_dist_mid',         'inter_line_dist_end',
                'team_length_start',           'team_length_mid',             'team_length_end',
                'runner_to_def_dist_start',    'runner_to_def_dist_mid',      'runner_to_def_dist_end',
                'ball_carrier_to_def_dist_start', 'ball_carrier_to_def_dist_mid', 'ball_carrier_to_def_dist_end']:
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
            df_runs.at[idx, f'sc_runner_{suffix}']         = m['sc_runner']
            df_runs.at[idx, f'sc_interligne_{suffix}']     = m['sc_interligne']
            df_runs.at[idx, f'sc_ball_carrier_{suffix}']   = m['sc_ball_carrier']
            df_runs.at[idx, f'inter_line_dist_{suffix}']          = m['inter_line_dist']
            df_runs.at[idx, f'team_length_{suffix}']               = m['team_length']
            df_runs.at[idx, f'runner_to_def_dist_{suffix}']        = m['runner_to_def_dist']
            df_runs.at[idx, f'ball_carrier_to_def_dist_{suffix}']  = m['ball_carrier_to_def_dist']

    print(f"  [OK] {match_id} — {total} runs enrichis.")
    return df_runs


# =============================================================================
# Point d'entrée
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
        print(f"\n[OK] Agrégé : {out_all}  ({len(df_all)} runs)")
    else:
        print("\n[WARN] Aucun résultat produit.")


if __name__ == "__main__":
    main()