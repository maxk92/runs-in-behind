"""
enrich_annotations.py
=====================
Production script that:
  1. Reads manual annotation files (one or several, raw or validated ground-truth).
  2. For each annotated run, finds the best-matching segment produced by the
     automated discretisation pipeline.
  3. Extracts quantitative indicators from the raw position data.
  4. Merges designated playing position from the DFL matchsheet.
  5. Computes a "current scoreline" from goal events in the DFL event data.
  6. Optionally: flags runs followed by a shot within K seconds (configurable).
  7. Exports one CSV per annotation file (and optionally a combined CSV).

The row "auto_segment_excel_row" contains the index of the corresponding automated segment in the Excel file. Because exel files are 1-indexed, this is the row number in the Excel file (not the DataFrame index). If no match is found, this field is NaN.
The row "auto_segment_id"  can be used with panda to analyse the matched automated segments. It is the index of the corresponding automated segment in the DataFrame produced by the loop script (and saved as runs_behind_{match_id}.csv). If no match is found, this field is NaN.

Dependencies
------------
    pip install pandas numpy scipy lxml floodlight-science
    (floodlight must be importable, same as Discretisation_optimised2_1.py)
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

import floodlight.io.dfl as dfl
from floodlight.models.kinematics import VelocityModel
from floodlight.transforms.filter import butterworth_lowpass

from common import config as _config
from common.iou import temporal_iou as _shared_temporal_iou


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit this block
# ─────────────────────────────────────────────────────────────────────────────

# Root folder that contains all DFL open-data XML files
DATA_DIR = _config.DATA_DIR

# Folder that contains the annotation CSV(s)  (or a single file path)
# Canonical folder -- also used by Extraction_Evaluation.py and
# Stats_Manual_Annotation_App.py (previously a separate, differently-shaped
# "annotation_app_output" folder).
ANNOTATION_DIR = _config.ANNOTATION_DIR

# Glob pattern inside ANNOTATION_DIR – adjust to pick specific files
ANNOTATION_GLOB = "*.csv"          # e.g. "*_validated.csv"  or  "*_david.csv"

# Folder where the loop script saved its per-match automated outputs
# (the files named  runs_behind_{match_id}.csv)
AUTO_OUTPUT_DIR = _config.AUTO_OUTPUT_DIR

# Output folder for the enriched CSVs produced by THIS script
OUTPUT_DIR = _config.ENRICHED_OUTPUT_DIR

# How many seconds after a run must a shot happen to flag it?
SHOT_WINDOW_S = 10          # set to None to skip shot-flag computation

# How many seconds BEFORE the run start to look back when computing the
# "pre-run" velocity indicators (mean velocity and peak velocity during the
# k seconds preceding the run's first frame). Set to None to skip this
# computation entirely.
PRE_RUN_WINDOW_S = 2.0      # k, in seconds

# Minimum IoU required to accept a match between an annotation and an
# automated segment.  Candidates below this threshold are left as NaN.
MIN_IOU = 0.01               # 0 = accept any overlap; 1 = perfect match only

# ── Reuse metrics already computed by the automated pipeline ───────────────
# The runs_behind_{match_id}.csv files (AUTO_OUTPUT_DIR) are the movements
# already extracted AND whose kinematic indicators (distance, speed, zones,
# pre-run window...) have ALREADY been computed by
# Discretisation_optimised2_1.py. When an annotation is matched to one of
# these automated movements (via find_best_segment / IoU >= MIN_IOU), there
# is no need to reload the raw XML positions and recompute everything: the
# values already present in runs_behind_*.csv are reused directly (with a
# km/h → m/s unit conversion, see _auto_row_to_indicators()).
# Only annotations WITHOUT an automated match (or with a match whose
# indicators are missing) still trigger extract_run_indicators().
REUSE_AUTO_METRICS = True

# Butterworth filter parameters 
BUTTERWORTH_WN    = 0.5
BUTTERWORTH_ORDER = 1
FRAMERATE         = 25      # frames per second

# Path to the JSON describing each team's playing direction per half.
# Structure: { match_id: { "Home": { "firstHalf": "...", "secondHalf": "..." } } }
# Only "Home" is stored — "Away" is always the opposite direction in the same half.
DIRECTION_JSON_PATH = _config.DIRECTION_JSON

# Canonical direction used ONLY for zone assignment (D/M/A × 1-5).
# Every run is mirrored on x so that, for zoning purposes, everyone "plays"
# in this direction — raw x_start/x_mid/x_end/y_* stay untouched.
ZONE_TARGET_DIRECTION = "right_to_left"

# ─────────────────────────────────────────────────────────────────────────────
# PITCH ZONE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Standard DFL coordinate system:
#   x ∈ [-52.5, 52.5]  (negative = left goal, positive = right goal)
#   y ∈ [-34,   34  ]  (negative = bottom touchline)
#
# Vertical bands (5 channels) based on horizontal cut-points requested:
#   corner flag → penalty-box line → near post → far post → penalty-box line → corner flag
#   Approx. y boundaries: -34, -20.16, -3.66, 3.66, 20.16, 34
#   (penalty area width = 40.32 m wide → ±20.16; goal posts ±3.66)
#
# Horizontal thirds (3 zones):
#   Defensive third   x < -17.5
#   Middle third     -17.5 ≤ x ≤ 17.5
#   Attacking third   x >  17.5
#
# Combined: 3 thirds × 5 channels = 15 zones, labelled as e.g. "D1", "M3", "A5"
#   Third prefix: D = defensive, M = middle, A = attacking
#   Channel suffix R, HR, C, HL,L (Right, Half Right, Center, Half Left, Left) from bottom (negative y) to top (positive y)

THIRD_BOUNDARIES_X = [-17.5, 17.5]                # two cut-points → three thirds
CHANNEL_BOUNDARIES_Y = [-20.16, -3.66, 3.66, 20.16]  # four cut-points → five channels


def load_play_directions(path: str = DIRECTION_JSON_PATH) -> dict:
    """
    Load the playing-direction JSON.

    Returns {} (with a warning) if the file is missing/unreadable, so the
    rest of the pipeline still runs (zones just won't be direction-normalised).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not load direction JSON at '{path}': {exc}. "
              "Zones will NOT be normalised for playing direction.")
        return {}


# Loaded once at import time.
PLAY_DIRECTIONS = load_play_directions()

_OPPOSITE_DIRECTION = {
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
}


def get_play_direction(match_id: str, team: str, half: str,
                       directions: dict = PLAY_DIRECTIONS) -> str | None:
    """
    Return 'left_to_right' or 'right_to_left' — the direction `team`
    ('Home' or 'Away') is attacking during `half`.

    Only "Home" is stored in the JSON; "Away" is derived as the opposite
    direction in the same half. Returns None if unknown (missing match_id,
    missing half, or unrecognised team), in which case callers should skip
    normalisation rather than guess.
    """
    home_dir = directions.get(match_id, {}).get("Home", {}).get(half)
    if home_dir is None:
        return None
    if team == "Home":
        return home_dir
    if team == "Away":
        return _OPPOSITE_DIRECTION.get(home_dir)
    return None


def assign_zone(x: float, y: float, direction: str | None = None,
                target: str = ZONE_TARGET_DIRECTION) -> tuple[str, str]:
    """
    Return a zone label like 'D1', 'M3', 'A5' for a given (x, y) position.

    If `direction` is given and differs from `target`, x and y are mirrored first
    so that every run is zoned as if it were played in `target` direction
    (e.g. everyone "plays right to left"). This only affects which zone
    label is returned — it does not mutate the caller's raw coordinates.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: DFL file discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_dfl_file(data_dir: str, match_id: str, keyword: str) -> str:
    """Return the first file in data_dir whose name contains match_id AND keyword."""
    files = os.listdir(data_dir)
    hits = [f for f in files if match_id in f and keyword in f]
    if not hits:
        raise FileNotFoundError(
            f"No file for match '{match_id}' with keyword '{keyword}' in {data_dir}"
        )
    return os.path.join(data_dir, hits[0])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Position data → per-player velocity arrays
# ─────────────────────────────────────────────────────────────────────────────

def load_position_data(match_id: str, data_dir: str):
    """
    Load raw position data using floodlight.
    Returns (xy_dict, teamsheets, pitch)
      xy_dict = {half: {team: XY object}}
    """
    path_pos  = find_dfl_file(data_dir, match_id, "positions")
    path_info = find_dfl_file(data_dir, match_id, "matchinformation")
    xy, _possession, _ballstatus, teamsheets, pitch = dfl.read_position_data_xml(
        path_pos, path_info,
        teamsheet_home=None, teamsheet_away=None,
    )
    return xy, teamsheets, pitch


def compute_velocities(xy_half_team, Wn=BUTTERWORTH_WN, order=BUTTERWORTH_ORDER):
    """Return velocity array (frames × players) in m/s after Butterworth filtering."""
    xy_f = butterworth_lowpass(xy_half_team, remove_short_seqs=True,
                               Wn=Wn, order=order)
    vm = VelocityModel()
    vm.fit(xy_f)
    return vm.velocity().property, xy_f.xy   # (vel_array, filtered_xy_array)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Matchsheet — parse matchinfo XML for the authoritative player map
# ─────────────────────────────────────────────────────────────────────────────

def parse_matchinfo_players(path_info: str) -> pd.DataFrame:
    """
    Parse the DFL matchinformation XML directly and return a DataFrame with one
    row per player:

        person_id        – DFL-OBJ-XXXXXX  (primary key used in tracking data)
        jID              – shirt / jersey number (int, join key in annotation files)
        shortname        – e.g. "Kingsley Coman"
        first_name
        last_name
        playing_position – e.g. "OLM", "DMR", "TW" (empty string if not provided)
        starting         – bool
        team_id          – DFL-CLU-XXXXXX
        team_name        – e.g. "FC Bayern München"
        team_role        – "home" or "guest"

    Both teams are included. Non-playing staff are excluded.
    """
    root = ET.parse(path_info).getroot()
    rows = []
    for team_el in root.findall(".//Team"):
        team_id   = team_el.get("TeamId", "")
        team_name = team_el.get("TeamName", "")
        team_role = team_el.get("Role", "")          # "home" or "guest"
        # Map DFL role to the floodlight convention used elsewhere in this script
        team_loc  = "Home" if team_role == "home" else "Away"

        for player_el in team_el.findall("Players/Player"):
            shirt_raw = player_el.get("ShirtNumber", "")
            try:
                jid = int(shirt_raw)
            except ValueError:
                jid = None

            rows.append({
                "person_id":        player_el.get("PersonId", ""),
                "jID":              jid,
                "shortname":        player_el.get("Shortname", ""),
                "first_name":       player_el.get("FirstName", ""),
                "last_name":        player_el.get("LastName", ""),
                "playing_position": player_el.get("PlayingPosition", ""),
                "starting":         player_el.get("Starting", "false").lower() == "true",
                "team_id":          team_id,
                "team_name":        team_name,
                "team_role":        team_role,
                "team_loc":         team_loc,   # "Home" / "Away"
            })

    return pd.DataFrame(rows)


def build_person_id_to_xid(teamsheets: dict) -> dict:
    """
    Build a {person_id: (team_loc, xID)} lookup from floodlight teamsheets.

    Floodlight stores the DFL PersonId in a column typically named 'pID' or
    'player_id', and the column index into the XY array as 'xID'.
    We search for both robustly.
    """
    mapping = {}   # person_id → (team_loc, xID)
    for team_loc, ts_obj in teamsheets.items():
        ts = ts_obj.teamsheet
        pid_col = next((c for c in ts.columns
                        if c.lower() in ("pid", "player_id", "personid", "person_id")),
                       None)
        xid_col = next((c for c in ts.columns if c.lower() == "xid"), None)
        if pid_col is None or xid_col is None:
            continue
        for _, row in ts.iterrows():
            pid = str(row[pid_col]).strip()
            xid = row[xid_col]
            if pid and not pd.isna(xid):
                mapping[pid] = (team_loc, int(xid))
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Event data → goals and shots
# ─────────────────────────────────────────────────────────────────────────────

def _utc_to_elapsed(event_utc: pd.Timestamp,
                    kickoff_utc: pd.Timestamp,
                    second_half_utc: pd.Timestamp | None) -> tuple[float, str]:
    """
    Convert an event UTC timestamp to (elapsed_s_in_half, half_label).
    elapsed_s is measured from the respective half kickoff.
    """
    if second_half_utc is None or event_utc < second_half_utc:
        return (event_utc - kickoff_utc).total_seconds(), "firstHalf"
    else:
        return (event_utc - second_half_utc).total_seconds(), "secondHalf"


def parse_events_xml(match_id: str, data_dir: str) -> dict:
    path_events = find_dfl_file(data_dir, match_id, "events")
    path_info   = find_dfl_file(data_dir, match_id, "matchinformation")

    # ── Kickoff UTC from matchinfo ───────────────────────────────────────
    info_root   = ET.parse(path_info).getroot()
    kickoff_str = info_root.find(".//KickoffTime")
    kickoff_utc = (pd.Timestamp(kickoff_str.text).tz_convert("UTC")
                   if kickoff_str is not None and kickoff_str.text else None)

    # ── Team IDs home/away ────────────────────────────────────────────────
    home_team_id = away_team_id = None
    for el in info_root.iter():
        role = el.get("Role", "")
        tid  = el.get("TeamId") or el.get("TeamID", "")
        if not tid:
            continue
        if role == "home":
            home_team_id = tid
        elif role in ("guest", "away"):
            away_team_id = tid

    # ── Parse events XML ──────────────────────────────────────────────────
    ev_root = ET.parse(path_events).getroot()

    # First pass: find kickoffs via GameSection (reliable)
    # The XML may contain several KickOff events (restart after a goal, etc.)
    # → we rely on GameSection="firstHalf" / "secondHalf"
    kickoff_utc_from_xml = None
    second_half_utc      = None
    all_kickoff_times    = []

    for event in ev_root.findall("Event"):
        t_str = event.attrib.get("EventTime")
        if not t_str:
            continue
        for child in event:
            if child.tag != "KickOff":
                continue
            ts      = pd.Timestamp(t_str).tz_convert("UTC")
            section = child.get("GameSection", "")
            all_kickoff_times.append(ts)
            if section == "firstHalf" and kickoff_utc_from_xml is None:
                kickoff_utc_from_xml = ts
            elif section == "secondHalf" and second_half_utc is None:
                second_half_utc = ts

    # If GameSection is absent: fallback — first kickoff = 1st half,
    # then look for a gap > 10 min to find the start of the 2nd half
    all_kickoff_times.sort()
    if not all_kickoff_times:
        raise ValueError(f"No KickOff events found for match {match_id}")

    if kickoff_utc_from_xml is None:
        kickoff_utc_from_xml = all_kickoff_times[0]

    if second_half_utc is None:
        for i in range(1, len(all_kickoff_times)):
            gap = (all_kickoff_times[i] - all_kickoff_times[i - 1]).total_seconds()
            if gap > 600:   # pause > 10 min → start of the 2nd half
                second_half_utc = all_kickoff_times[i]
                break

    # Prefer the kickoff found in matchinfo, fall back to the one from the XML
    if kickoff_utc is None:
        kickoff_utc = kickoff_utc_from_xml

    print(f"  Kickoff 1st half : {kickoff_utc}")
    print(f"  Kickoff 2nd half : {second_half_utc}")

    # Tags that indicate a shot (regardless of nesting depth)
    SHOT_TAGS = {"ShotAtGoal", "ShotOnTarget", "ShotOffTarget",
                 "SavedShot", "ShotWide", "ShotHigh", "BlockedShot"}

    goal_rows  = []
    shot_rows  = []
    home_score = 0
    away_score = 0

    for event in ev_root.findall("Event"):
        t_str = event.attrib.get("EventTime")
        if not t_str:
            continue
        event_utc = pd.Timestamp(t_str).tz_convert("UTC")
        elapsed_s, half = _utc_to_elapsed(event_utc, kickoff_utc, second_half_utc)

        # Walk all descendants to find ShotAtGoal
        # (may be a direct child OR wrapped in <Penalty>, <FreeKick>, etc.)
        for shot_el in event.iter():
            if shot_el.tag not in SHOT_TAGS:
                continue

            # Team is an attribute of the shot tag itself
            team_id   = shot_el.get("Team", "")
            shot_team = "Home" if team_id == home_team_id else "Away"

            shot_rows.append({
                "elapsed_s": elapsed_s,
                "half":      half,
                "team":      shot_team,
                "shot_type": shot_el.tag,
            })

            # Goal = SuccessfulShot direct child of the shot tag
            goal_el = shot_el.find("SuccessfulShot")
            if goal_el is not None:
                if shot_team == "Home":
                    home_score += 1
                else:
                    away_score += 1
                goal_rows.append({
                    "elapsed_s":    elapsed_s,
                    "half":         half,
                    "scoring_team": shot_team,
                    "home_score":   home_score,
                    "away_score":   away_score,
                })

    df_goals = (pd.DataFrame(goal_rows) if goal_rows
                else pd.DataFrame(columns=["elapsed_s", "half", "scoring_team",
                                           "home_score", "away_score"]))
    df_shots = (pd.DataFrame(shot_rows) if shot_rows
                else pd.DataFrame(columns=["elapsed_s", "half", "team", "shot_type"]))

    print(f"  Events parsed: {len(df_goals)} goals, {len(df_shots)} shots")
    if not df_goals.empty:
        print(df_goals[["half", "elapsed_s", "scoring_team",
                         "home_score", "away_score"]].to_string(index=False))

    return {
        "goals":           df_goals,
        "shots":           df_shots,
        "kickoff_utc":     kickoff_utc,
        "second_half_utc": second_half_utc,
    }

def get_scoreline_at(elapsed_s: float, half: str,
                     df_goals: pd.DataFrame) -> str:
    """
    Return the scoreline string 'H–A' at the moment (half, elapsed_s).
    elapsed_s is seconds since the start of the respective half.
    """
    if df_goals.empty:
        return "0-0"

    # Convert everything to a single "total_elapsed" for comparison
    # firstHalf: 0–45*60;  secondHalf: 45*60 + elapsed
    def to_total(row):
        return row["elapsed_s"] if row["half"] == "firstHalf" else 45 * 60 + row["elapsed_s"]

    run_total = elapsed_s if half == "firstHalf" else 45 * 60 + elapsed_s
    df_goals = df_goals.copy()
    df_goals["total_elapsed"] = df_goals.apply(to_total, axis=1)

    prior = df_goals[df_goals["total_elapsed"] <= run_total]
    if prior.empty:
        return "0-0"
    last = prior.iloc[-1]
    return f"{int(last['home_score'])}-{int(last['away_score'])}"


def get_goal_indicator(scoreline: str, team: str) -> int:
    """
    Signed "goal-deficit" indicator for the run, from the runner's team
    perspective, based on the scoreline 'H-A' at the moment of the run.

    diff = away_score - home_score  (i.e. how many goals Away leads by;
           negative means Home leads)

    If the runner is on the Away team  -> indicator =  diff
    If the runner is on the Home team  -> indicator = -diff

    Examples (Home-Away):
      0-1 (Away +1)  -> Away run = +1, Home run = -1
      1-2 (Away +1)  -> Away run = +1, Home run = -1
      0-2 (Away +2)  -> Away run = +2, Home run = -2
      2-0 (Home +2)  -> Away run = -2, Home run = +2
      1-1 (level)    -> 0 for either team
    """
    try:
        home_s_str, away_s_str = scoreline.split("-")
        home_s, away_s = int(home_s_str), int(away_s_str)
    except (AttributeError, ValueError):
        return np.nan

    diff = away_s - home_s
    return diff if team == "Away" else -diff


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Extract kinematic indicators for a single run
# ─────────────────────────────────────────────────────────────────────────────

def extract_run_indicators(start_frame: int, end_frame: int,
                           half: str, person_id: str,
                           xy_dict: dict,
                           pid_to_xid: dict,
                           match_id: str | None = None,
                           team: str | None = None,
                           pre_window_s: float | None = PRE_RUN_WINDOW_S) -> dict | None:
    """
    Compute kinematic indicators for a single run from raw position data.

    Parameters
    ----------
    start_frame, end_frame : absolute frame numbers within the half
    half                   : "firstHalf" or "secondHalf"
    person_id              : DFL PersonId string, e.g. "DFL-OBJ-0026PM"
    xy_dict                : {half: {team: XY}} from load_position_data()
    pid_to_xid             : {person_id: (team_loc, xID)} from build_person_id_to_xid()
    match_id, team         : used to look up playing direction (via
                              get_play_direction) so that zone_start/zone_end
                              are normalised for playing direction. If either
                              is None, or the direction is unknown, zones are
                              computed WITHOUT normalisation (raw x/y).
    pre_window_s           : length (in seconds) of the "pre-run" window used
                              to compute pre_run_mean_vel_ms / pre_run_peak_vel_ms
                              (the k seconds immediately BEFORE start_frame).
                              Set to None (or <= 0) to skip this computation.

    Returns a dict of indicators, or None if the player/frames are not found.
    """
    lookup = pid_to_xid.get(person_id)
    if lookup is None:
        return None
    player_team, player_xid = lookup

    # Get the xy array for the correct half and team
    xy_half_team = xy_dict[half][player_team]
    vel_arr, xy_arr = compute_velocities(xy_half_team)

    # player_xid is the column index (0-based) in the floodlight XY object
    n_frames_half = xy_arr.shape[0]
    sf = max(0, min(start_frame, n_frames_half - 1))
    ef = max(0, min(end_frame,   n_frames_half - 1))

    if sf >= ef:
        return None

    # ── Pre-run window: k seconds BEFORE the run's first frame ────────────
    pre_run_mean_vel = np.nan
    pre_run_peak_vel = np.nan
    pre_run_window_used_s = np.nan
    if pre_window_s is not None and pre_window_s > 0:
        pre_frames = int(round(pre_window_s * FRAMERATE))
        pre_sf = max(0, sf - pre_frames)   # clipped to the start of the half
        pre_ef = sf                        # window ends right where the run starts (exclusive)
        if pre_ef > pre_sf:
            v_pre_seg = vel_arr[pre_sf:pre_ef, player_xid]
            v_pre_valid = v_pre_seg[~np.isnan(v_pre_seg)]
            if v_pre_valid.size > 0:
                pre_run_mean_vel = float(np.mean(v_pre_valid))
                pre_run_peak_vel = float(np.max(v_pre_valid))
            pre_run_window_used_s = round((pre_ef - pre_sf) / FRAMERATE, 2)

    x_col = player_xid * 2
    y_col = player_xid * 2 + 1

    x_seg = xy_arr[sf:ef + 1, x_col]
    y_seg = xy_arr[sf:ef + 1, y_col]
    v_seg = vel_arr[sf:ef + 1, player_xid]   # m/s

    # Remove NaN-only segments
    valid = ~(np.isnan(x_seg) | np.isnan(y_seg))
    if valid.sum() < 2:
        return None

    x_valid = x_seg[valid]
    y_valid = y_seg[valid]
    v_valid = v_seg[~np.isnan(v_seg)] if not np.all(np.isnan(v_seg)) else np.array([0.0])

    dx = np.diff(x_valid)
    dy = np.diff(y_valid)
    length_m  = float(np.sqrt(dx**2 + dy**2).sum())
    duration_s = (ef - sf) / FRAMERATE
    mean_vel   = float(np.nanmean(v_seg))          # m/s
    peak_vel   = float(np.nanmax(v_valid)) if v_valid.size > 0 else 0.0

    x_start = float(x_valid[0])
    y_start = float(y_valid[0])
    x_end   = float(x_valid[-1])
    y_end   = float(y_valid[-1])
    x_mid   = float(np.nanmedian(x_valid))
    y_mid   = float(np.nanmedian(y_valid))

    direction = None
    if match_id is not None and team is not None:
        direction = get_play_direction(match_id, team, half)
        if direction is None:
            print(f"    [WARN] Unknown playing direction for match={match_id} "
                  f"team={team} half={half} — zone NOT normalised for this run.")

    zone_x_start, zone_y_start = assign_zone(x_start, y_start, direction=direction)
    zone_x_mid, zone_y_mid     = assign_zone(x_mid, y_mid, direction=direction)
    zone_x_end, zone_y_end     = assign_zone(x_end, y_end, direction=direction)
    zone_start = f"{zone_x_start}{zone_y_start}" if isinstance(zone_x_start, str) else zone_x_start
    zone_mid   = f"{zone_x_mid}{zone_y_mid}"     if isinstance(zone_x_mid, str) else zone_x_mid
    zone_end   = f"{zone_x_end}{zone_y_end}"     if isinstance(zone_x_end, str) else zone_x_end

    return {
        "length_m":    round(length_m, 2),
        "duration_s":  round(duration_s, 2),
        "mean_vel_ms": round(mean_vel, 3),
        "peak_vel_ms": round(peak_vel, 3),
        "pre_run_mean_vel_ms": (round(pre_run_mean_vel, 3)
                                 if not np.isnan(pre_run_mean_vel) else np.nan),
        "pre_run_peak_vel_ms": (round(pre_run_peak_vel, 3)
                                 if not np.isnan(pre_run_peak_vel) else np.nan),
        "pre_run_window_s":    pre_run_window_used_s,
        "x_start":     round(x_start, 2),
        "x_mid":       round(x_mid, 2),
        "x_end":       round(x_end, 2),
        "y_start":     round(y_start, 2),
        "y_mid":       round(y_mid, 2),
        "y_end":       round(y_end, 2),
        "zone_start":  zone_start,
        "third_start": zone_x_start,
        "lane_start": zone_y_start,
        "zone_mid":    zone_mid,
        "third_mid":   zone_x_mid,
        "lane_mid":    zone_y_mid,
        "zone_end":    zone_end,
        "third_end": zone_x_end,
        "lane_end": zone_y_end,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Segment matching — IoU on temporal intervals
# ─────────────────────────────────────────────────────────────────────────────

def load_automated_segments(match_id: str, auto_output_dir: str) -> pd.DataFrame:
    """
    Load the automated segmentation output for a match.

    The loop script (Loop_with_setpiece_offsets.py) writes one file per match:
        runs_behind_{match_id}.csv

    That file contains ALL halves for the match in a single CSV.
    Returns an empty DataFrame if no file is found.
    """
    pattern = os.path.join(auto_output_dir, f"runs_behind_{match_id}.csv")
    hits = glob.glob(pattern)
    if not hits:
        # Broader fallback: any CSV in auto_output_dir that contains match_id
        hits = glob.glob(os.path.join(auto_output_dir, f"*{match_id}*.csv"))
    if not hits:
        return pd.DataFrame()

    df = pd.read_csv(hits[0])
    if "segment_id" not in df.columns:
        # segment_id = stable 0-based technical identifier, used for any
        # downstream join/lookup (df.iloc[segment_id], filters, etc.)
        df.insert(0, "segment_id", range(len(df)))
    if "excel_row" not in df.columns:
        # excel_row = row number as displayed in Excel/LibreOffice
        # (+1 for the header, +1 because Excel is 1-based) — ONLY for
        # manual visual verification, never use it in the logic.
        df.insert(1, "excel_row", df.index + 2)
    return df


def _temporal_iou(a_start: float, a_end: float,
                  b_start: float, b_end: float) -> float:
    """
    Intersection-over-Union of two temporal intervals [a_start, a_end]
    and [b_start, b_end], expressed in the same unit (seconds or frames).

    Returns a value in [0, 1].  Returns 0 if either interval is degenerate.
    """
    return _shared_temporal_iou(a_start, a_end, b_start, b_end)


def find_best_segment(annot_row, df_auto, min_iou=0.01):
    """
    Find the best automated segment (max IoU) for an annotation row.

    Returns (best_seg_id, best_excel_row, best_iou, matched_row) where
    matched_row is the full pd.Series of the best candidate in df_auto
    (useful to reuse its already-computed indicators without recomputing),
    or None if no match was found / accepted.
    """
    if df_auto.empty:
        return None, None, 0.0, None

    raw_jid = annot_row["player_jid"]
    if pd.isna(raw_jid):
        print(f"    [WARN] Missing player_jid for segment {annot_row.get('segment_id', '?')}")
        return None, None, 0.0, None

    ann_jid   = int(raw_jid)
    ann_start = annot_row["start_frame"]
    ann_end   = annot_row["end_frame"]
    ann_half  = annot_row["half"]

    # ── Filter by half ───────────────────────────────────────────────────
    df_half = df_auto[df_auto["half"] == ann_half] if "half" in df_auto.columns else df_auto.copy()
    if df_half.empty:
        print(f"    [WARN] seg {annot_row['segment_id']}: no run in half='{ann_half}' "
              f"(values present: {df_auto['half'].unique() if 'half' in df_auto.columns else 'N/A'})")
        return None, None, 0.0, None

    # ── Filter by player ─────────────────────────────────────────────────
    jid_col = next((c for c in df_half.columns
                    if c.lower() in ("jid", "player_jid", "jersey")), None)

    if jid_col is None:
        print(f"    [WARN] seg {annot_row['segment_id']}: jID column not found "
              f"(columns: {list(df_half.columns)})")
        return None, None, 0.0, None

    # Compare as float to avoid any type mismatch
    df_player = df_half[df_half[jid_col].astype(float) == float(ann_jid)]

    if df_player.empty:
        uniq = sorted(df_half[jid_col].dropna().astype(int).unique())
        print(f"    [WARN] seg {annot_row['segment_id']}: jID={ann_jid} absent from runs "
              f"(players present in this half: {uniq})")
        return None, None, 0.0, None

    # ── IoU on frames ────────────────────────────────────────────────────
    inter = np.maximum(0.0,
        np.minimum(df_player["end_frame"].values,   ann_end) -
        np.maximum(df_player["start_frame"].values, ann_start))
    ann_len   = ann_end - ann_start
    cand_lens = df_player["end_frame"].values - df_player["start_frame"].values
    union = ann_len + cand_lens - inter
    iou   = np.where(union > 0, inter / union, 0.0)

    best_idx = int(np.argmax(iou))
    best_iou = float(iou[best_idx])

    if best_iou < min_iou:
        print(f"    [WARN] seg {annot_row['segment_id']}: best_iou={best_iou:.4f} < min_iou={min_iou} → not matched")
        return None, None, best_iou, None

    # ── Retrieve the segment identifier (0-based, for logic) ───────────────
    seg_id_col = next((c for c in df_player.columns
                       if c.lower() in ("segment_id", "seg_id", "xid", "id")), None)
    if seg_id_col is None:
        best_seg_id = int(df_player.index[best_idx])
    else:
        best_seg_id = int(df_player.iloc[best_idx][seg_id_col])

    # ── Corresponding Excel row (visual verification only) ─────────────────
    if "excel_row" in df_player.columns:
        best_excel_row = int(df_player.iloc[best_idx]["excel_row"])
    else:
        best_excel_row = best_seg_id + 2

    return best_seg_id, best_excel_row, round(best_iou, 4), df_player.iloc[best_idx]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Shot-follow-up flag
# ─────────────────────────────────────────────────────────────────────────────

def flag_shot_within_window(elapsed_s: float, half: str, run_team: str,
                             df_shots: pd.DataFrame,
                             window_s: float = SHOT_WINDOW_S) -> bool:
    """
    Return True if run_team produced a shot within window_s seconds after
    the run ended (same half).
    """
    if df_shots.empty or window_s is None:
        return False

    mask = (
        (df_shots["half"] == half) &
        (df_shots["team"] == run_team) &
        (df_shots["elapsed_s"] >= elapsed_s) &
        (df_shots["elapsed_s"] <= elapsed_s + window_s)
    )
    return bool(mask.any())


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Reuse indicators already computed by the automated pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Mapping "expected output INDICATOR_COLS column" -> "equivalent column in
# runs_behind_{match_id}.csv (df_auto)". The speed columns in df_auto are in
# km/h; the enrichment pipeline expects m/s, hence the (/3.6) conversion
# applied in _auto_row_to_indicators().
_AUTO_COL_MAP = {
    "auto_start_frame":       "start_frame",
    "auto_end_frame":         "end_frame",
    "length_m":            "distance_m",
    "duration_s":          "duration_s",
    "mean_vel_ms":         "avg_velocity_kmh",   # /3.6
    "peak_vel_ms":         "peak_speed_kmh",     # /3.6
    "pre_run_mean_vel_ms": "pre_run_mean_vel_kmh",  # /3.6
    "pre_run_peak_vel_ms": "pre_run_peak_vel_kmh",  # /3.6
    "pre_run_window_s":    "pre_run_window_s",
    "x_start":             "x_start",
    "y_start":             "y_start",
    "x_end":               "x_end",
    "y_end":               "y_end",
    "x_mid":               "x_mid",
    "y_mid":               "y_mid",
    "zone_start":          "zone_start",
    "zone_mid":            "zone_mid",
    "third_start":         "third_start",
    "third_mid":           "third_mid",
    "third_end":           "third_end",
    "lane_start":           "lane_start",
    "lane_mid":             "lane_mid",
    "lane_end":             "lane_end",
    "zone_end":            "zone_end",
}
# Columns whose source value is in km/h and must be divided by 3.6
# to obtain m/s.
_AUTO_KMH_COLS = {"mean_vel_ms", "peak_vel_ms", "pre_run_mean_vel_ms", "pre_run_peak_vel_ms"}


def _auto_row_to_indicators(auto_row: pd.Series, indicator_cols: list) -> dict | None:
    """
    Build an indicators dict (same keys as INDICATOR_COLS) from an
    already-computed row of df_auto (runs_behind_{match_id}.csv).

    Returns None if an expected source column is missing from the auto
    DataFrame (in which case the caller must recompute from raw positions).
    """
    out = {}
    for target_col in indicator_cols:
        src_col = _AUTO_COL_MAP.get(target_col)
        if src_col is None or src_col not in auto_row.index:
            return None
        val = auto_row[src_col]
        if target_col in _AUTO_KMH_COLS and pd.notna(val):
            val = float(val) / 3.6
        out[target_col] = val
    return out


def _indicators_complete(indicators: dict, indicator_cols: list) -> bool:
    """True if all indicator columns are present and non-NaN."""
    for col in indicator_cols:
        if col not in indicators:
            return False
        val = indicators[col]
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Process one annotation file
# ─────────────────────────────────────────────────────────────────────────────

def process_annotation_file(annot_path: str,
                             data_dir: str,

                             auto_output_dir: str        = AUTO_OUTPUT_DIR,
                             shot_window_s: float | None = SHOT_WINDOW_S,
                             min_iou:       float        = MIN_IOU,
                             pre_run_window_s: float | None = PRE_RUN_WINDOW_S,
                             reuse_auto_metrics: bool     = REUSE_AUTO_METRICS) -> pd.DataFrame:
    """
    Full enrichment pipeline for a single annotation CSV.

    Returns an enriched DataFrame (one row per annotated run).
    """
    print(f"\n{'─'*70}")
    print(f"  Processing: {os.path.basename(annot_path)}")
    print(f"{'─'*70}")

    # ── 1. Load annotations ───────────────────────────────────────────────
    df_annot = pd.read_csv(annot_path)
    required_cols = {"segment_id", "half", "start_frame", "end_frame",
                     "start_time_s", "end_time_s", "player_jid", "team"}
    missing = required_cols - set(df_annot.columns)
    if missing:
        raise ValueError(f"Annotation file is missing columns: {missing}")

    # Derive match_id from filename (DFL-MAT-XXXXXX pattern)
    fname = os.path.basename(annot_path)
    match_id = None
    import re
    m = re.search(r"DFL-MAT-([A-Z0-9]{6})", fname, re.IGNORECASE)
    if m:
        match_id = m.group(1)
    else:
        raise ValueError(
            f"Cannot derive match_id from filename '{fname}'. "
            "Expected pattern: DFL-MAT-XXXXXX"
        )

    half_label = None
    if "firstHalf" in fname or "first_half" in fname.lower():
        half_label = "firstHalf"
    elif "secondHalf" in fname or "second_half" in fname.lower():
        half_label = "secondHalf"
    # If both halves in one file, half is taken from the row's 'half' column.

    print(f"  Match: {match_id}  |  Half hint from filename: {half_label}")

    use_reuse_auto_metrics = reuse_auto_metrics

    # ── 2. Position data: LAZY loading ──────────────────────────────────────
    # We only load the XML positions + teamsheets if at least one row is not
    # already fully covered by a reused auto match — this loading is the
    # most expensive part of the pipeline (Butterworth + VelocityModel).
    _pos_lazy = {"xy_dict": None, "pid_to_xid": None, "loaded": False}

    def _get_position_data():
        if not _pos_lazy["loaded"]:
            print("  Loading position data (no auto match available → computation needed) …")
            xy_dict_, teamsheets_, _pitch_ = load_position_data(match_id, data_dir)
            _pos_lazy["xy_dict"]     = xy_dict_
            _pos_lazy["pid_to_xid"]  = build_person_id_to_xid(teamsheets_)
            _pos_lazy["loaded"]      = True
        return _pos_lazy["xy_dict"], _pos_lazy["pid_to_xid"]

    # ── 3. Player map: parse matchinfo XML directly ───────────────────────
    # df_players: one row per player with person_id, jID, playing_position, …
    path_info   = find_dfl_file(data_dir, match_id, "matchinformation")
    df_players  = parse_matchinfo_players(path_info)

    # Build jID → person_id lookup (within each team to avoid cross-team
    # shirt-number collisions, e.g. both teams can have a #11)
    # Result: {(team_loc, jID) → person_id}
    jid_to_pid = {}
    for _, p in df_players.iterrows():
        if p["jID"] is not None:
            jid_to_pid[(p["team_loc"], int(p["jID"]))] = p["person_id"]
    # Also build a jID-only fallback (last writer wins — use with caution)
    jid_to_pid_fallback = {}
    for _, p in df_players.iterrows():
        if p["jID"] is not None:
            jid_to_pid_fallback[int(p["jID"])] = p["person_id"]

    print(f"  Player map: {len(df_players)} players "
          f"(mapping to xID loaded on demand, on auto-match miss)")

    # ── 4. Event data (goals + shots) ─────────────────────────────────────
    print("  Parsing event data …")
    events = parse_events_xml(match_id, data_dir)
    df_goals = events["goals"]
    df_shots = events["shots"]

    # ── 5. Automated segment index (whole match) ──────────────────────────
    df_auto = load_automated_segments(match_id, auto_output_dir)
    if df_auto.empty:
        print(f"  [WARN] No automated segment file found for {match_id} "
              f"in {auto_output_dir} — auto_segment_id will be NaN for all rows.")

    # ── 6. Iterate over annotations ───────────────────────────────────────

    # Helper: safe integer cast — returns None (→ NaN in DataFrame) for
    # any value that is missing or cannot be converted.
    def _safe_int(val):
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return int(val)
        except (TypeError, ValueError):
            return None

    def _safe_float(val, ndigits=2):
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return np.nan
            return round(float(val), ndigits)
        except (TypeError, ValueError):
            return np.nan

    INDICATOR_COLS = ["auto_start_frame", "auto_end_frame",
                    "length_m", "duration_s", "mean_vel_ms", "peak_vel_ms",
                      "pre_run_mean_vel_ms", "pre_run_peak_vel_ms", "pre_run_window_s",
                      "x_start", "y_start", 
                      "x_end", "y_end",
                      "x_mid", "y_mid", 
                      "zone_start","third_start", "lane_start",
                      "zone_mid", "third_mid", "lane_mid",
                      "zone_end", "third_end", "lane_end"]

    enriched_rows = []
    n_reused_from_auto = 0

    for row_idx, row in df_annot.iterrows():
        half = row["half"]

        # ── 6a. Find matching automated segment (best IoU) ───────────────
        auto_seg_id, auto_excel_row, match_iou, auto_matched_row = find_best_segment(
            row, df_auto, min_iou=min_iou
        )
        # ── 6b. Resolve PersonId from jID + team ─────────────────────────
        jid_val = _safe_int(row["player_jid"])
        sf_val  = _safe_int(row["start_frame"])
        ef_val  = _safe_int(row["end_frame"])

        # Primary lookup: (team_loc, jID) → person_id  (no cross-team ambiguity)
        team_loc  = row["team"]   # "Home" or "Away"
        person_id = None
        if jid_val is not None:
            person_id = jid_to_pid.get((team_loc, jid_val))
            if person_id is None:
                # Fallback: ignore team side (handles "guest"/"away" mismatches)
                person_id = jid_to_pid_fallback.get(jid_val)
                if person_id is not None:
                    print(f"    [INFO] segment {row['segment_id']}: resolved jID={jid_val} "
                          f"via fallback (team side mismatch?)")

        if person_id is None:
            print(f"    [WARN] Row {row_idx} (segment {row['segment_id']}) — "
                  f"could not resolve PersonId for jID={jid_val}, team={team_loc}")

        # ── 6c. Kinematic indicators — reused from df_auto if possible ──────
        # If the annotation is matched to an automated movement, the latter
        # already has all these indicators computed (Discretisation_optimised2_1.py)
        # → reuse them directly, without reloading the XML positions.
        indicators = None
        duration_source = None
        if use_reuse_auto_metrics and auto_matched_row is not None:
            indicators = _auto_row_to_indicators(auto_matched_row, INDICATOR_COLS)
            if indicators is not None and _indicators_complete(indicators, INDICATOR_COLS):
                n_reused_from_auto += 1
                duration_source = "reused_from_auto_csv"
            else:
                indicators = None   # missing/NaN columns → will recompute

        if indicators is None:
            # Prefer the frame range of the matched automated segment (the
            # actual detected movement) when one exists, even if we ended up
            # here because its indicators were missing/incomplete in
            # df_auto. Only fall back to the annotation's own frames when
            # there is truly no automated match — in that case duration_s
            # (and the other indicators) will legitimately coincide with
            # the annotation window, since it's the only one available.
            if auto_matched_row is not None:
                recompute_sf = _safe_int(auto_matched_row["start_frame"])
                recompute_ef = _safe_int(auto_matched_row["end_frame"])
                frame_source = "auto_segment"
            else:
                recompute_sf = sf_val
                recompute_ef = ef_val
                frame_source = "annotation"

            if person_id is None or recompute_sf is None or recompute_ef is None:
                indicators = {k: np.nan for k in INDICATOR_COLS}
                duration_source = "missing"
            else:
                xy_dict, pid_to_xid = _get_position_data()
                indicators = extract_run_indicators(
                    start_frame  = recompute_sf,
                    end_frame    = recompute_ef,
                    half         = half,
                    person_id    = person_id,
                    xy_dict      = xy_dict,
                    pid_to_xid   = pid_to_xid,
                    match_id     = match_id,
                    team         = team_loc,
                    pre_window_s = pre_run_window_s,
                )
                if indicators is None:
                    print(f"    [WARN] No position data for segment {row['segment_id']} "
                          f"(PersonId={person_id}, {half}, frames from {frame_source})")
                    indicators = {k: np.nan for k in INDICATOR_COLS}
                    duration_source = "missing"
                else:
                    duration_source = f"recomputed_from_{frame_source}"

        # ── 6d. Designated position from matchinfo ────────────────────────
        if person_id is not None:
            p_row = df_players[df_players["person_id"] == person_id]
            designated_position = (
                p_row.iloc[0]["playing_position"] if not p_row.empty else np.nan
            )
        elif jid_val is not None:
            # Last-resort: match on jID alone (may be ambiguous if both teams share number)
            p_rows = df_players[df_players["jID"] == jid_val]
            designated_position = (
                p_rows.iloc[0]["playing_position"] if not p_rows.empty else np.nan
            )
        else:
            designated_position = np.nan

        # ── 6e. Scoreline at run start ────────────────────────────────────
        scoreline = get_scoreline_at(
            _safe_float(row["start_time_s"]), half, df_goals
        )
        goal_indicator = get_goal_indicator(scoreline, row["team"])

        # ── 6f. Shot-follow-up flag ───────────────────────────────────────
        shot_flag = False
        if shot_window_s is not None:
            shot_flag = flag_shot_within_window(
                elapsed_s  = _safe_float(row["end_time_s"]),
                half       = half,
                run_team   = row["team"],
                df_shots   = df_shots,
                window_s   = shot_window_s,
    )

        # ── 6g. Assemble output row ───────────────────────────────────────
        out = {
            # Identifiers
            "annotation_segment_id": _safe_int(row["segment_id"]),
            "auto_segment_id":       auto_seg_id,          # 0-based ID, use for any analysis/join
            "auto_segment_excel_row": auto_excel_row,       # Excel row — visual check only
            "match_iou":             match_iou if auto_seg_id is not None else np.nan,
            "match_id":              match_id,
            "half":                  half,
            "annotation_source":     row.get("annotation_source", np.nan),

            # Player / team
            "player":                row.get("player", np.nan),
            "player_jid":            jid_val,
            "person_id":             person_id,
            "team":                  row["team"],
            "designated_position":   designated_position,

            # Timing (from annotation)
            "start_frame":           sf_val,
            "end_frame":             ef_val,
            "start_time_s":          _safe_float(row["start_time_s"]),
            "end_time_s":            _safe_float(row["end_time_s"]),
            "duration_annot_s":      (
                _safe_float(row.get("duration_s"))
                if pd.notna(row.get("duration_s"))
                else _safe_float(row["end_time_s"] - row["start_time_s"])
            ),

            # Outcome (if present)
            "outcome":               row.get("outcome", np.nan),

            # Kinematic indicators (from position data)
            **indicators,
            "duration_source":       duration_source,  # reused_from_auto_csv / recomputed_from_auto_segment / recomputed_from_annotation / missing

            # Context
            "scoreline_at_run_start": scoreline,
            "goal_indicator":         goal_indicator,
        }

        if shot_window_s is not None:
            out[f"shot_within_{int(shot_window_s)}s"] = shot_flag

        enriched_rows.append(out)

    df_enriched = pd.DataFrame(enriched_rows)

    # ── 7. Summary ────────────────────────────────────────────────────────
    n_matched = df_enriched["auto_segment_id"].notna().sum()
    print(f"  ✓ {len(df_enriched)} runs enriched "
          f"({n_matched} matched to automated segments, "
          f"{n_reused_from_auto} indicators reused from runs_behind_*.csv "
          f"— no recomputation from raw positions)")

    return df_enriched


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect annotation files
    annot_files = sorted(
        glob.glob(os.path.join(ANNOTATION_DIR, ANNOTATION_GLOB))
    )
    if not annot_files:
        raise FileNotFoundError(
            f"No annotation files found in '{ANNOTATION_DIR}' "
            f"matching pattern '{ANNOTATION_GLOB}'"
        )

    print(f"Found {len(annot_files)} annotation file(s) to process.")

    all_enriched = []

    for annot_path in annot_files:
        try:
            df_enriched = process_annotation_file(
                annot_path       = annot_path,
                data_dir         = DATA_DIR,
                auto_output_dir  = AUTO_OUTPUT_DIR,
                shot_window_s    = SHOT_WINDOW_S,
                min_iou          = MIN_IOU,
                pre_run_window_s = PRE_RUN_WINDOW_S,
                reuse_auto_metrics = REUSE_AUTO_METRICS,
            )

            # Per-file output
            stem     = Path(annot_path).stem
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_enriched.csv")
            df_enriched.to_csv(out_path, index=False)
            print(f"  → Saved: {out_path}")

            all_enriched.append(df_enriched)

        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {os.path.basename(annot_path)}: {exc}")
            traceback.print_exc()

    # Combined output
    if len(all_enriched) > 1:
        df_all = pd.concat(all_enriched, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_annotations_enriched.csv")
        df_all.to_csv(combined_path, index=False)
        print(f"\n✓ Combined file saved: {combined_path}")
        print(f"  Total runs: {len(df_all)}")

    print("\nDone.")


if __name__ == "__main__":
    main()