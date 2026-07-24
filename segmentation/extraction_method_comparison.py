"""
compare_discretisation.py
=========================
Visual comparison of the OLD vs NEW velocity segmentation (discretisation)
approaches on DFL tracking data for a specific player and time window.

OLD approach : simple valley detection only (old_code/extract_discrete_efforts.ipynb)
NEW approach : valley detection + metabolic-power sub-segmentation
               + vectorised valley filtering
               + setpiece blackout exclusion (Discretisation_optimised2_1.py)

Usage
-----
Adjust the CONFIG block below, then run:
    python compare_discretisation.py
"""

# ============================================================================
# Standard library
# ============================================================================
import os
import sys
import json
import xml.etree.ElementTree as ET

# ============================================================================
# Third-party
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.signal import find_peaks

import floodlight.io.dfl as dfl
from floodlight.models.kinematics import VelocityModel
from floodlight.transforms.filter import butterworth_lowpass

# ============================================================================
# Local — NEW pipeline module (must be importable from this script's location)
# Adjust the path below if Discretisation_optimised2_1.py lives elsewhere.
# ============================================================================
# sys.path.insert(0, r"C:\path\to\your\pipeline")
from Discretisation_optimised2_1 import (
    extract_movements_for_player   as new_extract_movements,
    split_segment_by_power,
    build_setpiece_blackout_frames,
    _apply_blackout_mask,
    _build_time_anchors,
    classify_speed_category,
    SPEED_THRESHOLDS,
    SETPIECE_OFFSETS,
)

from common import config
from common.filters import RIB_MIN_DISTANCE_M, RIB_MIN_DIRECTION, RIB_SPEED_CATEGORIES


# ============================================================================
# ── CONFIG  (edit these freely) ─────────────────────────────────────────────
# ============================================================================

MATCH_ID     = "J03WMX"
PLAYER_NAME  = "Serge Gnabry"
HALF         = "firstHalf"          # "firstHalf" or "secondHalf"
START_TIME   = "07:30"              # MM:SS  (match time, NOT UTC)
END_TIME     = "08:00"              # MM:SS

DATA_DIR  = config.DATA_DIR
JSON_PATH = config.DIRECTION_JSON

# Butterworth filter parameters (must match your pipeline settings)
BUTTERWORTH_Wn    = 0.5
BUTTERWORTH_ORDER = 1
FRAMERATE         = 25

# NEW pipeline segmentation parameters
MIN_VALLEY_DISTANCE = 15
VALLEY_DEPTH_ABS    = 0.15
VALLEY_DEPTH_REL    = 0.05
POWER_THRESHOLD     = 6.0
MIN_SEGMENT_FRAMES  = 10

# OLD pipeline parameter
OLD_MIN_VALLEY_DISTANCE = 10

# Run-in-behind classification criteria -- imported from common.filters (the
# single source of truth) instead of a local copy, which had drifted to
# distance_m > 3.0 vs the production value of > 2.5 used everywhere else.


# ============================================================================
# ── Helper: MM:SS  →  half-local frame index ─────────────────────────────────
# ============================================================================

def mmss_to_seconds(mmss: str) -> float:
    """Convert 'MM:SS' string to total seconds (float)."""
    parts = mmss.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected MM:SS format, got '{mmss}'")
    return int(parts[0]) * 60 + float(parts[1])


def mmss_to_frame(mmss: str, half: str, kickoff_frames: dict,
                  framerate: int = 25) -> int:
    """
    Convert a match-time 'MM:SS' string to the *half-local* frame index that
    corresponds to that match timestamp.

    For firstHalf  : frame = kickoff_frames['firstHalf']  + seconds * framerate
    For secondHalf : seconds are counted from 45:00, so we subtract 45*60 first.
                     frame = kickoff_frames['secondHalf'] + (seconds-2700)*framerate
    """
    total_s = mmss_to_seconds(mmss)
    if half == "firstHalf":
        return kickoff_frames["firstHalf"] + int(round(total_s * framerate))
    else:
        return kickoff_frames["secondHalf"] + int(round((total_s - 45 * 60) * framerate))


def frame_to_mmss(frame: int, half: str, kickoff_frames: dict,
                  framerate: int = 25) -> str:
    """Convert a half-local frame index back to a 'MM:SS' match-time string."""
    if half == "firstHalf":
        total_s = (frame - kickoff_frames["firstHalf"]) / framerate
    else:
        total_s = (frame - kickoff_frames["secondHalf"]) / framerate + 45 * 60
    total_s = max(0.0, total_s)
    mins    = int(total_s) // 60
    secs    = int(total_s) % 60
    return f"{mins:02d}:{secs:02d}"


# ============================================================================
# ── Data loading helpers ─────────────────────────────────────────────────────
# ============================================================================

def _find_file(data_dir: str, match_id: str, keyword: str) -> str:
    """Return the first file in data_dir whose name contains both match_id and keyword."""
    candidates = [
        f for f in os.listdir(data_dir)
        if match_id in f and keyword in f
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No file containing '{match_id}' and '{keyword}' found in {data_dir}"
        )
    return os.path.join(data_dir, candidates[0])


def load_match_data(match_id: str, data_dir: str):
    """
    Load DFL position and event data for a single match.
    Returns (xy, possession, teamsheets, path_events, path_info).
    """
    path_positions = _find_file(data_dir, match_id, "positions")
    path_events    = _find_file(data_dir, match_id, "events")
    path_info      = _find_file(data_dir, match_id, "matchinformation")

    print(f"[load] positions : {os.path.basename(path_positions)}")
    print(f"[load] events    : {os.path.basename(path_events)}")
    print(f"[load] info      : {os.path.basename(path_info)}")

    xy, possession, _ballstatus, teamsheets, _pitch = dfl.read_position_data_xml(
        path_positions, path_info,
        teamsheet_home=None, teamsheet_away=None,
    )
    return xy, possession, teamsheets, path_events, path_info


def find_player(player_name: str, teamsheets: dict) -> tuple[str, int, object]:
    """
    Locate a player by name across all teamsheets.
    Returns (team_name, xID, teamsheet_row).
    Raises ValueError if not found.
    """
    for team_name, ts_obj in teamsheets.items():
        ts = ts_obj.teamsheet
        ts_obj.add_xIDs()
        ts = ts_obj.teamsheet  # refresh after add_xIDs
        match = ts[ts["player"] == player_name]
        if not match.empty:
            xid = int(match["xID"].iloc[0])
            return team_name, xid, match.iloc[0]
    raise ValueError(
        f"Player '{player_name}' not found in any teamsheet. "
        f"Available players: {[n for ts in teamsheets.values() for n in ts.teamsheet['player'].tolist()]}"
    )


# ============================================================================
# ── OLD segmentation logic (ported directly from the notebook) ───────────────
# ============================================================================

def old_extract_movements_for_player(
        player_id, velocities, tracking_data, possession,
        min_valley_distance: int = 10,
):
    """
    Verbatim reproduction of the notebook's valley-only segmentation.
    No depth filtering, no metabolic-power sub-split, no setpiece blackout.
    """
    player_velocity = velocities[:, player_id]
    vel_for_peaks   = np.where(np.isnan(player_velocity), 0.0, player_velocity)

    valleys, _ = find_peaks(-vel_for_peaks, distance=min_valley_distance)

    boundaries = np.unique(np.concatenate([[0], valleys, [len(player_velocity) - 1]]))

    movements = []
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
        movements.append(pd.DataFrame({
            "frame": frames, "x": x, "y": y, "velocity": v,
            "possession": p, "movement_id": i,
        }))
    return movements


# ============================================================================
# ── Shared movement summariser ───────────────────────────────────────────────
# ============================================================================

def summarise_movement(seg_df: pd.DataFrame, framerate: int = 25,
                       location: str = "Home") -> dict:
    """
    Compute the same summary statistics used in process_match so that
    run-in-behind criteria can be evaluated consistently on both methods.
    """
    v_ms   = seg_df["velocity"].values
    x_vals = seg_df["x"].values
    y_vals = seg_df["y"].values

    valid_v      = np.where(np.isnan(v_ms), -np.inf, v_ms)
    peak_idx     = int(np.argmax(valid_v))
    peak_kmh     = float(v_ms[peak_idx]) * 3.6 if not np.isnan(v_ms[peak_idx]) else 0.0
    dx           = np.diff(x_vals)
    dy           = np.diff(y_vals)
    distance_m   = float(np.sqrt(dx * dx + dy * dy).sum())

    poss_vals        = seg_df["possession"].values
    home_ratio       = float((poss_vals == 1).mean())
    possession_label = "Home" if home_ratio >= 0.5 else "Away"
    contested        = 0.35 < home_ratio < 0.65

    return {
        "start_frame":          int(seg_df["frame"].iloc[0]),
        "end_frame":            int(seg_df["frame"].iloc[-1]),
        "peak_speed_kmh":       round(peak_kmh, 2),
        "speed_category":       classify_speed_category(peak_kmh),
        "distance_m":           round(distance_m, 2),
        "x_start":              float(x_vals[0]),
        "y_start":              float(y_vals[0]),
        "x_end":                float(x_vals[-1]),
        "y_end":                float(y_vals[-1]),
        "possession":           possession_label,
        "possession_contested": contested,
        "location":             location,
    }


# ============================================================================
# ── Run-in-behind classifier ─────────────────────────────────────────────────
# ============================================================================

def is_run_in_behind(seg_summary: dict, dict_direction: dict,
                     match_id: str, half: str, location: str) -> bool:
    """
    Return True if the segment satisfies all run-in-behind criteria:
      - speed_category in {running, sprinting}
      - distance_m > RIB_MIN_DISTANCE_M
      - direction  > RIB_MIN_DIRECTION
      - possession == location OR possession_contested
    """
    cat = seg_summary.get("speed_category", "")
    if cat not in RIB_SPEED_CATEGORIES:
        return False
    if seg_summary.get("distance_m", 0) <= RIB_MIN_DISTANCE_M:
        return False

    # Compute direction (L1-normalised x-component, signed by attack direction)
    dx = seg_summary["x_end"] - seg_summary["x_start"]
    dy = seg_summary["y_end"] - seg_summary["y_start"]
    denom = abs(dx) + abs(dy)
    base_dir = dx / denom if denom > 0 else 0.0

    home_direction = dict_direction.get(match_id, {}).get("Home", {}).get(half, None)
    if location == "Away":
        home_direction = (
            "right_to_left" if home_direction == "left_to_right"
            else "left_to_right" if home_direction == "right_to_left"
            else home_direction
        )
    multiplier = 1 if home_direction == "left_to_right" else -1
    direction  = base_dir * multiplier

    if direction <= RIB_MIN_DIRECTION:
        return False

    poss      = seg_summary.get("possession", "")
    contested = seg_summary.get("possession_contested", False)
    if poss != location and not contested:
        return False

    return True


# ============================================================================
# ── Core comparison logic ────────────────────────────────────────────────────
# ============================================================================

def run_comparison(
        match_id:    str,
        player_name: str,
        half:        str,
        start_time:  str,
        end_time:    str,
        data_dir:    str,
        json_path:   str,
):
    """
    Main entry point.
    Loads data, applies both segmentation methods, builds the comparison figure.
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MATCH   : {match_id}")
    print(f"  PLAYER  : {player_name}")
    print(f"  HALF    : {half}   window: {start_time} → {end_time}")
    print("=" * 70)

    xy, possession_raw, teamsheets, path_events, path_info = load_match_data(
        match_id, data_dir
    )

    with open(json_path, "r") as fh:
        dict_direction = json.load(fh)

    # ── 2. Locate player ──────────────────────────────────────────────────────
    team, xid, player_row = find_player(player_name, teamsheets)
    print(f"\n[player] '{player_name}' → team={team}, xID={xid}, "
          f"position={player_row.get('position', 'N/A')}")

    # ── 3. Build time anchors (kickoff frames per half) ────────────────────────
    anchors       = _build_time_anchors(path_events, path_info, framerate=FRAMERATE)
    kickoff_frames = anchors["kickoff_frames"]

    # ── 4. Convert MM:SS to half-local frame indices ──────────────────────────
    frame_start = mmss_to_frame(start_time, half, kickoff_frames, FRAMERATE)
    frame_end   = mmss_to_frame(end_time,   half, kickoff_frames, FRAMERATE)

    n_half_frames = xy[half][team].xy.shape[0]
    frame_start   = max(0, min(frame_start, n_half_frames - 1))
    frame_end     = max(0, min(frame_end,   n_half_frames - 1))

    if frame_start >= frame_end:
        raise ValueError(
            f"Empty or inverted window: start_frame={frame_start}, end_frame={frame_end}. "
            f"Check that {start_time} < {end_time} and both are within the {half}."
        )

    print(f"\n[window] {start_time}→{end_time}  "
          f"= half-local frames [{frame_start}, {frame_end}]  "
          f"({(frame_end - frame_start) / FRAMERATE:.1f}s, "
          f"{frame_end - frame_start} frames)")

    # ── 5. Butterworth-filtered velocity for the whole half ───────────────────
    xy_half   = xy[half][team]
    xy_filt   = butterworth_lowpass(
        xy_half, remove_short_seqs=True,
        Wn=BUTTERWORTH_Wn, order=BUTTERWORTH_ORDER,
    )
    vm = VelocityModel()
    vm.fit(xy_filt)
    vel_full = vm.velocity().property          # shape (n_frames, n_players)
    xy_filt_arr = xy_filt.xy                   # shape (n_frames, 2*n_players)

    # ── 6. Check the player has data in the window ────────────────────────────
    player_vel_window = vel_full[frame_start:frame_end, xid]
    if np.all(np.isnan(player_vel_window)):
        print(f"\n[WARNING] '{player_name}' has no valid tracking data between "
              f"{start_time} and {end_time}. Nothing to plot.")
        return

    # ── 7. Build setpiece blackout intervals (NEW only) ───────────────────────
    blackout_intervals = build_setpiece_blackout_frames(
        path_events, path_info, framerate=FRAMERATE, anchors=anchors
    )

    # ── 8. Run OLD segmentation on the full half, then filter to window ───────
    possession_half   = np.asarray(possession_raw[half]).flatten().astype(int)
    old_movements_all = old_extract_movements_for_player(
        xid, vel_full, xy_filt_arr, possession_half,
        min_valley_distance=OLD_MIN_VALLEY_DISTANCE,
    )
    old_movements = [
        m for m in old_movements_all
        if not m.empty
        and m["frame"].iloc[-1] >= frame_start
        and m["frame"].iloc[0]  <= frame_end
    ]

    # ── 9. Run NEW segmentation on the full half, then filter to window ───────
    new_movements_all = new_extract_movements(
        xid, vel_full, xy_filt_arr, possession_half,
        min_valley_distance=MIN_VALLEY_DISTANCE,
        valley_depth_abs=VALLEY_DEPTH_ABS,
        valley_depth_rel=VALLEY_DEPTH_REL,
        power_threshold=POWER_THRESHOLD,
        min_segment_frames=MIN_SEGMENT_FRAMES,
        framerate=FRAMERATE,
    )

    # Apply setpiece blackout to NEW movements: discard segments whose START
    # falls inside a blackout window (same logic as _apply_blackout_mask).
    def in_blackout(seg_df: pd.DataFrame, half: str,
                    blackout_intervals: dict) -> bool:
        ivs = blackout_intervals.get(half, np.empty((0, 2), dtype=np.int64))
        if len(ivs) == 0:
            return False
        sf = int(seg_df["frame"].iloc[0])
        idx = np.searchsorted(ivs[:, 0], sf, side="right") - 1
        return bool(idx >= 0 and sf < ivs[idx, 1])

    new_movements = [
        m for m in new_movements_all
        if not m.empty
        and m["frame"].iloc[-1] >= frame_start
        and m["frame"].iloc[0]  <= frame_end
        and not in_blackout(m, half, blackout_intervals)
    ]

    print(f"\n[segments in window]  OLD: {len(old_movements)}   NEW: {len(new_movements)}")

    if not old_movements and not new_movements:
        print("[WARNING] No segments found in the time window for either method.")
        return

    # ── 10. Summarise all segments + classify runs-in-behind ─────────────────
    def summarise_all(movements, label):
        summaries = []
        for seg in movements:
            s = summarise_movement(seg, framerate=FRAMERATE, location=team)
            s["is_rib"] = is_run_in_behind(s, dict_direction, match_id, half, team)
            s["label"]  = label
            summaries.append(s)
        return summaries

    old_summaries = summarise_all(old_movements, "OLD")
    new_summaries = summarise_all(new_movements, "NEW")

    n_rib_old = sum(s["is_rib"] for s in old_summaries)
    n_rib_new = sum(s["is_rib"] for s in new_summaries)
    print(f"[runs-in-behind]      OLD: {n_rib_old}   NEW: {n_rib_new}")

    # ── 11. Build x-axis: half-local frames in the window ────────────────────
    frames_window = np.arange(frame_start, frame_end)
    vel_window_ms = vel_full[frame_start:frame_end, xid]          # m/s
    vel_window_kmh = vel_window_ms * 3.6                           # km/h (for readability)

    def x_tick_formatter(frame_val, pos=None):
        return frame_to_mmss(int(frame_val), half, kickoff_frames, FRAMERATE)

    # ── 12. Build the figure ──────────────────────────────────────────────────
    fig, (ax_old, ax_new) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    fig.patch.set_facecolor("#0E1117")
    for ax in (ax_old, ax_new):
        ax.set_facecolor("#0E1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── Colour palette ────────────────────────────────────────────────────────
    COLOR_VEL       = "#A8C7FA"   # velocity curve
    COLOR_SEG_NORM  = "#4A90D9"   # normal segment fill
    COLOR_SEG_RIB   = "#FF6B35"   # run-in-behind fill (vivid orange-red)
    COLOR_BREAK_OLD = "#FFD700"   # OLD valley markers
    COLOR_BREAK_NEW = "#00E5A0"   # NEW valley / power-split markers
    COLOR_SPEED_LINE = "#666688"  # speed threshold dashed lines

    # Speed threshold horizontal lines (km/h)
    threshold_lines = [
        (SPEED_THRESHOLDS["walking"],  "walk/jog",  "7 km/h"),
        (SPEED_THRESHOLDS["jogging"],  "jog/run",  "14 km/h"),
        (SPEED_THRESHOLDS["running"],  "run/sprint","21 km/h"),
    ]

    def draw_panel(ax, movements, summaries, breaks_color, panel_label, method_desc):
        """
        Draw one subplot: velocity curve, coloured segment fills, break markers.
        """
        # Velocity curve
        ax.plot(frames_window, vel_window_kmh,
                color=COLOR_VEL, linewidth=1.1, alpha=0.9, zorder=3,
                label="Velocity (km/h)")

        # Speed threshold lines
        for thresh_kmh, _, thresh_label in threshold_lines:
            ax.axhline(thresh_kmh, color=COLOR_SPEED_LINE, linestyle="--",
                       linewidth=0.7, alpha=0.6, zorder=1)
            ax.text(frame_start, thresh_kmh + 0.2, thresh_label,
                    color=COLOR_SPEED_LINE, fontsize=6.5, va="bottom", zorder=2)

        # Segment fills + break markers
        break_positions = []
        for seg_df, summary in zip(movements, summaries):
            sf = int(seg_df["frame"].iloc[0])
            ef = int(seg_df["frame"].iloc[-1])

            # Clip to visible window
            sf_vis = max(sf, frame_start)
            ef_vis = min(ef, frame_end - 1)
            if sf_vis >= ef_vis:
                continue

            fill_color = COLOR_SEG_RIB if summary["is_rib"] else COLOR_SEG_NORM
            ax.axvspan(sf_vis, ef_vis, alpha=0.25 if summary["is_rib"] else 0.12,
                       color=fill_color, zorder=2)

            # Speed category label at mid-segment
            mid_frame = (sf_vis + ef_vis) // 2
            mid_vel   = np.nanmax(vel_window_kmh[
                max(0, sf_vis - frame_start):ef_vis - frame_start + 1
            ])
            ax.text(mid_frame, mid_vel + 1.0,
                    summary["speed_category"][0].upper(),  # W/J/R/S
                    ha="center", va="bottom", fontsize=6.5,
                    color=COLOR_SEG_RIB if summary["is_rib"] else "#AAAACC",
                    fontweight="bold", zorder=5)

            # Segment boundary (break) at the START of each segment
            if sf >= frame_start:
                break_positions.append(sf)

        # Draw break markers as vertical dotted lines + 'x' scatter
        if break_positions:
            for bp in break_positions:
                ax.axvline(bp, color=breaks_color, linestyle=":",
                           linewidth=1.0, alpha=0.8, zorder=4)
            vel_at_breaks = np.interp(
                break_positions,
                frames_window,
                np.where(np.isnan(vel_window_kmh), 0.0, vel_window_kmh),
            )
            ax.scatter(break_positions, vel_at_breaks,
                       marker="x", s=60, color=breaks_color,
                       linewidths=1.5, zorder=6, label="Segment break")

        # Blackout shading on NEW panel (for context, if any blackout in window)
        if "NEW" in panel_label:
            ivs_half = blackout_intervals.get(half, np.empty((0, 2), dtype=np.int64))
            for row in ivs_half:
                bk_s = max(int(row[0]), frame_start)
                bk_e = min(int(row[1]), frame_end)
                if bk_s < bk_e:
                    ax.axvspan(bk_s, bk_e, alpha=0.18, color="#FF3366",
                               zorder=1, label="Setpiece blackout")

        # Panel label + stats
        n_rib  = sum(s["is_rib"] for s in summaries)
        n_segs = len(summaries)
        ax.set_title(
            f"{panel_label}  ·  {n_segs} segments  ·  {n_rib} run(s)-in-behind  "
            f"|  {method_desc}",
            color="#EEEEEE", fontsize=9.5, loc="left", pad=5,
        )
        ax.set_ylabel("Speed (km/h)", color="#CCCCCC", fontsize=8)
        ax.tick_params(colors="#AAAAAA", labelsize=7.5)
        ax.yaxis.label.set_color("#CCCCCC")

        # Y-limit: a bit above the max speed in window
        max_v = float(np.nanmax(vel_window_kmh)) if not np.all(np.isnan(vel_window_kmh)) else 30.0
        ax.set_ylim(-0.5, max_v * 1.15 + 2)

        # Legend (deduplicated)
        handles, labels_leg = ax.get_legend_handles_labels()
        seen  = {}
        uniq  = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.update({l: True}))]
        uniq.append((mpatches.Patch(color=COLOR_SEG_RIB, alpha=0.5),   "Run-in-behind"))
        uniq.append((mpatches.Patch(color=COLOR_SEG_NORM, alpha=0.35), "Other segment"))
        if any("blackout" in lb.lower() for _, lb in uniq):
            uniq.append((mpatches.Patch(color="#FF3366", alpha=0.3), "Setpiece blackout"))

        ax.legend(
            *zip(*uniq) if uniq else ([], []),
            loc="upper right", fontsize=7,
            facecolor="#1A1D24", edgecolor="#444444", labelcolor="#CCCCCC",
        )

    draw_panel(
        ax_old, old_movements, old_summaries,
        breaks_color=COLOR_BREAK_OLD,
        panel_label="OLD METHOD",
        method_desc="Valley detection only  (min_distance=10 fr, no depth filter, no power split, no blackout)",
    )
    draw_panel(
        ax_new, new_movements, new_summaries,
        breaks_color=COLOR_BREAK_NEW,
        panel_label="NEW METHOD",
        method_desc=(
            f"Valley detection (depth-filtered, min_dist={MIN_VALLEY_DISTANCE} fr)  "
            f"+ metabolic-power sub-split  + setpiece blackout"
        ),
    )

    # ── Shared x-axis formatting ──────────────────────────────────────────────
    ax_new.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax_new.xaxis.set_major_formatter(mticker.FuncFormatter(x_tick_formatter))
    ax_new.set_xlabel("Match time (MM:SS)", color="#CCCCCC", fontsize=8)
    plt.setp(ax_new.get_xticklabels(), rotation=20, ha="right")

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Discretisation comparison  ·  {player_name}  ·  {match_id}  ·  "
        f"{half}  {START_TIME}–{END_TIME}",
        color="#FFFFFF", fontsize=11, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    print("\n[done] Figure displayed.")


# ============================================================================
# ── Entry point ──────────────────────────────────────────────────────────────
# ============================================================================

if __name__ == "__main__":
    run_comparison(
        match_id    = MATCH_ID,
        player_name = PLAYER_NAME,
        half        = HALF,
        start_time  = START_TIME,
        end_time    = END_TIME,
        data_dir    = DATA_DIR,
        json_path   = JSON_PATH,
    )