"""
╔══════════════════════════════════════════════════════════════════╗
║   Inter-Annotator Analysis – Runs-in-Behind                     ║
║   David · Simon · Frederik  (Frederik absent in secondHalf)     ║
╚══════════════════════════════════════════════════════════════════╝

Metrics computed per pair and per half:
  • Agreement rate (recall_a, recall_b)
  • Mean/std absolute temporal error on start_frame & end_frame
  • Mean temporal IoU on matched segments
  • List of isolated segments (no match)

Generated visualisations:
  [1] dashboard_metrics.png      — Summary dashboard
  [2] player_heatmap.png         — Segment heatmap per player
  [3] stats_table.png            — Styled metrics table
  [4] iou_distribution.png       — IoU distribution by pair
  [5] ribbon_{half}.png          — Sankey ribbons between matched segments
  [6] arcs_{half}.png            — Swim-lane with Bézier arcs
  [7] confusion_matrices.png     — Confusion matrices (n_matched & mean IoU)
"""

import os
import itertools
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as _mcm
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

from common import config
from common.iou import temporal_iou as _shared_temporal_iou
from common.timecodes import frames_to_timecode
from common.annotations import discover_match_annotation_files

# ══════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
DATA_DIR   = config.ANNOTATION_DIR
MATCH_ID   = "DFL-MAT-J03WMX"
TOLERANCE  = 0
FPS        = 25                          # frames per second → used for timecode conversion
OUTPUT_DIR = config.STATS_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Presentation theme ─────────────────────────────────────────────
BG_DARK     = "#0d1b2a"          # dark figure background
BG_PANEL    = "#162032"          # panel background
ACCENT_BLUE = "#4e9af1"
ACCENT_TEAL = "#2ec4b6"
ACCENT_CORAL= "#e76f51"
ACCENT_GOLD = "#f4a261"
ACCENT_LIME = "#8ecae6"
TEXT_MAIN   = "#e8edf2"
TEXT_DIM    = "#8899aa"
GRID_LINE   = "#1e2d3d"

PRESENTATION_PALETTE = [
    ACCENT_BLUE, ACCENT_TEAL, ACCENT_CORAL, ACCENT_GOLD,
    "#a8dadc", "#457b9d", "#e9c46a", "#f77f00",
    "#84a98c", "#b5838d", "#6d6875", "#c9ada7",
]

matplotlib.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "text.color"       : TEXT_MAIN,
    "axes.labelcolor"  : TEXT_DIM,
    "xtick.color"      : TEXT_DIM,
    "ytick.color"      : TEXT_DIM,
    "axes.edgecolor"   : GRID_LINE,
    "axes.facecolor"   : BG_PANEL,
    "figure.facecolor" : BG_DARK,
    "axes.grid"        : True,
    "grid.color"       : GRID_LINE,
    "grid.linewidth"   : 0.6,
})


# ══════════════════════════════════════════════════════════════════
# UTILITY – Frame → Timecode
# ══════════════════════════════════════════════════════════════════
def frame_to_timecode(frame: int, fps: float = FPS) -> str:
    """Convert a frame number to a MM:SS.mmm timecode string."""
    return frames_to_timecode(frame, fps=fps, fmt="ms")


# ══════════════════════════════════════════════════════════════════
# 1.  LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════
def load_annotations(data_dir: str, match_id: str) -> dict:
    # match_id here includes the "DFL-MAT-" prefix (see MATCH_ID above);
    # discover_match_annotation_files expects the bare match id, so strip it.
    bare_match_id = match_id.replace("DFL-MAT-", "")
    file_map = discover_match_annotation_files(data_dir, bare_match_id)

    if not file_map:
        raise FileNotFoundError(
            f"No file found:\n  {os.path.join(data_dir, f'{match_id}_*.csv')}\n"
            "→ Check DATA_DIR in the CONFIGURATION section."
        )

    dfs = {}
    for key, path in sorted(file_map.items()):
        df  = pd.read_csv(path)
        df["player_jid"] = (
            pd.to_numeric(df["player_jid"], errors="coerce")
            .fillna(-1).astype(int)
        )
        df["half"] = df["half"].astype(str).str.strip()
        # keep the player name as-is (already in the CSV)
        if "player" not in df.columns:
            df["player"] = df["player_jid"].astype(str)
        dfs[key] = df
        print(f"    {key:32s}  {len(df):3d} segments")

    print(f"\n  {len(dfs)} file(s) loaded.\n")
    return dfs


# ══════════════════════════════════════════════════════════════════
# 2.  ALGORITHME DE MATCHING
# ══════════════════════════════════════════════════════════════════
def _temporal_iou(s_a, e_a, s_b, e_b) -> float:
    return _shared_temporal_iou(s_a, e_a, s_b, e_b)


def match_segments(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   tolerance: int = 0) -> list:
    matched = []
    used_b  = set()

    for ia, row_a in df_a.iterrows():
        candidates = df_b[
            (df_b["half"]       == row_a["half"]) &
            (df_b["player_jid"] == row_a["player_jid"])
        ]
        best         = None
        best_overlap = -np.inf

        for ib, row_b in candidates.iterrows():
            if ib in used_b:
                continue
            # directionnel : le start de B doit être dans le segment A
            b_in_a = (row_a["start_frame"] - tolerance <= row_b["start_frame"]
                      <= row_a["end_frame"] + tolerance)
            if b_in_a:
                overlap = (min(row_a["end_frame"], row_b["end_frame"])
                           - max(row_a["start_frame"], row_b["start_frame"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (ia, ib, row_a, row_b)

        if best is not None:
            ia, ib, row_a, row_b = best
            matched.append({
                "ia"     : ia,
                "ib"     : ib,
                "iou"    : _temporal_iou(row_a["start_frame"], row_a["end_frame"],
                                         row_b["start_frame"], row_b["end_frame"]),
                "d_start": abs(row_a["start_frame"] - row_b["start_frame"]),
                "d_end"  : abs(row_a["end_frame"]   - row_b["end_frame"]),
            })
            used_b.add(ib)
    return matched


# ══════════════════════════════════════════════════════════════════
# 3.  PAIRWISE STATISTICS
# ══════════════════════════════════════════════════════════════════
def compute_pair_stats(dfs: dict, tolerance: int = 0) -> tuple:
    annotators_by_half = {}
    for key in dfs:
        half, ann = key.split("_", 1)
        annotators_by_half.setdefault(half, []).append(ann)

    records_stats    = []
    records_isolated = []
    all_match_details = []   # new: all matched pairs with IoU

    for half, annotators in annotators_by_half.items():
        matched_indices = {ann: set() for ann in annotators}

        for ann_a, ann_b in itertools.combinations(sorted(annotators), 2):
            df_a = dfs[f"{half}_{ann_a}"]
            df_b = dfs[f"{half}_{ann_b}"]

            matched_ab = match_segments(df_a, df_b, tolerance)  # A → B
            matched_ba = match_segments(df_b, df_a, tolerance)  # B → A

            for m in matched_ab:
                matched_indices[ann_a].add(m["ia"])
                matched_indices[ann_b].add(m["ib"])
                all_match_details.append({
                    "half"    : half,
                    "pair"    : f"{ann_a}↔{ann_b}",
                    "iou"     : m["iou"],
                    "d_start" : m["d_start"],
                    "d_end"   : m["d_end"],
                    "player"  : df_a.loc[m["ia"], "player"] if "player" in df_a.columns else str(df_a.loc[m["ia"], "player_jid"]),
                })

            # B → A : on ajoute les indices non encore couverts
            for m in matched_ba:
                matched_indices[ann_b].add(m["ia"])
                matched_indices[ann_a].add(m["ib"])
                all_match_details.append({
                    "half"    : half,
                    "pair"    : f"{ann_a}↔{ann_b}",
                    "iou"     : m["iou"],
                    "d_start" : m["d_start"],
                    "d_end"   : m["d_end"],
                    "player"  : df_b.loc[m["ia"], "player"] if "player" in df_b.columns else str(df_b.loc[m["ia"], "player_jid"]),
                })

            d_start = [m["d_start"] for m in matched_ab]
            d_end   = [m["d_end"]   for m in matched_ab]
            ious    = [m["iou"]     for m in matched_ab]

            # IoU sur tous les segments (non-matchés = 0)
            n_unmatched_ab = len(df_a) - len(matched_ab)
            ious_all = ious + [0.0] * n_unmatched_ab

            records_stats.append({
                "half"              : half,
                "annotator_a"       : ann_a,
                "annotator_b"       : ann_b,
                "n_a"               : len(df_a),
                "n_b"               : len(df_b),
                "n_matched_ab"      : len(matched_ab),
                "n_matched_ba"      : len(matched_ba),
                "recall_a_%"        : round(100 * len(matched_ab) / len(df_a), 1) if len(df_a) else float("nan"),
                "recall_b_%"        : round(100 * len(matched_ba) / len(df_b), 1) if len(df_b) else float("nan"),
                "mean_Δstart"       : round(np.mean(d_start), 1) if d_start else float("nan"),
                "std_Δstart"        : round(np.std(d_start),  1) if d_start else float("nan"),
                "mean_Δend"         : round(np.mean(d_end),   1) if d_end   else float("nan"),
                "std_Δend"          : round(np.std(d_end),    1) if d_end   else float("nan"),
                "mean_IoU_matched"  : round(np.mean(ious),     3) if ious     else float("nan"),
                "mean_IoU_all"      : round(np.mean(ious_all), 3) if ious_all else float("nan"),
            })

        for ann in annotators:
            df = dfs[f"{half}_{ann}"]
            for idx, row in df.iterrows():
                if idx not in matched_indices[ann]:
                    records_isolated.append({
                        "half"           : half,
                        "annotator"      : ann,
                        "segment_id"     : row.get("segment_id", idx),
                        "player"         : row.get("player", str(row["player_jid"])),
                        "player_jid"     : row["player_jid"],
                        "start_frame"    : row["start_frame"],
                        "end_frame"      : row["end_frame"],
                        "start_timecode" : frame_to_timecode(int(row["start_frame"])),
                        "end_timecode"   : frame_to_timecode(int(row["end_frame"])),
                        "duration_frames": int(row["end_frame"]) - int(row["start_frame"]),
                        "duration_tc"    : frame_to_timecode(
                                               int(row["end_frame"]) - int(row["start_frame"])
                                           ),
                    })

    stats_df    = pd.DataFrame(records_stats)
    isolated_df = pd.DataFrame(records_isolated)
    details_df  = pd.DataFrame(all_match_details)
    return stats_df, isolated_df, details_df


# ══════════════════════════════════════════════════════════════════
# 4.  COLORS – Union-Find
# ══════════════════════════════════════════════════════════════════
def build_color_map(dfs: dict, tolerance: int = 0) -> tuple:
    annotators_by_half = {}
    for key in dfs:
        half, ann = key.split("_", 1)
        annotators_by_half.setdefault(half, []).append(ann)

    color_map           = {}
    match_pairs_by_half = {}
    color_counter       = 0

    for half, annotators in annotators_by_half.items():
        parent = {}

        def find(x):
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(x, x) != root:
                parent[x], x = root, parent[x]
            return root

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        pairs_this_half = []
        for ann_a, ann_b in itertools.combinations(sorted(annotators), 2):
            df_a = dfs[f"{half}_{ann_a}"]
            df_b = dfs[f"{half}_{ann_b}"]

            pairs_set = set()  # évite les doublons si un match apparaît dans les deux sens

            for m in match_segments(df_a, df_b, tolerance):   # A → B
                ia, ib = m["ia"], m["ib"]
                union((half, ann_a, ia), (half, ann_b, ib))
                key = (ann_a, ia, ann_b, ib)
                if key not in pairs_set:
                    pairs_set.add(key)
                    pairs_this_half.append(key)

            for m in match_segments(df_b, df_a, tolerance):   # B → A (ajout)
                ib, ia = m["ia"], m["ib"]   # df_b est le premier arg donc ia/ib inversés
                union((half, ann_a, ia), (half, ann_b, ib))
                key = (ann_a, ia, ann_b, ib)
                if key not in pairs_set:
                    pairs_set.add(key)
                    pairs_this_half.append(key)

        match_pairs_by_half[half] = pairs_this_half

        groups   = {}
        all_keys = [(half, ann, i)
                    for ann in annotators
                    for i in dfs[f"{half}_{ann}"].index]
        for k in all_keys:
            root = find(k)
            groups.setdefault(root, []).append(k)

        for root, members in groups.items():
            if len(members) > 1:
                for m in members:
                    color_map[m] = color_counter
                color_counter += 1
            else:
                color_map[members[0]] = None

    return color_map, match_pairs_by_half, color_counter


# ══════════════════════════════════════════════════════════════════
# 5.  PALETTE UTILITY
# ══════════════════════════════════════════════════════════════════
def _make_palette(n: int) -> list:
    if n <= len(PRESENTATION_PALETTE):
        return PRESENTATION_PALETTE[:max(n, 1)]
    import matplotlib as _mpl
    if hasattr(_mpl, "colormaps"):
        base = _mpl.colormaps["tab20"]
    else:
        base = _mcm.get_cmap("tab20")
    return [base(i / max(n, 1)) for i in range(n)]


# ══════════════════════════════════════════════════════════════════
# 6.  NEW — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════
def plot_summary_dashboard(stats_df: pd.DataFrame, isolated_df: pd.DataFrame,
                            output_dir: str = "."):
    """
    Main presentation figure:
    2×3 grid of bars for recall_a%, recall_b%, n_matched, mean_IoU,
    mean_Δstart, mean_Δend — one color per pair, one group per half.
    """
    if stats_df.empty:
        print("    No data for the dashboard.")
        return

    halves = stats_df["half"].unique()
    pairs  = (stats_df["annotator_a"] + "↔" + stats_df["annotator_b"]).unique()
    n_pairs = len(pairs)
    pair_colors = {p: PRESENTATION_PALETTE[i % len(PRESENTATION_PALETTE)]
                   for i, p in enumerate(pairs)}

    metrics = [
        ("recall_a_%",   "Recall A  (%)",          "%",   100),
        ("recall_b_%",      "Recall B  (%)",            "%",   100),
        ("n_matched_ab",    "Matched segments A→B",     "",    None),
        ("n_matched_ba",    "Matched segments B→A",     "",    None),
        ("mean_IoU_matched","Mean IoU (matched only)",  "",    1.0),
        ("mean_IoU_all",    "Mean IoU (all segments)",  "",    1.0),
    ]

    fig = plt.figure(figsize=(20, 14), facecolor=BG_DARK)
    fig.suptitle(
        f"Inter-Annotator Analysis  ·  {MATCH_ID}\n"
        f"Tolerance = {TOLERANCE} frames",
        fontsize=18, fontweight="bold", color=TEXT_MAIN,
        y=0.97,
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                  left=0.07, right=0.97, top=0.91, bottom=0.07)

    for idx, (col, title, unit, vmax) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor(BG_PANEL)

        x_positions = []
        bar_colors  = []
        bar_values  = []
        bar_labels  = []
        tick_labels = []

        x = 0
        for half in sorted(halves):
            sub = stats_df[stats_df["half"] == half].copy()
            sub["pair"] = sub["annotator_a"] + "↔" + sub["annotator_b"]
            for _, row in sub.iterrows():
                val = row[col]
                if pd.isna(val):
                    val = 0
                x_positions.append(x)
                bar_colors.append(pair_colors[row["pair"]])
                bar_values.append(val)
                bar_labels.append(val)
                lbl = f"{row['pair']}\n{half[:4]}"
                tick_labels.append(lbl)
                x += 1
            x += 0.4   # espace entre halves

        bars = ax.bar(x_positions, bar_values, color=bar_colors,
                      width=0.72, zorder=3, edgecolor=BG_DARK, linewidth=0.8)

        # values above the bars
        for bar, val in zip(bars, bar_labels):
            fmt = f"{val:.1f}{unit}" if unit in ("%", "f") else f"{val:.3f}" if unit == "" and vmax == 1.0 else f"{int(val)}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (vmax or max(bar_values, default=1)) * 0.015,
                    fmt, ha="center", va="bottom",
                    fontsize=8.5, color=TEXT_MAIN, fontweight="bold")

        if vmax:
            ax.set_ylim(0, vmax * 1.18)
        else:
            ax.set_ylim(0, max(bar_values, default=1) * 1.22 if bar_values else 1)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, fontsize=7.5, color=TEXT_DIM)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=TEXT_MAIN, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

        # reference line at 100% or 1.0
        if vmax in (100, 1.0):
            ax.axhline(vmax, color=TEXT_DIM, linewidth=0.7,
                       linestyle="--", alpha=0.5, zorder=1)

    # global legend
    legend_handles = [
        mpatches.Patch(color=pair_colors[p], label=p) for p in pairs
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(pairs), fontsize=10,
               framealpha=0.15, edgecolor=GRID_LINE,
               bbox_to_anchor=(0.5, 0.01))

    out_path = os.path.join(output_dir, "dashboard_metrics.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"   Dashboard saved       : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 7.  NEW — SEGMENT HEATMAP BY PLAYER
# ══════════════════════════════════════════════════════════════════
def plot_player_heatmap(dfs: dict, output_dir: str = "."):
    """
    Heatmap: rows = players (player_jid), columns = annotator×half.
    Value = number of annotated segments.
    Allows quick identification of players where annotators disagree.
    """
    records = []
    for key, df in dfs.items():
        half, ann = key.split("_", 1)
        for _, row in df.iterrows():
            records.append({"half": half, "annotator": ann,
                            "player": row.get("player", str(row["player_jid"])),
                            "player_jid": row["player_jid"]})
    if not records:
        return

    big_df = pd.DataFrame(records)
    big_df["col"] = big_df["half"] + "\n" + big_df["annotator"].str.capitalize()
    pivot = big_df.groupby(["player", "col"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()

    n_rows, n_cols = pivot.shape
    fig_h = max(4, n_rows * 0.55 + 2)
    fig_w = max(6, n_cols * 1.4 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pres", [BG_PANEL, "#1a4a6e", ACCENT_BLUE, ACCENT_TEAL], N=256
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0)

    # cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            txt_color = TEXT_MAIN if val > pivot.values.max() * 0.4 else TEXT_DIM
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, fontsize=9, color=TEXT_MAIN)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(pivot.index),
                       fontsize=9, color=TEXT_MAIN)

    ax.set_title("Annotated segments per player and annotator",
                 fontsize=14, fontweight="bold", color=TEXT_MAIN, pad=14)
    ax.set_xlabel("Annotator · Half", fontsize=10, color=TEXT_DIM)
    ax.set_ylabel("Player", fontsize=10, color=TEXT_DIM)

    # half group separators
    halves_order = []
    for c in pivot.columns:
        h = c.split("\n")[1] if "\n" in c else ""
        if h not in halves_order:
            halves_order.append(h)
    half_counts = {}
    for c in pivot.columns:
        h = c.split("\n")[1] if "\n" in c else ""
        half_counts[h] = half_counts.get(h, 0) + 1
    x_sep = 0
    for h in halves_order[:-1]:
        x_sep += half_counts[h]
        ax.axvline(x_sep - 0.5, color=ACCENT_GOLD, linewidth=1.5,
                   linestyle="--", alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Segment count", color=TEXT_DIM, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "player_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"   Heatmap saved         : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 8.  NEW — STYLED TABLE
# ══════════════════════════════════════════════════════════════════
def plot_stats_table_figure(stats_df: pd.DataFrame, output_dir: str = "."):
    """Generates a formatted matplotlib table, ready to be inserted into a slide."""
    if stats_df.empty:
        return

    display_cols = ["half", "annotator_a", "annotator_b",
                    "n_a", "n_b", "n_matched_ab", "n_matched_ba",
                    "recall_a_%", "recall_b_%",
                    "mean_IoU_matched", "mean_IoU_all", "mean_Δstart", "mean_Δend"]
    col_labels = ["Half", "Ann. A", "Ann. B",
                  "N(A)", "N(B)", "Matched A→B", "Matched B→A",
                  "Recall A%", "Recall B%",
                  "IoU matched", "IoU all", "Mean Δstart", "Mean Δend"]

    df_disp = stats_df[display_cols].copy()
    n_rows = len(df_disp)
    n_cols = len(display_cols)

    fig_h = 1.4 + n_rows * 0.55
    fig_w = n_cols * 1.35

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    cell_data = [df_disp.iloc[r].tolist() for r in range(n_rows)]
    # format numbers
    for r in range(n_rows):
        for c in range(n_cols):
            v = cell_data[r][c]
            if isinstance(v, float):
                cell_data[r][c] = f"{v:.1f}" if display_cols[c] in ("recall_a_%", "recall_b_%", "mean_Δstart", "mean_Δend") else f"{v:.3f}"

    row_colors = []
    for r in range(n_rows):
        row_colors.append(
            [BG_PANEL if r % 2 == 0 else "#1c2d3e"] * n_cols
        )

    tbl = ax.table(
        cellText   = cell_data,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        cellColours= row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # style header
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(ACCENT_BLUE)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor(BG_DARK)

    # style data cells + conditional coloring for recall & IoU
    for r in range(n_rows):
        for c in range(n_cols):
            cell = tbl[r + 1, c]
            cell.set_text_props(color=TEXT_MAIN, fontsize=9.5)
            cell.set_edgecolor(BG_DARK)
            # conditional color: recall%
            col_name = display_cols[c]
            try:
                val = float(stats_df.iloc[r][col_name])
            except Exception:
                val = None
            if col_name in ("recall_a_%", "recall_b_%") and val is not None:
                intensity = val / 100
                r_col = mcolors.to_rgba(ACCENT_TEAL, alpha=0.1 + 0.5 * intensity)
                cell.set_facecolor(r_col)
            elif col_name in ("mean_IoU_matched", "mean_IoU_all") and val is not None:
                intensity = min(val, 1.0)
                r_col = mcolors.to_rgba(ACCENT_CORAL, alpha=0.1 + 0.5 * intensity)
                cell.set_facecolor(r_col)

    ax.set_title("Inter-annotator metrics – summary",
                 fontsize=14, fontweight="bold",
                 color=TEXT_MAIN, pad=16, loc="left", x=0.0)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "stats_table.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"   Table saved           : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 9.  NEW — IoU DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
def plot_iou_distribution(details_df: pd.DataFrame, output_dir: str = "."):
    """
    Violin + scatter plot of IoU by pair and by half.
    Shows the spread of match quality.
    """
    if details_df.empty:
        print("    No match details for IoU distribution.")
        return

    pairs  = sorted(details_df["pair"].unique())
    halves = sorted(details_df["half"].unique())
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, len(halves),
                              figsize=(7 * len(halves), 6),
                              facecolor=BG_DARK, sharey=True)
    if len(halves) == 1:
        axes = [axes]

    fig.suptitle("Temporal IoU distribution by annotator pair",
                 fontsize=15, fontweight="bold", color=TEXT_MAIN, y=1.02)

    for ax, half in zip(axes, halves):
        ax.set_facecolor(BG_PANEL)
        sub = details_df[details_df["half"] == half]
        data_by_pair = [sub[sub["pair"] == p]["iou"].values for p in pairs]
        positions = range(len(pairs))

        # violins
        filtered_indices = [i for i, d in enumerate(data_by_pair) if len(d) > 1]
        parts = ax.violinplot(
            [d for d in data_by_pair if len(d) > 1],
            positions=[i for i, d in enumerate(data_by_pair) if len(d) > 1],
            showmedians=True, showextrema=True,
        )
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PRESENTATION_PALETTE[filtered_indices[j] % len(PRESENTATION_PALETTE)])
            pc.set_alpha(0.55)
            pc.set_edgecolor(TEXT_MAIN)
        for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part_name in parts:
                parts[part_name].set_color(TEXT_MAIN)
                parts[part_name].set_linewidth(1.2)

        # individual point scatter
        for i, (pair, d) in enumerate(zip(pairs, data_by_pair)):
            jitter = np.random.uniform(-0.08, 0.08, size=len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       color=PRESENTATION_PALETTE[i % len(PRESENTATION_PALETTE)],
                       alpha=0.7, s=28, zorder=4, edgecolors=BG_DARK, linewidths=0.4)

        # reference line IoU = 0.5
        ax.axhline(0.5, color=ACCENT_GOLD, linewidth=1,
                   linestyle="--", alpha=0.7, label="IoU = 0.5")
        ax.axhline(1.0, color=ACCENT_TEAL, linewidth=0.7,
                   linestyle=":", alpha=0.5, label="IoU = 1.0")

        ax.set_xticks(list(positions))
        ax.set_xticklabels(pairs, fontsize=10, color=TEXT_MAIN)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Temporal IoU", fontsize=10, color=TEXT_DIM)
        ax.set_title(f"Half: {half}", fontsize=12,
                     fontweight="bold", color=TEXT_MAIN, pad=10)
        ax.legend(fontsize=8.5, framealpha=0.15,
                  edgecolor=GRID_LINE, labelcolor=TEXT_MAIN)
        ax.spines[["top", "right"]].set_visible(False)

        # annotated median
        for i, d in enumerate(data_by_pair):
            if len(d):
                med = np.median(d)
                ax.text(i, med + 0.04, f"{med:.2f}",
                        ha="center", va="bottom", fontsize=8,
                        color=TEXT_MAIN, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "iou_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"   IoU distribution      : {out_path}")



def _short_name(name: str, max_len: int = 10) -> str:
    """Abbreviates a player name: 'Kingsley Coman' → 'K. Coman'."""
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"[:max_len]
    return str(name)[:max_len]

# ══════════════════════════════════════════════════════════════════
# 9b.  NON-LINEAR SCALE — empty zone compression
# ══════════════════════════════════════════════════════════════════
def _build_nonlinear_scale(all_segments: list, gap_compression: float = 0.15,
                            pad_frames: int = 80):
    """
    Builds a real_frame → visual_position mapping that:
      • stretches zones where segments are present
      • compresses empty gaps between segments

    Parameters
    ----------
    all_segments    : list of (start_frame, end_frame)
    gap_compression : fraction of the original width kept for gaps
                      (0.15 = empty zones are reduced to 15% of their size)
    pad_frames      : margin around segments in real frames

    Returns
    -------
    to_vis(frame)   : function real_frame → visual coordinate
    from_vis(vis)   : inverse function (piecewise linear approximation)
    vis_min, vis_max: total visual extent
    tick_info       : list of (vis_pos, label_str) for the X axis
    """
    if not all_segments:
        def identity(x): return x
        return identity, identity, 0, 1000, [(0, "0"), (500, "500"), (1000, "1000")]

    # ── 1. Build "active" intervals (union of all segments + pad) ──────────────
    segs = sorted(all_segments, key=lambda s: s[0])
    merged = []
    for s, e in segs:
        s, e = s - pad_frames, e + pad_frames
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # global bounds
    raw_min = merged[0][0]
    raw_max = merged[-1][1]

    # ── 2. Build the list of "pieces" (active / gap / active …) ──────────────
    pieces = []   # each piece: (raw_start, raw_end, is_active)
    cursor = raw_min
    for s, e in merged:
        if cursor < s:
            pieces.append((cursor, s, False))   # gap
        pieces.append((s, e, True))             # active segment
        cursor = e
    if cursor < raw_max:
        pieces.append((cursor, raw_max, False))

    # ── 3. Compute visual widths ───────────────────────────────────────────────
    vis_widths = []
    for rs, re, active in pieces:
        raw_w = re - rs
        vis_w = raw_w if active else raw_w * gap_compression
        vis_widths.append(vis_w)

    total_vis = sum(vis_widths)

    # ── 4. Mapping functions ──────────────────────────────────────────────────
    # breakpoints : (raw_start, raw_end, vis_start, vis_end)
    breakpoints = []
    vis_cursor = 0.0
    for (rs, re, _), vw in zip(pieces, vis_widths):
        breakpoints.append((rs, re, vis_cursor, vis_cursor + vw))
        vis_cursor += vw

    def to_vis(frame):
        for rs, re, vs, ve in breakpoints:
            if rs <= frame <= re:
                t = (frame - rs) / max(re - rs, 1)
                return vs + t * (ve - vs)
        # out of bounds: linear extrapolation on the last/first piece
        if frame < breakpoints[0][0]:
            rs, re, vs, ve = breakpoints[0]
            t = (frame - rs) / max(re - rs, 1)
            return vs + t * (ve - vs)
        rs, re, vs, ve = breakpoints[-1]
        t = (frame - rs) / max(re - rs, 1)
        return vs + t * (ve - vs)

    def from_vis(v):
        for rs, re, vs, ve in breakpoints:
            if vs <= v <= ve:
                t = (v - vs) / max(ve - vs, 1e-9)
                return rs + t * (re - rs)
        return v

    vis_min = 0.0
    vis_max = total_vis

    # ── 5. Ticks: one value every ~5 active segments or at start/end ─────────
    tick_info = []
    seen = set()
    for s, e in merged:
        for frame in [s + pad_frames, e - pad_frames]:
            frame = int(frame)
            bucket = round(frame / 50) * 50
            if bucket not in seen:
                seen.add(bucket)
                tick_info.append((to_vis(bucket), str(bucket)))
    # always include min and max
    tick_info.insert(0, (to_vis(raw_min + pad_frames),
                         str(int(raw_min + pad_frames))))
    tick_info.append((to_vis(raw_max - pad_frames),
                      str(int(raw_max - pad_frames))))

    return to_vis, from_vis, vis_min, vis_max, tick_info


# ══════════════════════════════════════════════════════════════════
# 10.  SANKEY RIBBON OUTPUT  (improved version)
# ══════════════════════════════════════════════════════════════════
def plot_ribbons(dfs: dict, tolerance: int = 0, output_dir: str = "."):
    color_map, match_pairs_by_half, n_colors = build_color_map(dfs, tolerance)
    palette = _make_palette(n_colors)

    annotators_by_half = {}
    for key in dfs:
        half, ann = key.split("_", 1)
        annotators_by_half.setdefault(half, []).append(ann)

    for half, annotators in annotators_by_half.items():
        annotators_sorted = sorted(annotators)
        n_ann = len(annotators_sorted)
        ann_y = {ann: i for i, ann in enumerate(annotators_sorted)}

        BAR_H = 0.38
        fig_h = 2.5 + n_ann * 1.8
        fig, ax = plt.subplots(figsize=(22, fig_h), facecolor=BG_DARK)
        ax.set_facecolor(BG_PANEL)
        fig.patch.set_facecolor(BG_DARK)

        # ── Non-linear scale ──────────────────────────────────────
        all_segs = [
            (row["start_frame"], row["end_frame"])
            for ann in annotators_sorted
            for _, row in dfs[f"{half}_{ann}"].iterrows()
        ]
        to_vis, from_vis, vis_min, vis_max, tick_info = _build_nonlinear_scale(
            all_segs, gap_compression=0.12, pad_frames=60
        )
        ax.set_xlim(vis_min, vis_max)
        ax.set_ylim(-0.8, n_ann - 0.2)

        # vertical grid on each active segment (start and end)
        for vis_pos, _ in tick_info:
            ax.axvline(vis_pos, color=GRID_LINE, linewidth=0.5, zorder=0)

        # ribbons — transformed coordinates
        from matplotlib.path import Path as MPath
        drawn = set()
        for ann_a, ia, ann_b, ib in match_pairs_by_half.get(half, []):
            conn_key = (min(ann_a, ann_b), min(ia, ib), max(ann_a, ann_b), max(ia, ib))
            if conn_key in drawn:
                continue
            drawn.add(conn_key)

            df_a = dfs[f"{half}_{ann_a}"]
            df_b = dfs[f"{half}_{ann_b}"]
            cid  = color_map.get((half, ann_a, ia))
            col  = palette[cid] if cid is not None else "#334455"

            ya = ann_y[ann_a]
            yb = ann_y[ann_b]
            if ya > yb:
                ya, yb = yb, ya
                df_a, df_b = df_b, df_a
                ia, ib = ib, ia

            s_a = to_vis(df_a.loc[ia, "start_frame"])
            e_a = to_vis(df_a.loc[ia, "end_frame"])
            s_b = to_vis(df_b.loc[ib, "start_frame"])
            e_b = to_vis(df_b.loc[ib, "end_frame"])

            y_top_bot = ya + BAR_H
            y_bot_top = yb - BAR_H
            y_ctrl    = (y_top_bot + y_bot_top) / 2

            verts = [
                (s_a, y_top_bot), (s_a, y_ctrl), (s_b, y_ctrl), (s_b, y_bot_top),
                (e_b, y_bot_top), (e_b, y_ctrl), (e_a, y_ctrl), (e_a, y_top_bot),
                (s_a, y_top_bot),
            ]
            codes = [
                MPath.MOVETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.LINETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.CLOSEPOLY,
            ]
            ax.add_patch(mpatches.PathPatch(
                MPath(verts, codes),
                facecolor=col, alpha=0.28,
                edgecolor=col, linewidth=0.8, zorder=1,
            ))

        # bars — transformed coordinates
        for ann in annotators_sorted:
            y  = ann_y[ann]
            df = dfs[f"{half}_{ann}"]
            for idx, row in df.iterrows():
                cid  = color_map.get((half, ann, idx))
                col  = palette[cid] if cid is not None else "#334455"
                ecol = BG_DARK      if cid is not None else "#445566"
                vs = to_vis(row["start_frame"])
                ve = to_vis(row["end_frame"])
                vis_w = max(ve - vs, 1)
                ax.barh(y, width=vis_w, left=vs, height=BAR_H * 2,
                        color=col, edgecolor=ecol, linewidth=0.8, zorder=3)
                mid_vis = (vs + ve) / 2
                ax.text(mid_vis, y, _short_name(row.get('player', str(row['player_jid']))),
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=TEXT_MAIN if cid is not None else TEXT_DIM,
                        zorder=5)

        # X axis: ticks in visual coordinates, labels in real frames
        t_pos  = [t[0] for t in tick_info]
        t_lbls = [t[1] for t in tick_info]
        ax.set_xticks(t_pos)
        ax.set_xticklabels(t_lbls, fontsize=7.5, color=TEXT_DIM, rotation=35, ha="right")

        # "compressed scale" annotation
        ax.text(0.99, -0.08,
                " Non-linear axis — empty zones compressed",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=TEXT_DIM, style="italic")

        ax.set_yticks(list(ann_y.values()))
        ax.set_yticklabels([a.capitalize() for a in annotators_sorted],
                           fontsize=12, color=TEXT_MAIN)
        ax.invert_yaxis()
        ax.set_xlabel("Frame (non-linear scale)", fontsize=11, color=TEXT_DIM)
        ax.set_title(
            f"Sankey Ribbons  ·  {half}    |    tolerance = {tolerance} frames",
            fontsize=14, fontweight="bold", color=TEXT_MAIN, pad=14,
        )
        ax.spines[["top", "right"]].set_visible(False)

        handles = [
            mpatches.Patch(color=PRESENTATION_PALETTE[0], alpha=0.85,
                           label="Matched segment (color = group)"),
            mpatches.Patch(color="#334455", label="Isolated segment"),
            mpatches.Patch(color=PRESENTATION_PALETTE[0], alpha=0.28,
                           label="Ribbon = overlap zone"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  framealpha=0.3, edgecolor=GRID_LINE,
                  labelcolor=TEXT_MAIN)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"ribbon_{half}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        plt.close(fig)
        print(f"   Ribbon saved         : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 11.  SWIM-LANE ARCS OUTPUT  (improved version)
# ══════════════════════════════════════════════════════════════════
def plot_arcs(dfs: dict, tolerance: int = 0, output_dir: str = "."):
    from matplotlib.path import Path

    color_map, match_pairs_by_half, n_colors = build_color_map(dfs, tolerance)
    palette = _make_palette(n_colors)

    annotators_by_half = {}
    for key in dfs:
        half, ann = key.split("_", 1)
        annotators_by_half.setdefault(half, []).append(ann)

    for half, annotators in annotators_by_half.items():
        annotators_sorted = sorted(annotators)
        n_ann = len(annotators_sorted)

        LANE_H  = 2.2
        BAR_CY  = 0.5
        BAR_H   = 0.30
        ARC_MAX = 0.90

        fig_h = 2.0 + n_ann * LANE_H
        fig, ax = plt.subplots(figsize=(22, fig_h), facecolor=BG_DARK)
        ax.set_facecolor(BG_PANEL)
        fig.patch.set_facecolor(BG_DARK)

        lane_base = {ann: i * LANE_H for i, ann in enumerate(annotators_sorted)}
        ann_bar_y = {ann: lane_base[ann] + BAR_CY for ann in annotators_sorted}

        # ── Non-linear scale ──────────────────────────────────────
        all_segs = [
            (row["start_frame"], row["end_frame"])
            for ann in annotators_sorted
            for _, row in dfs[f"{half}_{ann}"].iterrows()
        ]
        to_vis, from_vis, vis_min, vis_max, tick_info = _build_nonlinear_scale(
            all_segs, gap_compression=0.12, pad_frames=60
        )
        vis_span = max(vis_max - vis_min, 1)
        ax.set_xlim(vis_min, vis_max)
        ax.set_ylim(-0.3, n_ann * LANE_H)

        for ann in annotators_sorted:
            lb = lane_base[ann]
            ax.axhspan(lb, lb + LANE_H,
                       alpha=0.06 if annotators_sorted.index(ann) % 2 else 0.03,
                       color=ACCENT_BLUE, zorder=0)
            ax.axhline(lb, color=GRID_LINE, linewidth=0.8, zorder=1)
            ax.text(vis_min + vis_span * 0.002,
                    lb + BAR_CY, ann.capitalize(),
                    va="center", ha="left", fontsize=11,
                    fontweight="bold", color=TEXT_MAIN, zorder=6)

        # grid on active breakpoints
        for vis_pos, _ in tick_info:
            ax.axvline(vis_pos, color=GRID_LINE, linewidth=0.5, zorder=0)

        # bars — transformed coordinates
        for ann in annotators_sorted:
            y_bar = ann_bar_y[ann]
            df    = dfs[f"{half}_{ann}"]
            for idx, row in df.iterrows():
                cid  = color_map.get((half, ann, idx))
                col  = palette[cid] if cid is not None else "#334455"
                ecol = BG_DARK      if cid is not None else "#445566"
                vs = to_vis(row["start_frame"])
                ve = to_vis(row["end_frame"])
                vis_w = max(ve - vs, 1)
                ax.barh(y_bar, width=vis_w, left=vs, height=BAR_H,
                        color=col, edgecolor=ecol, linewidth=0.8, zorder=3)
                mid_vis = (vs + ve) / 2
                ax.text(mid_vis, y_bar,
                        _short_name(row.get('player', str(row['player_jid']))),
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=TEXT_MAIN if cid is not None else TEXT_DIM,
                        zorder=5)
                if cid is None:
                    ax.plot(vs, y_bar - BAR_H * 0.7,
                            marker="v", markersize=5,
                            color=ACCENT_CORAL, zorder=4, clip_on=False)

        # arcs — transformed coordinates
        drawn = set()
        for ann_a, ia, ann_b, ib in match_pairs_by_half.get(half, []):
            conn_key = (min(ann_a, ann_b), min(ia, ib), max(ann_a, ann_b), max(ia, ib))
            if conn_key in drawn:
                continue
            drawn.add(conn_key)

            df_a = dfs[f"{half}_{ann_a}"]
            df_b = dfs[f"{half}_{ann_b}"]
            cid  = color_map.get((half, ann_a, ia))
            col  = palette[cid] if cid is not None else ACCENT_BLUE

            ya = ann_bar_y[ann_a]
            yb = ann_bar_y[ann_b]
            # real frames for the displayed Δ, visual positions for drawing
            xa_raw = df_a.loc[ia, "start_frame"]
            xb_raw = df_b.loc[ib, "start_frame"]
            xa = to_vis(xa_raw)
            xb = to_vis(xb_raw)

            delta_raw = abs(xa_raw - xb_raw)
            vis_delta = abs(xa - xb)
            arc_up = min(0.15 + vis_delta / vis_span * ARC_MAX * 3, ARC_MAX)
            y_top  = min(ya, yb) - arc_up * LANE_H * 0.55

            verts = [(xa, ya), (xa, y_top), (xb, y_top), (xb, yb)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            ax.add_patch(mpatches.PathPatch(
                Path(verts, codes),
                facecolor="none", edgecolor=col,
                linewidth=1.6, alpha=0.8, zorder=2,
            ))
            ax.plot(xa, ya, "o", color=col, markersize=5, zorder=5)
            ax.plot(xb, yb, "o", color=col, markersize=5, zorder=5)
            if delta_raw > 0:
                ax.text((xa + xb) / 2, y_top - 0.04,
                        f"Δ{int(delta_raw)}f",
                        ha="center", va="bottom",
                        fontsize=7.5, color=col,
                        fontweight="bold", zorder=6)

        # X axis: ticks in visual coordinates, real frame labels
        t_pos  = [t[0] for t in tick_info]
        t_lbls = [t[1] for t in tick_info]
        ax.set_xticks(t_pos)
        ax.set_xticklabels(t_lbls, fontsize=7.5, color=TEXT_DIM, rotation=35, ha="right")

        # "compressed scale" annotation
        ax.text(0.99, -0.06,
                " Non-linear axis — empty zones compressed",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=TEXT_DIM, style="italic")

        ax.set_yticks([])
        ax.set_xlabel("Frame (non-linear scale)", fontsize=11, color=TEXT_DIM)
        ax.set_title(
            f"Swim-lane Arcs  ·  {half}    |    tolerance = {tolerance} frames",
            fontsize=14, fontweight="bold", color=TEXT_MAIN, pad=14,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)

        arc_line  = mlines.Line2D([], [], color=PRESENTATION_PALETTE[0],
                                   linewidth=1.6, alpha=0.8,
                                   label="Arc = start_frame connection (Δ shown)")
        iso_mark  = mlines.Line2D([], [], marker="v", color=ACCENT_CORAL,
                                   linestyle="None", markersize=6,
                                   label="Isolated segment (no match)")
        seg_patch = mpatches.Patch(color=PRESENTATION_PALETTE[0],
                                    alpha=0.85, label="Matched segment")
        gry_patch = mpatches.Patch(color="#334455", label="Isolated segment")
        ax.legend(handles=[seg_patch, gry_patch, arc_line, iso_mark],
                  loc="upper right", fontsize=9,
                  framealpha=0.3, edgecolor=GRID_LINE,
                  labelcolor=TEXT_MAIN)
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"arcs_{half}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        plt.close(fig)
        print(f"   Arcs saved           : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 12b.  CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════
def plot_confusion_matrices(stats_df: pd.DataFrame, output_dir: str = "."):
    """
    6 confusion matrices in a 3x2 grid:
      Rows    : n_matched / n_total  |  recall %
      Columns : Half 1  |  Half 2  |  Global (all halves)
    Cell [A, B] = direction A -> B.
    """
    if stats_df.empty:
        print("    No data for confusion matrices.")
        return

    annotators = sorted(
        set(stats_df["annotator_a"].tolist() + stats_df["annotator_b"].tolist())
    )
    n   = len(annotators)
    idx = {a: i for i, a in enumerate(annotators)}
    labels = [a.capitalize() for a in annotators]

    halves = sorted(stats_df["half"].unique())

    def _build_matrices(subset):
        """Return (mat_count, mat_total, mat_iou) from a stats_df subset."""
        agg = (
            subset
            .groupby(["annotator_a", "annotator_b"], as_index=False)
            .agg(
                n_matched_ab=("n_matched_ab", "sum"),
                n_matched_ba=("n_matched_ba", "sum"),
                n_a=("n_a", "sum"),
                n_b=("n_b", "sum"),
                mean_IoU=("mean_IoU_matched", "mean"),
            )
        )
        mc  = np.full((n, n), np.nan)
        mt  = np.full((n, n), np.nan)
        mio = np.full((n, n), np.nan)
        for _, row in agg.iterrows():
            ia = idx[row["annotator_a"]]
            ib = idx[row["annotator_b"]]
            mc[ia, ib]  = row["n_matched_ab"]
            mt[ia, ib]  = row["n_a"]
            mio[ia, ib] = round(row["mean_IoU"], 3) if not pd.isna(row["mean_IoU"]) else np.nan
            mc[ib, ia]  = row["n_matched_ba"]
            mt[ib, ia]  = row["n_b"]
            mio[ib, ia] = mio[ia, ib]   # IoU est symétrique
        return mc, mt, mio

    # build one set of matrices per half + global
    sections = [(h, stats_df[stats_df["half"] == h]) for h in halves]
    sections.append(("Global", stats_df))

    col_titles = [h for h, _ in sections]
    matrices   = [_build_matrices(sub) for _, sub in sections]

    # figure: 2 rows (n_matched, mean_IoU) × len(sections) cols
    n_cols = len(sections)
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(7 * n_cols, 12),
                             facecolor=BG_DARK)
    fig.suptitle(
        f"Confusion Matrices  ·  {MATCH_ID}\nCell [A, B] = direction A \u2192 B",
        fontsize=16, fontweight="bold", color=TEXT_MAIN, y=1.01,
    )

    def _draw_matrix(ax, values, cell_texts, title, cmap_name, vmin, vmax, fmt_cbar):
        masked = np.ma.masked_invalid(values)
        cmap   = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color=BG_DARK)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "\u2014", ha="center", va="center",
                            fontsize=13, color=TEXT_DIM)
                elif not np.isnan(values[i, j]):
                    brightness = (values[i, j] - vmin) / max(vmax - vmin, 1e-9)
                    color = TEXT_MAIN if brightness < 0.6 else BG_DARK
                    ax.text(j, i, cell_texts[i, j], ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color,
                            multialignment="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=10, color=TEXT_MAIN)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10, color=TEXT_MAIN)
        ax.set_xlabel("Annotateur B", fontsize=9, color=TEXT_DIM)
        ax.set_ylabel("Annotateur A", fontsize=9, color=TEXT_DIM)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_MAIN, pad=10)
        ax.set_facecolor(BG_DARK)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(fmt_cbar, color=TEXT_DIM, fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    # global vmax for n_matched (consistent color scale across all cols)
    all_counts = [mc for mc, _, _ in matrices]
    global_vmax_count = max(
        (float(np.nanmax(mc)) for mc in all_counts if not np.all(np.isnan(mc))),
        default=1.0
    )

    for col, (mc, mt, mio) in enumerate(matrices):
        half_label = col_titles[col]

        # ── Row 0 : n_matched / n_total ──────────────────────────────
        cell_count = np.full((n, n), "", dtype=object)
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(mc[i, j]):
                    cell_count[i, j] = f"{int(mc[i,j])} / {int(mt[i,j])}"
        _draw_matrix(axes[0, col], mc, cell_count,
                     f"{half_label}  —  n_matched / n_total",
                     "Blues", 0, global_vmax_count, "n_matched")

        # ── Row 1 : mean IoU ─────────────────────────────────────────
        cell_iou = np.full((n, n), "", dtype=object)
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(mio[i, j]):
                    cell_iou[i, j] = f"{mio[i,j]:.3f}"
        _draw_matrix(axes[1, col], mio, cell_iou,
                     f"{half_label}  —  Mean IoU",
                     "YlOrRd", 0, 1.0, "mean IoU")

    # column separators
    for col in range(1, n_cols):
        for row in range(2):
            axes[row, col].spines["left"].set_color(ACCENT_GOLD)
            axes[row, col].spines["left"].set_linewidth(1.5)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"   Confusion matrices    : {out_path}")


# ══════════════════════════════════════════════════════════════════
# 12.  ENRICHED CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════
def print_banner():
    print()
    print("╔" + "═" * 63 + "╗")
    print("║     Inter-Annotator Analysis · Runs-in-Behind" + " " * 14 + "║")
    print(f"║  Match     : {MATCH_ID}" + " " * (63 - 13 - len(MATCH_ID) - 1) + "║")
    print(f"║  Tolerance : {TOLERANCE} frames" + " " * (63 - 14 - len(str(TOLERANCE)) - 8) + "║")
    print("╚" + "═" * 63 + "╝")
    print()


def print_stats_rich(stats_df: pd.DataFrame, isolated_df: pd.DataFrame):
    print("┌" + "─" * 63 + "┐")
    print("│  PAIRWISE METRICS" + " " * 45 + "│")
    print("└" + "─" * 63 + "┘")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(stats_df.to_string(index=False))
    print()

    n_iso = len(isolated_df)
    print(f"┌─ Isolated segments: {n_iso} total " + "─" * max(0, 43 - len(str(n_iso))) + "┐")
    if isolated_df.empty:
        print("│  No isolated segment. " + " " * 38 + "│")
    else:
        iso_by_ann = isolated_df.groupby(["half", "annotator"]).size().reset_index(name="count")
        for _, row in iso_by_ann.iterrows():
            msg = f"  {row['annotator'].capitalize():12s}  {row['half']:12s}  →  {row['count']} isolated segment(s)"
            print(f"│{msg}" + " " * max(0, 63 - len(msg)) + "│")
    print("└" + "─" * 63 + "┘")
    print()

    # Global KPIs
    if not stats_df.empty:
        mean_recall      = stats_df[["recall_a_%", "recall_b_%"]].stack().mean()
        mean_iou_matched = stats_df["mean_IoU_matched"].mean()
        mean_iou_all     = stats_df["mean_IoU_all"].mean()
        mean_dstart      = stats_df["mean_Δstart"].mean()
        print("┌─ Global indicators " + "─" * 43 + "┐")
        print(f"│  Mean recall all pairs       : {mean_recall:6.1f} %" + " " * 26 + "│")
        print(f"│  Mean IoU (matched only)     : {mean_iou_matched:6.3f}  " + " " * 26 + "│")
        print(f"│  Mean IoU (all segments)     : {mean_iou_all:6.3f}  " + " " * 26 + "│")
        print(f"│  Mean Δstart                 : {mean_dstart:6.1f} frames" + " " * 22 + "│")
        print("└" + "─" * 63 + "┘")
    print()


# ══════════════════════════════════════════════════════════════════
# 13.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def main():
    print_banner()

    # Loading
    dfs = load_annotations(DATA_DIR, MATCH_ID)

    # Statistics
    stats_df, isolated_df, details_df = compute_pair_stats(dfs, TOLERANCE)

    # Console
    print_stats_rich(stats_df, isolated_df)

    # CSV saves
    stats_path    = os.path.join(OUTPUT_DIR, "inter_annotator_stats.csv")
    isolated_path = os.path.join(OUTPUT_DIR, "isolated_segments.csv")
    details_path  = os.path.join(OUTPUT_DIR, "match_details.csv")
    stats_df.to_csv(stats_path,    index=False)
    isolated_df.to_csv(isolated_path, index=False)
    details_df.to_csv(details_path,   index=False)
    print(f"   Stats CSV        → {stats_path}")
    print(f"   Isolated CSV     → {isolated_path}")
    print(f"   Match details    → {details_path}")
    print()

    # Visualisations
    print("Generating visualisations …")
    print()

    print("  [1/7] Metrics dashboard …")
    plot_summary_dashboard(stats_df, isolated_df, OUTPUT_DIR)

    print("  [2/7] Player heatmap …")
    plot_player_heatmap(dfs, OUTPUT_DIR)

    print("  [3/7] Styled table …")
    plot_stats_table_figure(stats_df, OUTPUT_DIR)

    print("  [4/7] IoU distribution …")
    plot_iou_distribution(details_df, OUTPUT_DIR)

    print("  [5/7] Sankey ribbons …")
    plot_ribbons(dfs, TOLERANCE, OUTPUT_DIR)

    print("  [6/7] Swim-lane Arcs …")
    plot_arcs(dfs, TOLERANCE, OUTPUT_DIR)

    print("  [7/7] Confusion matrices …")
    plot_confusion_matrices(stats_df, OUTPUT_DIR)

    print()
    print("╔" + "═" * 63 + "╗")
    print("║     Analysis complete!  All outputs are in:" + " " * 18 + "║")
    print(f"║  {OUTPUT_DIR[:60]}" + " " * max(0, 61 - len(OUTPUT_DIR[:60])) + "║")
    print("╚" + "═" * 63 + "╝")
    print()

    return stats_df, isolated_df, details_df


if __name__ == "__main__":
    stats, isolated, details = main()