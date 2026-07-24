"""
===============================================================
 Example usage of the Pitch Control Toolkit
===============================================================
Run this script to analyse a sequence of frames from a DFL match.
"""

from pitch_control_tool import DFLDataProvider, PitchControlAnalyzer
from common import config

# ── paths ──────────────────────────────────────────────────────
DATA_DIR       = config.DATA_DIR
OUTPUT_DIR     = config.PITCH_CONTROL_OUTPUT_DIR
DIRECTION_JSON = config.DIRECTION_JSON
MATCH_ID       = 'DFL-MAT-J03WMX'    # change to any available match id

# ── 1.  Instantiate the data provider ─────────────────────────
provider = DFLDataProvider(
    data_dir       = DATA_DIR,
    match_id       = MATCH_ID,
    direction_json = DIRECTION_JSON,
)

# ── 2.  Instantiate the analyser ──────────────────────────────
analyzer = PitchControlAnalyzer(
    provider   = provider,
    output_dir = OUTPUT_DIR,
    att_team   = 'home',
)

# ── 3.  Check available frame range ──────────────────────────
start, end = provider.frame_range()
print(f'Available frames: {start} → {end}')

# ── 4.  Run analysis on a frame window ────────────────────────
#
#   Standard options
#   ─────────────────
#   save_animation      → produce a .gif (or .mp4) of all frames
#   save_frame_images   → also save each frame as a .png
#   save_csv            → write metrics_detail.csv + metrics_summary.csv
#                         + frame_control.csv
#   animation_format    → 'gif' (no extra codec needed) or 'mp4' (needs ffmpeg)
#
#   Ball-speed radius freeze
#   ─────────────────────────
#   ball_speed_threshold → speed in km/h above which the distance-to-ball
#                          radius scaling is disabled (default 35 km/h).
#
#                          Why: normally each player's influence ellipse
#                          shrinks when the ball is far away. During a fast
#                          through-ball the ball may be 30+ m from players
#                          who are still running onto it, which would
#                          artificially collapse their ellipses and drop PC.
#                          When the ball travels above this threshold all
#                          players use MAX_RADIUS (10 m, see pitch_control_tool.py)
#                          regardless of their distance to the ball.
#
#                          The columns 'ball_speed_kmh' and 'radius_frozen'
#                          (True/False) are added to frame_control.csv and
#                          snapshot_summary.csv so you can audit every frame.
#
#   Snapshot / zone-control options
#   ─────────────────────────────────
#   save_snapshot_csv   → write snapshot_summary.csv with exactly 3 rows:
#                           • start frame  /  mid frame  /  end frame
#                         Each row contains:
#                           ┌─ pc_home_only / pc_away_only
#                           │    PC of each team in isolation (no opponent).
#                           ├─ pc_combined_home / pc_combined_away
#                           │    Standard combined PC as home/away fractions.
#                           ├─ ball_zone_<R>m_home / ball_zone_<R>m_away
#                           │    Mean combined PC in a circle of radius R
#                           │    around the ball.
#                           ├─ player_zone_<Name>_home / player_zone_<Name>_away
#                           │    Same metric around the target player.
#                           ├─ ball_speed_kmh   (km/h at that frame)
#                           └─ radius_frozen    (True if threshold was exceeded)
#
#   ball_control_radius → radius in metres for zone metrics (default 6 m)
#   target_player       → partial player name, case-insensitive.  None = skip.
#
df = analyzer.analyze(
    start_frame          = 21569,
    end_frame            = 21622,
    save_animation       = True,
    save_frame_images    = False,
    save_csv             = True,
    animation_format     = 'gif',
    # ── ball-speed radius freeze ─────────────────────────────
    ball_speed_threshold = 35.0,        # km/h; set to None or 0 to always freeze, or 9999 to never
    # ── snapshot / zone-control ──────────────────────────────
    save_snapshot_csv    = True,
    ball_control_radius  = 6.0,
    player_zone_radius   = 6.0,
    target_player        = 'Serge Gnabry',    # None to disable
)

# ── 5.  Optional: use the returned DataFrame directly ─────────
print(df.head())


# ── 6.  Further examples: a batch of away-team windows ────────
# (start_frame, end_frame, target_player, note)
AWAY_TEAM_EXAMPLES = [
    (13259, 13374, 'E. Martel', None),
    (19550, 19613, 'Kingsley Coman', None),
    (59688, 59767, 'N. Mazraoui', None),
    (71746, 71832, 'B. Schmitz', None),
    (13906, 13960, 'D. Ljubicic', 'known false positive of the pitch-control heuristic'),
    (25004, 25069, 'E. Martel', None),
    (42855, 42932, 'T. Müller', None),
    (52674, 52769, 'Serge Gnabry', None),
    (41853, 41969, 'Kingsley Coman', None),
]

for start_frame, end_frame, target_player, note in AWAY_TEAM_EXAMPLES:
    if note:
        print(f'[NOTE] {target_player} @ [{start_frame}, {end_frame}]: {note}')
    provider2 = DFLDataProvider(DATA_DIR, MATCH_ID, DIRECTION_JSON)
    analyzer2 = PitchControlAnalyzer(provider2, OUTPUT_DIR, att_team='away')
    df2 = analyzer2.analyze(start_frame, end_frame,
                            ball_speed_threshold=35.0, target_player=target_player)
