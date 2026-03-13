import json
import os
import pandas as pd
from Discretisation_Ilana_2nd_try import process_match

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR  = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223'
JSON_PATH = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\direction_idsse_video.json'

# Timezones handled automatically: matchinformation is always UTC (+00:00),
# events are always local German time (CEST +02:00 in summer, CET +01:00 in winter).
# pd.Timestamp.tz_convert('UTC') normalises both sides before any arithmetic.
ls_match_ids = ['J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']

# ── Load direction JSON once (shared across all matches) ──────────────────────

with open(JSON_PATH, "r") as f:
    dict_direction = json.load(f)

# ── Initialise result containers ──────────────────────────────────────────────

all_movements    = []
all_distances    = []
all_trajectories = {}

output_dir = r'C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_loop_Ilana'
os.makedirs(output_dir, exist_ok=True)

# ── Loop over matches ─────────────────────────────────────────────────────────

for match_id in ls_match_ids:
    try:
        df_mov, df_dist, dict_traj = process_match(
            match_id       = match_id,
            DATA_DIR       = DATA_DIR,
            dict_direction = dict_direction,
        )
        all_movements.append(df_mov)
        all_distances.append(df_dist)
        all_trajectories[match_id] = dict_traj

        # ── Per-match filter and save ─────────────────────────────────────────
        df_runs = df_mov[
            (df_mov['possession'] == df_mov['location']) &
            (df_mov['speed_category'].isin(['jogging', 'running', 'sprinting'])) &
            (df_mov['duration_s'] > 2) &
            ((df_mov['x_end'] - df_mov['x_start']) * df_mov['attack_sign'] > 3) &
            (df_mov['direction'] > 0.5) &
            (~df_mov['has_ball_touch'])
        ]
        df_runs.to_csv(
            os.path.join(output_dir, f'runs_behind_{match_id}.csv'), index=False
        )
        print(f"  → {len(df_runs)} runs-in-behind saved for {match_id}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {match_id} — {e}")
        traceback.print_exc()

# ── Concatenate results across all matches ────────────────────────────────────

df_all_movements = pd.concat(all_movements, ignore_index=True)
df_all_distances = pd.concat(all_distances)

print(f"\nTotal movements across all matches: {len(df_all_movements)}")
print(df_all_movements['speed_category'].value_counts())

# ── Save global files ─────────────────────────────────────────────────────────

df_all_movements.to_csv(os.path.join(output_dir, 'all_movements.csv'), index=False)
# df_all_distances.to_csv(os.path.join(output_dir, 'all_distances.csv'))

print(f"\nFiles saved in: {output_dir}")