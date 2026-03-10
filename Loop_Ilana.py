import json
import os
import pandas as pd
from Discrete_efforts_Ilana import process_match

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR  = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223'
JSON_PATH = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\direction_idsse_video.json'

ls_match_ids = ['J03WMX']#, 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']

# ── Load direction JSON once (shared across all matches) ──────────────────────

with open(JSON_PATH, "r") as f:
    dict_direction = json.load(f)

# ── Initialise result containers ──────────────────────────────────────────────

all_movements    = []
all_distances    = []
all_trajectories = {}

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

    except FileNotFoundError as e:
        # Skip match if data files are missing, continue with the rest
        print(f"[SKIP] {match_id} — missing file: {e}")
    except Exception as e:
        # Log any other error without breaking the loop
        print(f"[ERROR] {match_id} — {e}")

# ── Concatenate results across all matches ────────────────────────────────────

df_all_movements = pd.concat(all_movements, ignore_index=True)
df_all_distances = pd.concat(all_distances)

print(f"\nTotal movements across all matches: {len(df_all_movements)}")
print(df_all_movements['speed_category'].value_counts())

# ── Example filter: runs in behind candidates ─────────────────────────────────

df_runs_behind = df_all_movements[
    (df_all_movements['possession'] == df_all_movements['location']) & #possession component
    (df_all_movements['speed_category'].isin(['jogging', 'running', 'sprinting'])) & #speed component
    (df_all_movements['duration_s'] > 2) & #duration component
    #(df_all_movements['x_start'] * df_all_movements['attack_sign'] > -30) & #spatial component (last third of the pitch) +  direction
    ((df_all_movements['x_end'] - df_all_movements['x_start']) * df_all_movements['attack_sign'] > 3) &  #movement length component (at least 3m forward)
    (df_all_movements['direction'] > 0.5)   # strong forward component
    #During a set piece, we make a 6-second jump.
]

print(f"\nRuns-in-behind candidates detected: {len(df_runs_behind)}")

# ── Save results to CSV ───────────────────────────────────────────────────────

output_dir = r'C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_loop_Ilana'
os.makedirs(output_dir, exist_ok=True)

df_all_movements.to_csv(os.path.join(output_dir, 'all_movements.csv'), index=False)
#df_all_distances.to_csv(os.path.join(output_dir, 'all_distances.csv'))
df_runs_behind.to_csv(os.path.join(output_dir, 'runs_behind.csv'), index=False)

print(f"\nFiles saved in: {output_dir}")