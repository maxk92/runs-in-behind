import json
import os
import pandas as pd
import numpy as np
from Discretisation_optimised2_1 import process_match

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR  = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\open_data2223'
JSON_PATH = r'C:\Users\arnau\Sciebo\SharedDrive_Arnaud_Franziska\data\direction_idsse_video.json'

ls_match_ids = ['J03WMX', 'J03WN1', 'J03WPY', 'J03WOH', 'J03WQQ', 'J03WOY', 'J03WR9']

# ── Load direction JSON once (shared across all matches) ──────────────────────

with open(JSON_PATH, "r") as f:
    dict_direction = json.load(f)

# ── Initialise result containers ──────────────────────────────────────────────

all_movements    = []
all_distances    = []
all_trajectories = {}

output_dir = r'C:\Users\arnau\Documents\projetde\runs-in-behind\outputs_loop_with_offsets'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("PROCESSING WITH SETPIECE OFFSETS")
print("=" * 80)
print("Movements whose START or END falls within these windows after a setpiece are EXCLUDED:")
print("ThrowIn:      first 2 seconds")
print("FreeKick:     first 3 seconds")
print("GoalKick:     first 3 seconds")
print("KickOff:      first 6 seconds")
print("CornerKick:   first 4 seconds")
print("(Movements that only cross a blackout window mid-course are kept.)")
print("=" * 80)
print()

# ── Loop over matches ─────────────────────────────────────────────────────────

for match_id in ls_match_ids:
    print(f"\n{'='*60}")
    print(f"Processing match: {match_id}")
    print(f"{'='*60}")
    
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
        (
            (df_mov['possession'] == df_mov['location']) |
            df_mov['possession_contested']) &
            (df_mov['speed_category'].isin(['running', 'sprinting'])) &
            (df_mov['distance_m'] > 3) &
            (df_mov['direction'] > 0.3) 
]
        df_runs.to_csv(
            os.path.join(output_dir, f'runs_behind_{match_id}.csv'), index=False
        )
        print(f"  → {len(df_runs)} runs-in-behind saved for {match_id}")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {match_id} — {e}")
        traceback.print_exc()

# ── Concatenate results across all matches ────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY ACROSS ALL MATCHES")
print(f"{'='*60}")

df_all_movements = pd.concat(all_movements, ignore_index=True)
df_all_distances = pd.concat(all_distances)

print(f"\nTotal movements across all matches: {len(df_all_movements)}")
print("\nSpeed category distribution:")
print(df_all_movements['speed_category'].value_counts())

# Count movements with ball touch
n_with_ball = df_all_movements['has_ball_touch'].sum()
n_total = len(df_all_movements)
print(f"\nMovements with ball possession: {n_with_ball}/{n_total} ({n_with_ball/n_total*100:.1f}%)")

# ── Save global files ─────────────────────────────────────────────────────────

df_all_movements.to_csv(os.path.join(output_dir, 'all_movements.csv'), index=False)
# df_all_distances.to_csv(os.path.join(output_dir, 'all_distances.csv'))

print(f"\n{'='*60}")
print(f"All files saved in: {output_dir}")
print(f"{'='*60}")