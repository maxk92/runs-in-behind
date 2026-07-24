import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess

import pandas as pd

from common import config
from common.timecodes import hms_to_seconds

# Parameters
match_id   = "J03WPY"
match_name = "SF_12ST_F95_FCN"     # video filename (without extension) for match_id
video_path = os.path.join(config.VIDEO_DIR, match_name + ".mp4")
output_dir = os.path.join(config.VIDEO_OUTPUT_DIR, match_id)
fps = 25  # adjust if different

sec_before = 3
freeze_before = 1
freeze_after = 1
sec_after = 2

os.makedirs(output_dir, exist_ok=True)

# Load the automated runs-in-behind output for this match (both halves),
# written by segmentation/extract_runs_behind.py as runs_behind_{match_id}.csv.
# (Previously this read a data_deepruns/timecodes_{match_id}.csv file with
# absolute_start_frame/absolute_end_frame columns that no script in this
# repo produces -- see below for how the video seek time is now derived
# from the half-relative start_frame/end_frame instead.)
csv_path = os.path.join(config.AUTO_OUTPUT_DIR, f"runs_behind_{match_id}.csv")
df = pd.read_csv(csv_path)

# Per-half video-sync offsets: the video timestamp at which each half's
# tracking data (frame 0) begins. Needed because match recordings don't
# start exactly at kickoff, and the real-time halftime gap isn't present in
# the tracking data at all. Defaults to 00:00:00 for any match/half not
# listed (or if the offsets file itself is missing/unreadable), so a
# missing/broken offsets file only costs sync accuracy, not a hard crash.
try:
    with open(config.VIDEO_OFFSETS_JSON, "r") as f:
        video_offsets = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as exc:
    print(f"[WARN] Could not load video offsets JSON at '{config.VIDEO_OFFSETS_JSON}': "
          f"{exc}. Assuming 00:00:00 offset for every half.")
    video_offsets = {}
half_offset_s = {
    half: hms_to_seconds(video_offsets.get(match_id, {}).get(half, "00:00:00"))
    for half in ("firstHalf", "secondHalf")
}

for idx, row in df.iterrows():
    start_frame = row['start_frame']
    end_frame   = row['end_frame']
    offset_sec  = half_offset_s[row['half']]

    # Timepoints (seconds into the actual video file)
    run_start_sec = offset_sec + start_frame / fps
    run_end_sec   = offset_sec + end_frame / fps

    pre_start_sec = max(run_start_sec - 3, 0)
    pre_freeze_time = run_start_sec
    post_freeze_time = run_end_sec
    post_run_start_sec = run_end_sec
    post_run_duration = 1.5

    # Durations
    pre_duration = run_start_sec - pre_start_sec
    run_duration = run_end_sec - run_start_sec

    # File name base
    base = f"r{idx:03d}_" + row["player"] + '_' + row["team"]
    files = {
        "pre": f"{output_dir}/{base}_pre.mp4",
        "pre_freeze_img": f"{output_dir}/{base}_prefreeze.png",
        "pre_freeze": f"{output_dir}/{base}_prefreeze.mp4",
        "run": f"{output_dir}/{base}_run.mp4",
        "post_freeze_img": f"{output_dir}/{base}_postfreeze.png",
        "post_freeze": f"{output_dir}/{base}_postfreeze.mp4",
        "post_run": f"{output_dir}/{base}_postrun.mp4",
        "concat_list": f"{output_dir}/{base}_concat.txt",
        "final": f"{output_dir}/{base}_final.mp4"
    }

    # 1. Extract pre-run video
    subprocess.run([
        "ffmpeg", "-ss", f"{pre_start_sec:.3f}", "-i", video_path,
        "-t", f"{pre_duration:.3f}", "-c", "copy", files["pre"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Extract freeze frame before run
    subprocess.run([
        "ffmpeg", "-ss", f"{pre_freeze_time:.3f}", "-i", video_path,
        "-frames:v", "1", files["pre_freeze_img"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-loop", "1", "-t", "1", "-i", files["pre_freeze_img"],
        "-vf", f"fps={fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        files["pre_freeze"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Extract run
    subprocess.run([
        "ffmpeg", "-ss", f"{run_start_sec:.3f}", "-i", video_path,
        "-t", f"{run_duration:.3f}", "-c", "copy", files["run"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Extract freeze frame after run
    subprocess.run([
        "ffmpeg", "-ss", f"{post_freeze_time:.3f}", "-i", video_path,
        "-frames:v", "1", files["post_freeze_img"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-loop", "1", "-t", "1", "-i", files["post_freeze_img"],
        "-vf", f"fps={fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        files["post_freeze"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 5. Extract post-run continuation
    subprocess.run([
        "ffmpeg", "-ss", f"{post_run_start_sec:.3f}", "-i", video_path,
        "-t", f"{post_run_duration:.3f}", "-c", "copy", files["post_run"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 6. Concatenate all parts
    with open(files["concat_list"], "w") as f:
        f.write(f"file '{os.path.abspath(files['pre'])}'\n")
        f.write(f"file '{os.path.abspath(files['pre_freeze'])}'\n")
        f.write(f"file '{os.path.abspath(files['run'])}'\n")
        f.write(f"file '{os.path.abspath(files['post_freeze'])}'\n")
        f.write(f"file '{os.path.abspath(files['post_run'])}'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", files["concat_list"],
        "-c", "copy", files["final"]
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Saved: {files['final']}")

    # Cleanup temp files (optional)
    for k, path in files.items():
        if k != "final" and os.path.exists(path):
            os.remove(path)
