import os
import pandas as pd
import subprocess

# Parameters
match_id = "J03WPY"
match_name = "SF_12ST_F95_FCN"
video_path = "/home/max/drive/data/videos_openData2223/" + match_name + ".mp4"
csv_path = "/home/max/drive/coding/projects/27___deepRuns/data_deepruns/timecodes_" + match_id + ".csv"
output_dir = "/home/max/drive/data/videos_openData2223/" + match_id
fps = 25  # adjust if different

sec_before = 3
freeze_before = 1
freeze_after = 1
sec_after = 2

# Load CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    start_frame = row['absolute_start_frame']
    end_frame = row['absolute_end_frame']

    # Timepoints
    run_start_sec = start_frame / fps
    run_end_sec = end_frame / fps

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