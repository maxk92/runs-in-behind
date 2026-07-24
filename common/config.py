"""
Central path/config lookup for all scripts in this repo.

Every value can be overridden per machine via an environment variable
(e.g. in a local, untracked `.env` file loaded with `direnv`, or plain
`export RIB_DATA_DIR=...` in your shell) instead of editing source files.
The defaults below assume the layout documented in CLAUDE.md: raw DFL XML
under `/home/max/drive/data/openData2223`, and everything else (direction
JSON, manual annotations, script outputs) under the repo-local, git-ignored
`data/` directory.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, "data", *parts)


# --- Raw DFL data -----------------------------------------------------------
DATA_DIR  = _env_path("RIB_DATA_DIR", "/home/max/drive/data/openData2223")
VIDEO_DIR = _env_path("RIB_VIDEO_DIR", "/home/max/drive/data/videos_openData2223")

# --- Playing-direction JSON --------------------------------------------------
# `dict_direction[match_id]['Home'][half]` -> 'left_to_right' | 'right_to_left'
DIRECTION_JSON = _env_path("RIB_DIRECTION_JSON", _repo_path("direction_idsse_videos.json"))

# --- Video-sync offsets JSON --------------------------------------------------
# `{match_id: {'firstHalf': 'hh:mm:ss', 'secondHalf': 'hh:mm:ss'}}` -- the
# video-file timestamp at which each half's tracking data (frame 0) begins.
# Needed by cut_video_to_runs.py to seek into the raw match video, since
# recordings don't start exactly at kickoff and don't include a
# real-time-accurate halftime gap in the tracking data.
VIDEO_OFFSETS_JSON = _env_path("RIB_VIDEO_OFFSETS_JSON", _repo_path("offsets_idsse_videos.json"))

# --- Automated-pipeline output (Loop_with_setpiece_offsets.py -> runs_behind_*.csv) ---
AUTO_OUTPUT_DIR = _env_path("RIB_AUTO_OUTPUT_DIR", _repo_path("runs_behind"))

# --- Manual annotations (canonical schema, see common/annotations.py) -------
# One folder shared by Extraction_Evaluation.py, Stats_Manual_Annotation_App.py,
# and enriched_tool.py -- previously each pointed at a differently-named
# (and differently-shaped) folder; data/annotations/ is what's actually used.
ANNOTATION_DIR = _env_path("RIB_ANNOTATION_DIR", _repo_path("annotations"))

# --- Task 2c: enriched_tool.py output ---------------------------------------
ENRICHED_OUTPUT_DIR = _env_path("RIB_ENRICHED_OUTPUT_DIR", _repo_path("enriched_annotations"))

# --- Task 2a: Stats_Manual_Annotation_App.py output -------------------------
STATS_OUTPUT_DIR = _env_path("RIB_STATS_OUTPUT_DIR", _repo_path("annotation_stats"))

# --- Task 2b: Extraction_Evaluation.py false-negatives export ---------------
FALSE_NEGATIVES_PATH = _env_path("RIB_FALSE_NEGATIVES_PATH", _repo_path("false_negatives.csv"))

# --- contextualize_position_data.ipynb output -------------------------------
CONTEXT_OUTPUT_DIR = _env_path("RIB_CONTEXT_OUTPUT_DIR", _repo_path("context"))

# --- movement_classification.py output --------------------------------------
CLASSIFICATION_OUTPUT_DIR = _env_path(
    "RIB_CLASSIFICATION_OUTPUT_DIR", _repo_path("movement_classification"))

# --- pitch_control_tool.py / pitch_control_examples.py output ---------------
PITCH_CONTROL_OUTPUT_DIR = _env_path(
    "RIB_PITCH_CONTROL_OUTPUT_DIR", _repo_path("pitch_control"))

# --- cut_video_to_runs.py output --------------------------------------------
VIDEO_OUTPUT_DIR = _env_path("RIB_VIDEO_OUTPUT_DIR", _repo_path("video_clips"))

# --- Shared default match-id list (used by Loop_with_setpiece_offsets.py, ---
# --- Extraction_Evaluation.py, etc.) -- override with a comma-separated list.
_default_match_ids = "J03WMX,J03WN1,J03WPY,J03WOH,J03WQQ,J03WOY,J03WR9"
DEFAULT_MATCH_IDS = _env_path("RIB_MATCH_IDS", _default_match_ids).split(",")
