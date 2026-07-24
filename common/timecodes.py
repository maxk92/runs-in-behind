"""
Frame-number -> timecode-string conversion, parameterized to cover the two
formats previously duplicated as separate functions:
  - Extraction_Evaluation.py's `frames_to_tc`      -> fmt="hms"  ("hh:mm:ss")
  - Stats_Manual_Annotation_App.py's `frame_to_timecode` -> fmt="ms" ("mm:ss.mmm")
"""

import pandas as pd


def frames_to_timecode(frames, fps: float = 25, fmt: str = "hms") -> str:
    """
    Convert a frame number to a timecode string.

    fmt="hms": "hh:mm:ss" (matches Extraction_Evaluation.py's frames_to_tc).
               Returns '' for NaN input.
    fmt="ms":  "mm:ss.mmm" (matches Stats_Manual_Annotation_App.py's
               frame_to_timecode).
    """
    if fmt == "hms":
        if pd.isna(frames):
            return ""
        total_sec = int(frames) // fps
        h = int(total_sec) // 3600
        m = (int(total_sec) % 3600) // 60
        s = int(total_sec) % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    if fmt == "ms":
        total_seconds = frames / fps
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    raise ValueError(f"Unknown fmt: {fmt!r} (expected 'hms' or 'ms')")


def hms_to_seconds(hms: str) -> float:
    """Convert an 'hh:mm:ss' string (as used in offsets_idsse_videos.json)
    to total seconds."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
