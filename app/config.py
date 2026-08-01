"""
Shared configuration for the badminton analysis API.

Both Module 1 (pose/movement) and Module 2 (shuttle/tactical) read their
paths, model locations, and identifier rules from here so the two pipelines
can never disagree about where uploads/outputs live or what a valid
match_id/player_id looks like.

Folder structure expected on disk (all relative to BASE):

    project/
      app/                   <- this package
      TrackNetV3/            cloned + patched (Step 2)
      models/
        best.pt               Module 1 shot/stance classifier
        player_best.pt        Module 2 player detector + shot classifier
        court_best.pt         Module 2 court keypoint detector
      uploads/
      outputs/
        combine/              Module 2 "p1" tactical output (Step 11)
        m2/                   Module 2 "p2" per-shot landing output (Step 11)
      static/

Override the root with the BADMINTON_APP_BASE env var, e.g. "/kaggle/working".
"""
import os
import re
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE = Path(os.getenv("BADMINTON_APP_BASE", Path(__file__).resolve().parent.parent))

TRACKNET_DIR = BASE / "TrackNetV3"
MODELS_DIR = BASE / "models"
UPLOADS_DIR = BASE / "uploads"
OUTPUTS_DIR = BASE / "outputs"
STATIC_DIR = BASE / "static"

# Module 2 writes its two JSON deliverables into these (Step 11)
COMBINE_OUTPUTS_DIR = OUTPUTS_DIR / "combine"
M2_OUTPUTS_DIR = OUTPUTS_DIR / "m2"

for _dir in (UPLOADS_DIR, OUTPUTS_DIR, COMBINE_OUTPUTS_DIR, M2_OUTPUTS_DIR, STATIC_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Model paths
# ------------------------------------------------------------------
# Module 1 — generic pose model + Hirusha's fine-tuned 7-class shot/stance
# classifier. Same defaults as the original notebook; override via env var
# instead of editing this file.
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt")
MODULE1_CLASSIFIER_MODEL_PATH = Path(os.getenv(
    "MODULE1_CLASSIFIER_MODEL_PATH",
    "/kaggle/input/models/chathushikavindya/best-pt/pytorch/default/1/best.pt",
))

# Module 2 (Steps 4-5)
PLAYER_MODEL_PATH = Path(os.getenv("PLAYER_MODEL_PATH", MODELS_DIR / "player_best.pt"))
COURT_MODEL_PATH = Path(os.getenv("COURT_MODEL_PATH", MODELS_DIR / "court_best.pt"))

# TrackNetV3 checkpoints (Step 2)
TRACKNET_PT = Path(os.getenv("TRACKNET_PT", TRACKNET_DIR / "ckpts" / "TrackNet_best.pt"))
INPAINT_PT = Path(os.getenv("INPAINT_PT", TRACKNET_DIR / "ckpts" / "InpaintNet_best.pt"))


# ------------------------------------------------------------------
# Shared match_id / player_id contract (roadmap finding 0.5 / Step 1)
#
# Module 1 originally accepted free-form strings ("match_001", "player_01").
# Module 2 already validates strictly. We standardize on Module 2's format
# since it maps directly onto the Firestore schema shape
# players/{player_id}/matches/{match_id}. Copied verbatim from Module 2's
# pipeline (pipeline_main.py / cell 18, line ~2384) — logic unchanged, only
# relocated so both endpoints import the same function instead of each
# enforcing their own rule.
# ------------------------------------------------------------------
def validate_player_match_metadata(player_id, player_name, match_id):
    player_id = str(player_id).strip()
    player_name = str(player_name).strip()
    match_id = str(match_id).strip()

    if not re.fullmatch(r"p\d{3}", player_id):
        raise ValueError(
            "player_id must use the format p001, p002, etc."
        )

    expected_pattern = rf"{re.escape(player_id)}_m\d{{4}}"

    if not re.fullmatch(expected_pattern, match_id):
        raise ValueError(
            f"match_id must use the format {player_id}_m0001."
        )

    match_number = int(match_id.rsplit("_m", 1)[1])

    if not 1 <= match_number <= 20:
        raise ValueError(
            "Match number must be between 0001 and 0020."
        )

    if not player_name:
        raise ValueError("player_name cannot be empty.")

    return player_id, player_name, match_id
