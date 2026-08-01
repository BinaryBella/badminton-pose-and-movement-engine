"""
Step 2 — TrackNetV3 environment setup (no API wiring yet).

Pure environment bootstrapping: clone TrackNetV3, patch it for PyTorch 2.x
and safe single-worker/single-batch inference, and put the checkpoints
where app/config.py expects them. Run once per environment (e.g. once per
fresh Kaggle session), not per request.

Moved out of pythoncode-with-html-only-combine-p1-m2-p2-frontend-fixed.ipynb
cells 3-14 (git clone, torch.load patches, DataLoader safety patch,
checkpoint acquisition, standalone test), pointed at the shared
TRACKNET_DIR / MODELS_DIR / TRACKNET_PT / INPAINT_PT from app.config
instead of that notebook's own WORKING_DIR = Path("/kaggle/working").
Every replacement string, the gdown file id, and the predict.py patches
are unchanged from the original.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import gdown
import torch

from app import config

TRACKNET_DIR = config.TRACKNET_DIR
MODELS_DIR = config.MODELS_DIR
CKPT_DIR = TRACKNET_DIR / "ckpts"
TRACKNET_PT = config.TRACKNET_PT
INPAINT_PT = config.INPAINT_PT


# ------------------------------------------------------------------
# Locate model/checkpoint files among Kaggle input datasets (cell 5)
# ------------------------------------------------------------------
def find_input_file(input_dir, exact_names=(), suffixes=(), contains=()):
    files = [p for p in Path(input_dir).rglob("*") if p.is_file()]
    lower_exact = {name.lower() for name in exact_names}

    for path in files:
        if path.name.lower() in lower_exact:
            return path

    for path in files:
        name = path.name.lower()
        suffix_matches = not suffixes or any(name.endswith(s.lower()) for s in suffixes)
        contains_matches = not contains or all(t.lower() in name for t in contains)
        if suffix_matches and contains_matches:
            return path

    return None


def locate_kaggle_inputs(input_dir):
    """Find player_best.pt / court_best.pt (required) and
    TrackNet_best.pt / InpaintNet_best.pt (optional — will be downloaded
    if not found) among the notebook's attached Kaggle inputs."""
    sources = {
        "player": find_input_file(input_dir, exact_names=("player_best.pt",), suffixes=(".pt",), contains=("player",)),
        "court": find_input_file(input_dir, exact_names=("court_best.pt",), suffixes=(".pt",), contains=("court",)),
        "tracknet": find_input_file(input_dir, exact_names=("TrackNet_best.pt", "tracknet_best.pt"), suffixes=(".pt",), contains=("tracknet",)),
        "inpaint": find_input_file(input_dir, exact_names=("InpaintNet_best.pt", "inpaintnet_best.pt"), suffixes=(".pt",), contains=("inpaint",)),
    }

    print("Player model:", sources["player"])
    print("Court model:", sources["court"])
    print("TrackNet model:", sources["tracknet"] or "Will download automatically")
    print("InpaintNet model:", sources["inpaint"] or "Will download automatically")

    if sources["player"] is None:
        raise FileNotFoundError("Add a Kaggle input containing player_best.pt")
    if sources["court"] is None:
        raise FileNotFoundError("Add a Kaggle input containing court_best.pt")

    return sources


# ------------------------------------------------------------------
# Clone TrackNetV3 (cell 6)
# ------------------------------------------------------------------
def clone_tracknet():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not (TRACKNET_DIR / "predict.py").exists():
        if TRACKNET_DIR.exists():
            shutil.rmtree(TRACKNET_DIR)
        subprocess.run(
            ["git", "clone", "https://github.com/qaz812345/TrackNetV3.git", str(TRACKNET_DIR)],
            check=True,
        )
    else:
        print("TrackNetV3 already exists at", TRACKNET_DIR)
    print("predict.py exists:", (TRACKNET_DIR / "predict.py").exists())


# ------------------------------------------------------------------
# PyTorch 2.x torch.load compatibility patch (cell 8)
# ------------------------------------------------------------------
def patch_text_file(path, replacements):
    path = Path(path)
    if not path.exists():
        print("Skipped missing file:", path)
        return
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("Patched" if text != original else "Already patched", path.name)


def patch_torch_load():
    map_location = "cuda" if torch.cuda.is_available() else "cpu"

    patch_text_file(TRACKNET_DIR / "predict.py", [
        ("torch.load(args.tracknet_file)",
         f"torch.load(args.tracknet_file, map_location='{map_location}', weights_only=False)"),
        ("torch.load(args.inpaintnet_file)",
         f"torch.load(args.inpaintnet_file, map_location='{map_location}', weights_only=False)"),
    ])

    patch_text_file(TRACKNET_DIR / "train.py", [
        ("torch.load(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))",
         f"torch.load(os.path.join(args.save_dir, f'{{args.model_name}}_cur.pt'), map_location='{map_location}', weights_only=False)"),
    ])

    patch_text_file(TRACKNET_DIR / "test.py", [
        ("torch.load(args.tracknet_file)",
         f"torch.load(args.tracknet_file, map_location='{map_location}', weights_only=False)"),
        ("torch.load(args.inpaintnet_file)",
         f"torch.load(args.inpaintnet_file, map_location='{map_location}', weights_only=False)"),
    ])


# ------------------------------------------------------------------
# Safe num_workers / batch_size patch for predict.py (cell 11)
# ------------------------------------------------------------------
def patch_dataloader_safety():
    predict_path = TRACKNET_DIR / "predict.py"
    code_text = predict_path.read_text(encoding="utf-8")

    code_text = code_text.replace(
        "num_workers = args.batch_size if args.batch_size <= 16 else 16",
        "num_workers = 0",
    )
    # Replace only the DataLoader batch-size argument, not every occurrence
    # of args.batch_size.
    code_text = re.sub(r"batch_size\s*=\s*args\.batch_size", "batch_size=1", code_text)
    predict_path.write_text(code_text, encoding="utf-8")
    print("TrackNet configured with num_workers=0 and prediction batch_size=1")


def verify_patches():
    """Print every patched line back out so it's easy to eyeball after setup."""
    for line_number, line in enumerate(
        (TRACKNET_DIR / "predict.py").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if "num_workers" in line or "batch_size=" in line or "torch.load" in line:
            print(f"{line_number}: {line}")


# ------------------------------------------------------------------
# Checkpoint acquisition (cell 9)
# ------------------------------------------------------------------
def acquire_checkpoints(tracknet_source=None, inpaint_source=None,
                         gdown_file_id="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"):
    """Get TrackNet_best.pt / InpaintNet_best.pt into CKPT_DIR.

    Uses the Kaggle input copies when given, otherwise downloads the same
    checkpoint ZIP the Colab notebook used.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if tracknet_source is not None:
        shutil.copy2(tracknet_source, TRACKNET_PT)
    if inpaint_source is not None:
        shutil.copy2(inpaint_source, INPAINT_PT)

    if not TRACKNET_PT.exists() or not INPAINT_PT.exists():
        zip_path = config.BASE / "TrackNetV3_ckpts.zip"
        print("Downloading TrackNet checkpoints from Google Drive...")
        downloaded = gdown.download(id=gdown_file_id, output=str(zip_path), quiet=False)
        if downloaded is None or not zip_path.exists():
            raise RuntimeError("Checkpoint download failed. Confirm internet access is enabled.")

        extract_dir = config.BASE / "tracknet_ckpt_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        found_tracknet = next(extract_dir.rglob("TrackNet_best.pt"), None)
        found_inpaint = next(extract_dir.rglob("InpaintNet_best.pt"), None)
        if found_tracknet is None or found_inpaint is None:
            raise FileNotFoundError("The downloaded ZIP did not contain both TrackNet checkpoints")

        shutil.copy2(found_tracknet, TRACKNET_PT)
        shutil.copy2(found_inpaint, INPAINT_PT)

    assert TRACKNET_PT.exists(), f"Missing {TRACKNET_PT}"
    assert INPAINT_PT.exists(), f"Missing {INPAINT_PT}"
    print("TrackNet checkpoint:", TRACKNET_PT, TRACKNET_PT.exists())
    print("InpaintNet checkpoint:", INPAINT_PT, INPAINT_PT.exists())


# ------------------------------------------------------------------
# Player / court model files (cell 13, model-copy portion only —
# the os.environ writes from the original cell are gone: app.config
# already derives these same paths from BADMINTON_APP_BASE, so pushing
# them back into env vars here would just be two sources of truth for
# the same thing)
# ------------------------------------------------------------------
def setup_player_and_court_models(player_source, court_source):
    shutil.copy2(player_source, config.PLAYER_MODEL_PATH)
    shutil.copy2(court_source, config.COURT_MODEL_PATH)
    print("Player model:", config.PLAYER_MODEL_PATH, config.PLAYER_MODEL_PATH.exists())
    print("Court model:", config.COURT_MODEL_PATH, config.COURT_MODEL_PATH.exists())


# ------------------------------------------------------------------
# Optional standalone test, independent of the API (cell 14)
# ------------------------------------------------------------------
def run_standalone_test(video_path):
    prediction_dir = config.BASE / "tracknet_prediction_test"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(TRACKNET_DIR / "predict.py"),
        "--video_file", str(video_path),
        "--tracknet_file", str(TRACKNET_PT),
        "--inpaintnet_file", str(INPAINT_PT),
        "--save_dir", str(prediction_dir),
        "--large_video",
        "--eval_mode", "nonoverlap",
        "--batch_size", "1",
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(TRACKNET_DIR), check=True)

    csv_files = list(prediction_dir.glob("*_ball.csv"))
    if not csv_files:
        raise RuntimeError("TrackNet produced no ball CSV")
    print("Standalone TrackNet prediction complete:", csv_files[0])
    return csv_files[0]


def run_full_setup(input_dir):
    """Convenience wrapper: everything except the optional standalone test."""
    clone_tracknet()
    patch_torch_load()
    patch_dataloader_safety()
    verify_patches()

    sources = locate_kaggle_inputs(input_dir)
    acquire_checkpoints(tracknet_source=sources["tracknet"], inpaint_source=sources["inpaint"])
    setup_player_and_court_models(sources["player"], sources["court"])
    print("TrackNetV3 environment ready.")
