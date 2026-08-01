"""
Module 1 API router — player movement & pose analysis endpoints.

Changes relative to the original api-ready notebook (cells 10-13, 24):

Step 0 (cleanup):
  * Removed the broken shuttle-tracking stub from /api/process-video —
    the `TEMP_DIR = "/kaggle/working"` reassignment inside the function
    body shadowed the three `os.path.join(TEMP_DIR, ...)` calls above it
    (Python treats TEMP_DIR as local to the whole function once it's
    assigned anywhere inside it), so every request raised
    UnboundLocalError before background_tasks.add_task() ever ran.
  * The synchronous, unused `run_shuttle_tracking(...)` call is gone too
    — shuttle tracking gets reintroduced correctly, inside the
    background task, in Step 3.

Step 1 (shared paths + identifiers):
  * TEMP_DIR ("temp_exports") replaced with the shared UPLOADS_DIR /
    OUTPUTS_DIR from app.config, so Module 2's outputs/combine and
    outputs/m2 can live alongside these without collisions.
  * match_id / player_id are now validated with the same
    validate_player_match_metadata() Module 2 already uses, and a
    player_name field was added to match it (previously Module 1 didn't
    collect one). This is a breaking change to the existing request
    contract — the frontend form (Step 15) will need updating to match,
    and any caller still sending free-form ids like "match_001" /
    "player_01" will now get a 400 instead of being silently accepted.
"""
import concurrent.futures
import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .. import config
from ..module1.movement import extract_movement_features
from ..module1.render import render_movement_video, render_pose_video
from ..module1.tracking import run_tracking_inference
from ..shared.video_utils import transcode_to_h264

router = APIRouter()

# Background-task progress store, keyed by task_id (notebook cell 10).
# Module 2 will eventually report into this same dict too (Step 13) rather
# than keeping its own separate ANALYSIS_JOBS/ThreadPoolExecutor tracker
# (roadmap finding 0.4) — that unification happens in Step 13, not here.
progress_store = {}


def process_video_task(task_id, video_path, mp4_path, json_path, csv_path, match_id="p001_m0001", player_id="p001"):
    def progress_cb(current, total):
        pct = int((current / total) * 100) if total > 0 else 0
        progress_store[task_id] = {"status": "processing", "progress": pct}

    try:
        metrics = run_tracking_inference(
            video_path=video_path,
            out_mp4=mp4_path,
            out_json=json_path,
            out_csv=csv_path,
            model_path=config.POSE_MODEL_PATH,
            custom_model_path=str(config.MODULE1_CLASSIFIER_MODEL_PATH),
            progress_callback=progress_cb
        )

        # NOTE: these three filenames are still flat (not task_id-prefixed),
        # same as the original notebook — a second request processing
        # concurrently would overwrite these. Pre-existing limitation,
        # unchanged by Steps 0/1; worth revisiting once Step 13 unifies
        # the job model.
        movement_json_path = str(config.OUTPUTS_DIR / "movement_metrics.json")
        movement_csv_path = str(config.OUTPUTS_DIR / "movement_features.csv")
        movement_mp4_path = str(config.OUTPUTS_DIR / "movement_output.mp4")

        movement_metrics = extract_movement_features(
            input_json_path=json_path,
            output_json_path=movement_json_path,
            output_csv_path=movement_csv_path,
            output_video_path=None,
            original_video_path=None,
            match_id=match_id,
            player_id=player_id
        )

        progress_store[task_id]["progress"] = 90

        # Render both videos simultaneously using multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(render_pose_video, video_path, mp4_path, json_path)
            f2 = executor.submit(render_movement_video, movement_json_path, video_path, movement_mp4_path)
            concurrent.futures.wait([f1, f2])
            f1.result()
            f2.result()

        # Transcode both rendered videos to browser-playable H.264
        transcode_to_h264(mp4_path, mp4_path.replace(".mp4", "_web.mp4"))
        import os
        os.replace(mp4_path.replace(".mp4", "_web.mp4"), mp4_path)

        transcode_to_h264(movement_mp4_path, movement_mp4_path.replace(".mp4", "_web.mp4"))
        os.replace(movement_mp4_path.replace(".mp4", "_web.mp4"), movement_mp4_path)

        progress_store[task_id]["progress"] = 95

        progress_store[task_id] = {
            "status": "completed",
            "progress": 100,
            "metrics": metrics,
            "movement_metrics": movement_metrics,
            "exports": {
                "json_url": "/api/download/movement_metrics.json",
                "csv_url": "/api/download/output.csv",
                "mp4_url": "/api/download/output.mp4",
                "movement_csv_url": "/api/download/movement_features.csv",
                "movement_json_url": "/api/download/movement_metrics.json",
                "movement_mp4_url": "/api/download/movement_output.mp4"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints the ACTUAL underlying error to notebook output
        progress_store[task_id] = {"status": "failed", "progress": 0, "error": str(e)}


@router.post("/api/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    match_id: str = Form("p001_m0001"),
    player_id: str = Form("p001"),
    player_name: str = Form("shi_yuqi"),
):
    try:
        player_id, player_name, match_id = config.validate_player_match_metadata(
            player_id, player_name, match_id
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    video_path = config.UPLOADS_DIR / file.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_id = str(uuid.uuid4())
    progress_store[task_id] = {"status": "starting", "progress": 0}

    json_path = str(config.OUTPUTS_DIR / "output.json")
    csv_path = str(config.OUTPUTS_DIR / "output.csv")
    mp4_path = str(config.OUTPUTS_DIR / "output.mp4")

    background_tasks.add_task(
        process_video_task, task_id, str(video_path), mp4_path, json_path, csv_path, match_id, player_id
    )

    return {"task_id": task_id}


@router.get("/api/progress/{task_id}")
def get_progress(task_id: str):
    return progress_store.get(task_id, {"status": "not_found", "progress": 0})


@router.get("/api/progress-stream/{task_id}")
async def progress_stream(task_id: str):
    import asyncio
    import json as json_module

    async def event_generator():
        last_progress = -1
        last_status = None
        while True:
            data = progress_store.get(task_id, {"status": "not_found", "progress": 0})
            progress = data.get("progress", 0)
            status = data.get("status")

            if progress != last_progress or status != last_status:
                yield f"data: {json_module.dumps(data)}\n\n"
                last_progress = progress
                last_status = status

            if status in ["completed", "failed", "not_found"]:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/download/{filename}")
async def download_file(filename: str):
    # NOTE: Step 14 will add basename-only validation here
    # (Path(filename).name == filename) to close a path-traversal gap.
    # Left unchanged for now to keep this step's diff scoped to Step 0/1.
    file_path = config.OUTPUTS_DIR / filename
    if file_path.exists():
        media_type = "video/mp4" if filename.endswith(".mp4") else None
        return FileResponse(str(file_path), filename=filename, media_type=media_type)
    return {"error": "File not found"}
