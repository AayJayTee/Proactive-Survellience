import os
import sys
import json
import shutil
import uuid
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------
# Directory Setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUDIT_LOG_PATH = os.path.join(OUTPUT_DIR, "audit_log.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(AUDIT_LOG_PATH):
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8"):
        pass

# Ensure project root is importable
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from action_recognition.cnn3d_model import generate_model
from anomaly_detection.infer_anomaly import AnomalyInferencer
from captioning.caption_frames import SceneCaptioner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_MODEL_PATH = os.path.join(BASE_DIR, "action_recognition", "best_3dcnn.pth")
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "anomaly_detection", "best_anomaly_model.pth")

# ---------------------------------------------------
# In-memory stores
# ---------------------------------------------------
models = {}
jobs = {}
jobs_lock = Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_audit(event, job_id=None, details=None):
    row = {
        "timestamp": utc_now(),
        "event": event,
        "job_id": job_id,
        "details": details or {}
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def set_job(job_id, payload):
    with jobs_lock:
        jobs[job_id] = payload


def update_job(job_id, **changes):
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(changes)
        jobs[job_id]["updated_at"] = utc_now()


def get_job(job_id):
    with jobs_lock:
        return dict(jobs[job_id]) if job_id in jobs else None


def run_pipeline_job(job_id, input_path):
    update_job(
        job_id,
        status="running",
        current_stage="startup",
        stage_message="Pipeline started",
        progress=1.0,
        model_status={"action": "ready", "anomaly": "ready", "captioner": "ready"}
    )

    def progress_cb(update):
        stage_time = update.get("stage_time_sec")
        frames = update.get("frames")
        per_frame_ms = None

        if stage_time is not None and frames:
            per_frame_ms = round((float(stage_time) * 1000.0) / max(int(frames), 1), 2)

        update_job(
            job_id,
            status="running",
            current_stage=update.get("stage", "running"),
            stage_message=update.get("message", ""),
            progress=float(update.get("percent", 0.0)),
            stage_time_sec=stage_time,
            per_frame_ms=per_frame_ms,
            model_status={"action": "ready", "anomaly": "ready", "captioner": "ready"}
        )

    try:
        from run_pipeline import main as run_main

        report = run_main(
            input_path,
            action_model=models["action"],
            anomaly_inferencer=models["anomaly"],
            captioner=models["captioner"],
            output_subdir=job_id,
            progress_cb=progress_cb
        )

        update_job(
            job_id,
            status="completed",
            current_stage="done",
            stage_message="Analysis completed",
            progress=100.0,
            result=report,
            error=None,
            processing_time_sec=report.get("processing_time_sec"),
            per_frame_ms=report.get("avg_time_per_frame_ms")
        )
        write_audit("analyze_completed", job_id, {"processing_time_sec": report.get("processing_time_sec")})

    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            current_stage="failed",
            stage_message="Analysis failed",
            progress=100.0,
            error=str(exc)
        )
        write_audit("analyze_failed", job_id, {"error": str(exc)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading action model...")
    action_model = generate_model(num_classes=101)
    action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    action_model.to(DEVICE)
    action_model.eval()
    models["action"] = action_model
    print("Action model ready")

    print("Loading anomaly model...")
    models["anomaly"] = AnomalyInferencer(checkpoint_path=ANOMALY_MODEL_PATH, device=DEVICE)
    print("Anomaly model ready")

    print("Loading captioning models...")
    models["captioner"] = SceneCaptioner(device=DEVICE)
    print("Captioning models ready")

    write_audit("server_start", details={"device": DEVICE})
    yield
    write_audit("server_stop")
    models.clear()


app = FastAPI(title="Visionary AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/analyze")
async def analyze_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(DATA_DIR, job_id + ".mp4")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        set_job(job_id, {
            "job_id": job_id,
            "file_name": video.filename,
            "input_path": input_path,
            "status": "queued",
            "progress": 0.0,
            "current_stage": "queued",
            "stage_message": "Queued for processing",
            "stage_time_sec": None,
            "per_frame_ms": None,
            "processing_time_sec": None,
            "model_status": {"action": "ready", "anomaly": "ready", "captioner": "ready"},
            "result": None,
            "error": None,
            "created_at": utc_now(),
            "updated_at": utc_now()
        })

        write_audit("analyze_requested", job_id, {"file_name": video.filename})
        background_tasks.add_task(run_pipeline_job, job_id, input_path)

        return {
            "job_id": job_id,
            "status_url": "/api/status/" + job_id,
            "message": "Processing started"
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected server error: " + str(exc)}
        )


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record

@app.get("/api/audit-log")
async def get_audit_log(limit: int = 200):
    if not os.path.exists(AUDIT_LOG_PATH):
        return {"rows": []}

    rows = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    return {"rows": rows[-max(limit, 1):]}

@app.get("/api/data/summary")
async def data_summary():
    input_files = 0
    output_dirs = 0
    output_files = 0

    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            p = os.path.join(DATA_DIR, name)
            if os.path.isfile(p):
                input_files += 1

    if os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, name)
            # ignore audit log for "saved analysis data" check
            if os.path.abspath(p) == os.path.abspath(AUDIT_LOG_PATH):
                continue
            if os.path.isdir(p):
                output_dirs += 1
            elif os.path.isfile(p):
                output_files += 1

    has_data = (input_files + output_dirs + output_files) > 0

    return {
        "has_data": has_data,
        "input_files": input_files,
        "output_dirs": output_dirs,
        "output_files": output_files
    }

@app.delete("/api/data/{job_id}")
async def delete_saved_data(job_id: str):
    record = get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    deleted = {"input_deleted": False, "output_deleted": False}

    input_path = record.get("input_path")
    if input_path and os.path.exists(input_path):
        os.remove(input_path)
        deleted["input_deleted"] = True

    output_subdir = os.path.join(OUTPUT_DIR, job_id)
    if os.path.isdir(output_subdir):
        shutil.rmtree(output_subdir)
        deleted["output_deleted"] = True

    write_audit("data_deleted", job_id, deleted)
    update_job(job_id, stage_message="Saved data deleted")
    return {"job_id": job_id, "deleted": deleted}

@app.delete("/api/data")
async def delete_all_saved_data(include_audit: bool = False):
    # Block delete while jobs are active
    with jobs_lock:
        active = [jid for jid, rec in jobs.items() if rec.get("status") in {"queued", "running"}]
    if active:
        raise HTTPException(status_code=409, detail="Cannot delete all data while jobs are running.")

    deleted = {
        "input_files_deleted": 0,
        "output_dirs_deleted": 0,
        "output_files_deleted": 0,
        "audit_deleted": False
    }

    # Delete uploaded videos
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            p = os.path.join(DATA_DIR, name)
            if os.path.isfile(p):
                os.remove(p)
                deleted["input_files_deleted"] += 1

    # Delete outputs (optionally keep audit)
    if os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            p = os.path.join(OUTPUT_DIR, name)
            if os.path.abspath(p) == os.path.abspath(AUDIT_LOG_PATH) and not include_audit:
                continue
            if os.path.isdir(p):
                shutil.rmtree(p)
                deleted["output_dirs_deleted"] += 1
            elif os.path.isfile(p):
                os.remove(p)
                deleted["output_files_deleted"] += 1

    # Recreate empty audit file if deleted
    if include_audit:
        deleted["audit_deleted"] = True
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8"):
            pass
    else:
        write_audit("all_data_deleted", details=deleted)

    with jobs_lock:
        jobs.clear()

    return {"message": "All saved data deleted", "deleted": deleted}