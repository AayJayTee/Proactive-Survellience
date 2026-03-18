# api/main.py

import os
import sys
import shutil
import uuid
import json

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
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

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Model registry — populated once at startup
# ---------------------------------------------------
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy models once when the server starts."""
    print("⏳ Loading action recognition model...")
    m = generate_model(num_classes=101)
    m.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    models["action"] = m
    print("✅ Action model ready")

    print("⏳ Loading anomaly model...")
    models["anomaly"] = AnomalyInferencer(checkpoint_path=ANOMALY_MODEL_PATH, device=DEVICE)
    print("✅ Anomaly model ready")

    print("⏳ Loading captioning models (GIT + BLIP + PEGASUS) — takes ~1 min first time...")
    models["captioner"] = SceneCaptioner(device=DEVICE)
    print("✅ Captioning models ready")

    yield  # ← server accepts requests here

    models.clear()


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="Visionary AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Templates & Static Files
# ---------------------------------------------------
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ---------------------------------------------------
# Homepage
# ---------------------------------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------
# Video Analysis API
# ---------------------------------------------------
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(DATA_DIR, f"{file_id}.mp4")

    try:
        # 1️⃣ Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 2️⃣ Run pipeline in-process with pre-loaded models
        from run_pipeline import main as run_main
        report = run_main(
            input_path,
            action_model=models["action"],
            anomaly_inferencer=models["anomaly"],
            captioner=models["captioner"],
        )

        return report

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected server error: {str(e)}"}
        )