# api/main.py

import os
import shutil
import uuid
import json
import subprocess

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------
app = FastAPI(title="Visionary AI Backend")

# Enable CORS (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Directory Setup
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Templates & Static Mounting
# ---------------------------------------------------
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

app.mount(
    "/output",
    StaticFiles(directory=OUTPUT_DIR),
    name="output"
)

# ---------------------------------------------------
# Homepage Route
# ---------------------------------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ---------------------------------------------------
# Video Analysis API
# ---------------------------------------------------
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):

    try:
        # ---------------------------------------------------
        # 1️⃣ Save Uploaded Video
        # ---------------------------------------------------
        file_id = str(uuid.uuid4())
        input_path = os.path.join(DATA_DIR, f"{file_id}.mp4")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # ---------------------------------------------------
        # 2️⃣ Run Full AI Pipeline
        # ---------------------------------------------------
        subprocess.run(
            ["python", "run_pipeline.py", input_path],
            check=True
        )

        # ---------------------------------------------------
        # 3️⃣ Load Final Report
        # ---------------------------------------------------
        report_path = os.path.join(OUTPUT_DIR, "final_report.json")

        if not os.path.exists(report_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Pipeline failed to generate report"}
            )

        with open(report_path, "r") as f:
            report = json.load(f)

        # ---------------------------------------------------
        # 4️⃣ Extract Action Recognition
        # ---------------------------------------------------
        action_data = report.get("action_recognition", {})

        action_label = action_data.get("action", "Unknown")
        action_confidence = action_data.get("confidence", 0.0)

        # ---------------------------------------------------
        # 5️⃣ Extract Anomaly Information Safely
        # ---------------------------------------------------
        anomaly_data = report.get("anomaly", {})

        anomaly_score = None
        anomaly_status = "Unknown"

        if isinstance(anomaly_data, dict):
            anomaly_score = anomaly_data.get("score") or anomaly_data.get("max_score")

        if anomaly_score is not None:
            anomaly_status = "Anomaly 🚨" if anomaly_score > 0.5 else "Normal ✅"

        # ---------------------------------------------------
        # 6️⃣ Extract Caption
        # ---------------------------------------------------
        caption_text = report.get("caption", "No caption generated")

        # ---------------------------------------------------
        # 7️⃣ Final Clean Response for Frontend
        # ---------------------------------------------------
        response = {
            "action": {
                "label": action_label,
                "confidence": action_confidence
            },
            "anomaly": {
                "score": anomaly_score,
                "status": anomaly_status
            },
            "caption": caption_text,
            "outputs": {
                "gradcam_video": "/output/gradcam_video.mp4",
                "activation_video": "/output/activation_video.mp4"
            }
        }

        return response

    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Pipeline execution failed: {str(e)}"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )
