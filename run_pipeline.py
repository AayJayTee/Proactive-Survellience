# run_pipeline.py
# ============================================================
# FINAL END-TO-END PIPELINE
# Action + Anomaly + Explainability + Captioning
# ============================================================

import os
import cv2
import json
import sys
import torch
import time
import numpy as np
import warnings
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- SILENCE ALL WARNINGS ----------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "0"

warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# ---------------- IMPORTS ----------------
from action_recognition.infer_action import run_action_recognition
from action_recognition.cnn3d_model import generate_model
from anomaly_detection.infer_anomaly import AnomalyInferencer
from explainability.gradcam_action import ActionGradCAM, save_gradcam_video
from explainability.activation_maps import ActivationMapExplainer, save_activation_video
from captioning.caption_frames import SceneCaptioner, save_captions

# ---------------- CONFIG ----------------
ACTION_MODEL_PATH = "action_recognition/best_3dcnn.pth"
ANOMALY_MODEL_PATH = "anomaly_detection/anomaly_model.pth"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_KEYFRAMES = 3  # number of keyframes to extract for captioning (reduce for speed)


# ============================================================
# EVENT BUILDER
# ============================================================
def build_timeline(anomaly_scores, fps, action_label, action_conf, threshold):
    events = []
    active = None

    for i, score in enumerate(anomaly_scores):
        time_sec = i / fps

        if score > threshold:
            if active is None:
                active = {
                    "start": time_sec,
                    "max_score": float(score)
                }
            else:
                active["max_score"] = max(active["max_score"], float(score))
        else:
            if active:
                active["end"] = time_sec
                active["type"] = "Anomaly"
                events.append(active)
                active = None

    # Add action event
    events.append({
        "start": 0,
        "end": len(anomaly_scores)/fps,
        "type": action_label,
        "confidence": float(action_conf)
    })

    return events


# ============================================================
# REPORT GENERATION
# ============================================================
def export_csv(events, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)


def export_pdf(report_data, path):
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Visionary AI Surveillance Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(report_data["caption"], styles["Normal"]))
    elements.append(Spacer(1, 12))

    for e in report_data["timeline"]:
        text = f"{e['type']} event from {e['start']:.2f}s to {e['end']:.2f}s"
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)


# ============================================================
# MAIN PIPELINE
# ============================================================
def emit_progress(progress_cb, stage, percent, message, stage_time_sec=None, frames=None):
    if progress_cb is None:
        return
    payload = {
        "stage": stage,
        "percent": float(percent),
        "message": message
    }
    if stage_time_sec is not None:
        payload["stage_time_sec"] = float(stage_time_sec)
    if frames is not None:
        payload["frames"] = int(frames)
    progress_cb(payload)


def main(
    VIDEO_PATH,
    action_model=None,
    anomaly_inferencer=None,
    captioner=None,
    output_subdir=None,
    progress_cb=None
):
    pipeline_start = time.perf_counter()
    stage_times = {}

    if output_subdir:
        output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        output_url_prefix = "/output/" + output_subdir
    else:
        output_dir = OUTPUT_DIR
        output_url_prefix = "/output"

    os.makedirs(output_dir, exist_ok=True)
    emit_progress(progress_cb, "init", 2, "Loading video metadata")

    # Reuse or load action model
    if action_model is None:
        action_model = generate_model(num_classes=101)
        action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
        action_model.to(DEVICE)
        action_model.eval()

    # Decode video once for timing and captioning reuse
    t_decode = time.perf_counter()
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    stage_times["decode_sec"] = round(time.perf_counter() - t_decode, 4)
    emit_progress(progress_cb, "decode", 8, "Video decoded", stage_times["decode_sec"], max(total_frames, 1))

    # 1) ACTION
    t_action = time.perf_counter()
    action_result = run_action_recognition(
        VIDEO_PATH,
        ACTION_MODEL_PATH,
        preloaded_model=action_model
    )
    stage_times["action_sec"] = round(time.perf_counter() - t_action, 4)
    emit_progress(progress_cb, "action", 25, "Action recognition done", stage_times["action_sec"], max(total_frames, 1))

    # 2) ANOMALY
    t_anomaly = time.perf_counter()
    if anomaly_inferencer is None:
        anomaly_inferencer = AnomalyInferencer(
            checkpoint_path=ANOMALY_MODEL_PATH,
            device=DEVICE
        )

    anomaly_scores = anomaly_inferencer.infer(VIDEO_PATH, total_frames=total_frames)
    anomaly_threshold = anomaly_inferencer.threshold

    stage_times["anomaly_sec"] = round(time.perf_counter() - t_anomaly, 4)
    emit_progress(progress_cb, "anomaly", 50, "Anomaly inference done", stage_times["anomaly_sec"], max(total_frames, 1))

    # 3) EXPLAINABILITY: GRADCAM
    t_gradcam = time.perf_counter()
    gradcam = ActionGradCAM(ACTION_MODEL_PATH)
    cam_maps, _ = gradcam.generate(VIDEO_PATH)
    save_gradcam_video(VIDEO_PATH, cam_maps, os.path.join(output_dir, "gradcam_video.mp4"))
    stage_times["gradcam_sec"] = round(time.perf_counter() - t_gradcam, 4)
    emit_progress(progress_cb, "gradcam", 68, "Grad-CAM video generated", stage_times["gradcam_sec"], max(total_frames, 1))

    # 4) EXPLAINABILITY: ACTIVATION
    t_activation = time.perf_counter()
    activation_explainer = ActivationMapExplainer(ACTION_MODEL_PATH)
    activation_maps = activation_explainer.generate(VIDEO_PATH)
    save_activation_video(VIDEO_PATH, activation_maps, os.path.join(output_dir, "activation_video.mp4"))
    stage_times["activation_sec"] = round(time.perf_counter() - t_activation, 4)
    emit_progress(progress_cb, "activation", 82, "Activation map video generated", stage_times["activation_sec"], max(total_frames, 1))

    # 5) CAPTIONING
    t_caption = time.perf_counter()
    if captioner is None:
        captioner = SceneCaptioner(device=DEVICE)

    keyframes = all_frames[::max(len(all_frames) // CAPTION_KEYFRAMES, 1)]
    frame_captions = captioner.caption_video_frames(keyframes)
    final_caption = captioner.build_final_caption(
        frame_captions,
        action_label=action_result["action"],
        action_confidence=action_result["confidence"]
    )
    stage_times["caption_sec"] = round(time.perf_counter() - t_caption, 4)
    emit_progress(progress_cb, "caption", 92, "Captioning completed", stage_times["caption_sec"], max(len(keyframes), 1))

    # 6) TIMELINE + ALERTS
    t_post = time.perf_counter()
    timeline = build_timeline(
        anomaly_scores,
        fps,
        action_result["action"],
        action_result["confidence"],
        threshold=anomaly_threshold
    )

    alerts = []
    for e in timeline:
        if e.get("max_score", 0) > (anomaly_threshold * 2.0):
            alerts.append("High anomaly detected")
    stage_times["postprocess_sec"] = round(time.perf_counter() - t_post, 4)

    # 7) REPORT
    processing_time_sec = round(time.perf_counter() - pipeline_start, 3)
    avg_time_per_frame_ms = round((processing_time_sec * 1000.0) / max(total_frames, 1), 3)

    report = {
        "action_recognition": action_result,
        "anomaly": {
            "max_score": float(anomaly_scores.max()),
            "mean_score": float(anomaly_scores.mean()),
            "threshold": float(anomaly_threshold)
        },
        "caption": final_caption,
        "timeline": timeline,
        "alerts": alerts,
        "video_meta": {
            "total_frames": int(total_frames),
            "fps": float(fps)
        },
        "processing_time_sec": processing_time_sec,
        "avg_time_per_frame_ms": avg_time_per_frame_ms,
        "stage_times_sec": stage_times,
        "outputs": {
            "gradcam_video": output_url_prefix + "/gradcam_video.mp4",
            "activation_video": output_url_prefix + "/activation_video.mp4"
        }
    }

    with open(os.path.join(output_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    export_csv(timeline, os.path.join(output_dir, "events.csv"))
    export_pdf(report, os.path.join(output_dir, "report.pdf"))

    emit_progress(progress_cb, "done", 100, "Report ready", processing_time_sec, max(total_frames, 1))
    return report


if __name__ == "__main__":
    VIDEO_PATH = sys.argv[1]
    main(VIDEO_PATH)