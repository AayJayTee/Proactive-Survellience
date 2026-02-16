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
import numpy as np
import warnings

# ---------------- SILENCE ALL WARNINGS ----------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "0"

warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# ---------------- ACTION ----------------
from action_recognition.infer_action import run_action_recognition
from action_recognition.cnn3d_model import generate_model

# ---------------- ANOMALY ----------------
from anomaly_detection.feature_extractor import VideoFeatureExtractor
from anomaly_detection.infer_anomaly import AnomalyInferencer

# ---------------- EXPLAINABILITY ----------------
from explainability.gradcam_action import ActionGradCAM, save_gradcam_video
from explainability.activation_maps import (
    ActivationMapExplainer,
    save_activation_video
)

# ---------------- CAPTIONING ----------------
from captioning.caption_frames import (
    SceneCaptioner,
    save_captions
)

# ---------------- CONFIG ----------------
VIDEO_PATH = r"data\videos\v_BoxingPunchingBag_g01_c03.avi"

ACTION_MODEL_PATH = "action_recognition/best_3dcnn.pth"
ANOMALY_MODEL_PATH = "anomaly_detection/best_anomaly_model.pth"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


def select_keyframes(video_path, anomaly_scores, top_k=5):
    """
    Select keyframes based on anomaly peaks
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return []

    idxs = np.argsort(anomaly_scores)[-top_k:]
    idxs = sorted(idxs)

    keyframes = [frames[i] for i in idxs if i < len(frames)]
    return keyframes


def main():
    print("\n🚀 Starting Visionary Pipeline\n")

    # =========================================================
    # 1️⃣ ACTION RECOGNITION
    # =========================================================
    print("🔹 Running Action Recognition...")
    action_result = run_action_recognition(
        VIDEO_PATH,
        ACTION_MODEL_PATH
    )

    print(f"   Action: {action_result['action']} "
          f"({action_result['confidence']:.2f})")

    # =========================================================
    # 2️⃣ ANOMALY DETECTION
    # =========================================================
    print("\n🔹 Running Anomaly Detection...")

    backbone = generate_model(num_classes=101)
    backbone.load_state_dict(
        torch.load(ACTION_MODEL_PATH, map_location=DEVICE)
    )
    backbone.eval()

    feature_extractor = VideoFeatureExtractor(backbone, device=DEVICE)
    anomaly_model = AnomalyInferencer(
        checkpoint_path=ANOMALY_MODEL_PATH,
        device=DEVICE
    )

    features = feature_extractor.extract(VIDEO_PATH)

    total_frames = int(
        cv2.VideoCapture(VIDEO_PATH)
        .get(cv2.CAP_PROP_FRAME_COUNT)
    )

    anomaly_scores = anomaly_model.infer(features, total_frames)
    np.save(os.path.join(OUTPUT_DIR, "anomaly_scores.npy"), anomaly_scores)

    print(f"   Max anomaly score: {anomaly_scores.max():.4f}")

    # =========================================================
    # 3️⃣ EXPLAINABILITY
    # =========================================================
    print("\n🔹 Generating Explainability Outputs...")

    gradcam = ActionGradCAM(ACTION_MODEL_PATH)
    cam_maps, _ = gradcam.generate(VIDEO_PATH)

    gradcam_video_path = os.path.join(
        OUTPUT_DIR, "gradcam_video.mp4"
    )
    save_gradcam_video(
        VIDEO_PATH,
        cam_maps,
        gradcam_video_path
    )

    print("   ✓ Grad-CAM video saved")

    activation_explainer = ActivationMapExplainer(ACTION_MODEL_PATH)
    activation_maps = activation_explainer.generate(VIDEO_PATH)

    activation_video_path = os.path.join(
        OUTPUT_DIR, "activation_video.mp4"
    )
    save_activation_video(
        VIDEO_PATH,
        activation_maps,
        activation_video_path
    )

    print("   ✓ Activation map video saved")

    # =========================================================
    # 4️⃣ CAPTIONING (GIT + BART FUSION)
    # =========================================================
    print("\n🔹 Generating Captions...")

    keyframes = select_keyframes(
        VIDEO_PATH,
        anomaly_scores,
        top_k=5
    )

    captioner = SceneCaptioner(device=DEVICE)

    # Step 1: Frame-level captions (GIT)
    frame_captions = captioner.caption_video_frames(keyframes)

    # Step 2: Fuse captions (BART)
    final_caption = captioner.build_final_caption(
        frame_captions,
        action_label=action_result["action"],
        action_confidence=action_result["confidence"]
    )

    # Step 3: Save captions
    save_captions(
        {
            "frame_captions": frame_captions,
            "final_caption": final_caption
        },
        os.path.join(OUTPUT_DIR, "captions.json")
    )

    print(f"   Caption: {final_caption}")

    # =========================================================
    # 5️⃣ FINAL REPORT
    # =========================================================
    print("\n🔹 Saving Final Report...")

    report = {
        "video": VIDEO_PATH,
        "action_recognition": action_result,
        "anomaly": {
            "max_score": float(anomaly_scores.max()),
            "mean_score": float(anomaly_scores.mean())
        },
        "caption": final_caption,
        "outputs": {
            "anomaly_scores": "output/anomaly_scores.npy",
            "gradcam_video": "output/gradcam_video.mp4",
            "activation_video": "output/activation_video.mp4"
        }
    }

    report_path = os.path.join(
        OUTPUT_DIR, "final_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("   ✓ Final report saved")
    print("\n✅ Pipeline completed successfully!\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    main()