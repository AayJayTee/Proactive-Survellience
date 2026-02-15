# explainability/activation_maps.py

import cv2
import torch
import numpy as np
import os

from action_recognition.cnn3d_model import generate_model
from action_recognition.infer_action import load_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationMapExplainer:
    """
    Temporal activation-based explainability.
    Measures feature energy over time.
    """

    def __init__(self, model_path):
        self.model = generate_model(num_classes=101)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def generate(self, video_path):
        """
        Returns:
        importance: (T,) numpy array in [0,1]
        """

        clip = load_clip(video_path).to(DEVICE)  # (1, 3, T, H, W)

        # Get features
        _, feats = self.model(clip, return_features=True)  # (1, 256)

        # Feature energy → temporal importance
        T = clip.shape[2]
        importance = torch.norm(feats, p=2, dim=1)  # (1,)
        importance = importance.repeat(T)           # (T,)

        importance = importance.cpu().numpy()
        importance = importance / (importance.max() + 1e-8)

        return importance


def save_activation_video(video_path, importance, out_path):
    """
    Saves video with temporal activation bar overlay.
    Uses AVI + ffmpeg for browser-compatible MP4.
    """

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        print("⚠️ FPS invalid. Using default 20.")
        fps = 20

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # ✅ FIX: Get actual dimensions from first frame
    if len(frames) == 0:
        print("❌ No frames read from video!")
        return

    height, width = frames[0].shape[:2]
    print(f"✅ Detected video dimensions: {width}x{height}")

    # Match importance length to frames
    idxs = np.linspace(0, len(frames) - 1, len(importance)).astype(int)
    frames = [frames[i] for i in idxs]

    # ✅ Write to temporary AVI first (more reliable)
    temp_path = out_path.replace(".mp4", ".avi")
    writer = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        int(fps),
        (width, height)
    )

    print(f"Writer opened: {writer.isOpened()}")
    print(f"FPS value passed: {int(fps)}")
    print(f"Temp output path: {temp_path}")
    
    if not writer.isOpened():
        print(f"❌ VideoWriter failed to open! Check codec or permissions.")
        return

    bar_height = 20

    for frame, score in zip(frames, importance):
        frame = frame.copy()

        # Draw activation bar
        bar_width = int(score * width)
        cv2.rectangle(
            frame,
            (0, height - bar_height),
            (bar_width, height),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Activation: {score:.2f}",
            (10, height - bar_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        frame = np.uint8(frame)
        writer.write(frame)

    writer.release()
    print(f"✅ AVI temp file saved at: {temp_path}")
    
    # ✅ CONVERT TO BROWSER-COMPATIBLE MP4 using ffmpeg
    print(f"🔄 Converting AVI to MP4 with H.264 codec...")
    import subprocess
    
    try:
        cmd = f'ffmpeg -y -i "{temp_path}" -vcodec libx264 -pix_fmt yuv420p "{out_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ MP4 conversion successful!")
            print(f"✅ Final video saved at: {out_path}")
            os.remove(temp_path)
        else:
            print(f"❌ ffmpeg conversion failed: {result.stderr}")
            print(f"⚠️ Keeping temp AVI at: {temp_path}")
    except Exception as e:
        print(f"❌ Error during ffmpeg conversion: {e}")
        print(f"⚠️ Keeping temp AVI at: {temp_path}")
