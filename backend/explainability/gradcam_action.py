# explainability/gradcam_action.py

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from action_recognition.cnn3d_model import generate_model
from action_recognition.infer_action import load_clip

# ---------------- CONFIG ----------------
IMG_SIZE = 112
FRAMES = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


class ActionGradCAM:
    def __init__(self, model_path):
        self.model = generate_model(num_classes=101)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

        self.activations = None
        self.gradients = None

        # 🔥 Last Conv3D layer in YOUR CNN3D
        target_layer = self.model.features[12]

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out          # (B, C, T, H, W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]    # (B, C, T, H, W)

    def generate(self, video_path):
        """
        Returns:
        cam_maps: (T, H, W) numpy array in [0,1]
        class_idx: predicted class
        """
        clip = load_clip(video_path).to(DEVICE)  # (1, 3, T, H, W)

        logits = self.model(clip)
        class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        # ---- Grad-CAM computation ----
        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)  # (B, T, H, W)
        cam = F.relu(cam)

        # Normalize safely
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam[0].detach().cpu().numpy()  # (T, H, W)
        return cam, class_idx


def save_gradcam_video(video_path, cam_maps, out_path):
    """
    Saves Grad-CAM overlay video.
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

    # 🔥 Sample same number of frames as CAMs
    idxs = np.linspace(0, len(frames) - 1, len(cam_maps)).astype(int)
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

    for frame, cam in zip(frames, cam_maps):
        # 1️⃣ Resize CAM to frame size
        cam_resized = cv2.resize(cam, (width, height))

        # 2️⃣ Convert to heatmap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # 3️⃣ Ensure correct dtype
        frame = frame.astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)

        # 4️⃣ Overlay
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        overlay = np.uint8(overlay)

        writer.write(overlay)

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
    
    print("Frames read from input:", len(frames))
    print("CAM maps length:", len(cam_maps))

