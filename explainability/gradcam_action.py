# explainability/gradcam_action.py

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from action_recognition.cnn3d_model import generate_model
from action_recognition.infer_action import load_clip

# ---------------- CONFIG ----------------
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

        # ✅ Hook AFTER last ReLU (better than raw conv)
        target_layer = self.model.features[14]

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # (B, C, T, H, W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # (B, C, T, H, W)

    def generate(self, video_path):
        """
        Returns:
        cam_maps: (T, H, W)
        class_idx
        """

        clip = load_clip(video_path).to(DEVICE)

        logits = self.model(clip)
        class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        # 🔥 Improved Grad-CAM computation

        # Preserve temporal dimension
        weights = self.gradients.mean(dim=(3, 4), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)  # (B, T, H, W)
        cam = F.relu(cam)

        cam = cam[0]  # remove batch

        # Normalize per-frame
        cam_maps = []
        for t in range(cam.shape[0]):
            frame_cam = cam[t]
            frame_cam -= frame_cam.min()
            frame_cam /= (frame_cam.max() + 1e-8)
            cam_maps.append(frame_cam.cpu().detach().numpy())

        cam_maps = np.array(cam_maps)

        return cam_maps, class_idx


def save_gradcam_video(video_path, cam_maps, out_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 20

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return

    height, width = frames[0].shape[:2]

    # Sample frames to match CAM length
    idxs = np.linspace(0, len(frames) - 1, len(cam_maps)).astype(int)
    frames = [frames[i] for i in idxs]

    temp_path = out_path.replace(".mp4", ".avi")

    writer = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        5,
        (width, height)
    )

    for frame, cam in zip(frames, cam_maps):

        cam_resized = cv2.resize(cam, (width, height))
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        writer.write(overlay)

    writer.release()

    # Convert to MP4
    import subprocess

    cmd = f'ffmpeg -y -i "{temp_path}" -vcodec libx264 -pix_fmt yuv420p "{out_path}"'
    subprocess.run(cmd, shell=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)