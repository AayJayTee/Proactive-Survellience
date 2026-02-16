# anomaly_detection/feature_extractor.py

import torch
import numpy as np
import cv2
from torchvision import transforms

# ---------------------------------------------------
# Simple uniform temporal sampling
# ---------------------------------------------------
def uniform_sample_frames(video_path, num_segments=32, frames_per_segment=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segment_length = total_frames // num_segments
    sampled_frames = []

    for i in range(num_segments):
        start = i * segment_length
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        segment_frames = []
        for _ in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames.append(frame)

        if len(segment_frames) == 0:
            segment_frames = [np.zeros((224, 224, 3), dtype=np.uint8)]

        sampled_frames.append(segment_frames)

    cap.release()
    return sampled_frames


# ---------------------------------------------------
# Feature extractor wrapper
# ---------------------------------------------------
class VideoFeatureExtractor:
    """
    Produces (32, 2048) feature tensor per video
    """
    def __init__(self, backbone, device='cuda'):
        self.backbone = backbone.to(device)
        self.backbone.eval()
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                 std=[0.225, 0.225, 0.225])
        ])

    @torch.no_grad()
    def extract(self, video_path):
        segments = uniform_sample_frames(video_path)
        features = []

        for frames in segments:
            clip = []
            for f in frames:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = self.transform(f)
                clip.append(f)

            clip = torch.stack(clip, dim=1)  # (3, T, H, W)
            clip = clip.unsqueeze(0).to(self.device)

            # ✅ IMPORTANT FIX
            _, feat = self.backbone(clip, return_features=True)

            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            features.append(feat.squeeze(0).cpu())

        features = torch.stack(features)  # (32, 256)
        return features