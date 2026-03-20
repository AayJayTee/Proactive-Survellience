import cv2
import numpy as np
import torch

from .autoencoder import ConvolutionalAutoencoder


class AnomalyInferencer:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        frame_size=64,
        batch_size=64,
        latent_dim=256,
        threshold_override=None,
        default_threshold=0.005069
    ):
        self.device = torch.device(device)
        self.frame_size = int(frame_size)
        self.batch_size = int(batch_size)
        self.latent_dim = int(latent_dim)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = ConvolutionalAutoencoder(
            input_channels=1,
            latent_dim=self.latent_dim
        ).to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported anomaly checkpoint format")

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        if threshold_override is not None:
            self.threshold = float(threshold_override)
        elif isinstance(checkpoint, dict) and checkpoint.get("threshold") is not None:
            self.threshold = float(checkpoint["threshold"])
        else:
            self.threshold = float(default_threshold)

        print("Anomaly autoencoder loaded successfully")

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.frame_size, self.frame_size))
                normalized = resized.astype(np.float32) / 255.0
                frames.append(normalized)
        finally:
            cap.release()

        if not frames:
            raise RuntimeError("No frames were extracted from video")

        return np.stack(frames, axis=0)

    @torch.no_grad()
    def infer(self, video_path, total_frames=None):
        frames = self._load_video_frames(video_path)
        frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(self.device)

        scores = []
        for i in range(0, len(frames_tensor), self.batch_size):
            batch = frames_tensor[i:i + self.batch_size]
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    reconstructed = self.model(batch)
            else:
                reconstructed = self.model(batch)

            errors = ((batch - reconstructed) ** 2).mean(dim=[1, 2, 3])
            scores.extend(errors.detach().cpu().numpy().tolist())

        frame_scores = np.asarray(scores, dtype=np.float32)

        if total_frames is not None and total_frames > 0 and len(frame_scores) != total_frames:
            if len(frame_scores) > total_frames:
                frame_scores = frame_scores[:total_frames]
            else:
                pad_value = float(frame_scores[-1]) if len(frame_scores) else 0.0
                pad = np.full((total_frames - len(frame_scores),), pad_value, dtype=np.float32)
                frame_scores = np.concatenate([frame_scores, pad], axis=0)

        return frame_scores


def save_anomaly_scores(frame_scores, out_path):
    np.save(out_path, frame_scores)