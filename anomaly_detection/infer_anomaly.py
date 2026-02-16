# anomaly_detection/infer_anomaly.py

import torch
import numpy as np
from .learner import Learner


def pad_features(features, target_dim=2048):
    """
    Deterministically expand feature dimension using zero padding.
    """
    current_dim = features.size(1)

    if current_dim >= target_dim:
        return features[:, :target_dim]

    pad_size = target_dim - current_dim
    padding = torch.zeros(features.size(0), pad_size, device=features.device)

    return torch.cat([features, padding], dim=1)


class AnomalyInferencer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        self.model = Learner(input_dim=2048).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # Legacy vars-based checkpoint
        if any(k.startswith('vars.') for k in state_dict.keys()):
            print("🔧 Remapping anomaly checkpoint parameters (vars → classifier)...")

            legacy_keys = sorted(
                [k for k in state_dict.keys() if k.startswith('vars.')],
                key=lambda x: int(x.split('.')[1])
            )

            legacy_weights = [state_dict[k] for k in legacy_keys]
            classifier_params = list(self.model.classifier.parameters())

            with torch.no_grad():
                for p, w in zip(classifier_params, legacy_weights):
                    p.copy_(w)
        else:
            self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        print("✅ Anomaly model loaded successfully")

    @torch.no_grad()
    def infer(self, features, total_frames):
        """
        features: Tensor (32, feature_dim)
        """
        features = features.to(self.device)

        # 🔑 Deterministic feature expansion
        features = pad_features(features, 2048)

        scores = self.model(features).squeeze(-1).cpu().numpy()

        # Expand segment scores → frame-level
        frame_scores = np.zeros(total_frames, dtype=np.float32)
        steps = np.round(np.linspace(0, total_frames, 33)).astype(int)

        for i in range(32):
            frame_scores[steps[i]:steps[i + 1]] = scores[i]

        return frame_scores


def save_anomaly_scores(frame_scores, out_path):
    np.save(out_path, frame_scores)
    