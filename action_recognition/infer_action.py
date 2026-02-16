# action_recognition/infer_action.py

import cv2
import torch
import numpy as np

from action_recognition.cnn3d_model import generate_model

# ---------------- CONFIG ----------------
IMG_SIZE = 112
FRAMES = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


# Minimal UCF101 class list (extend fully once)
UCF101_CLASSES = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "Hammering",
    "HammerThrow",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo"
]

def load_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < FRAMES:
        raise RuntimeError("Video too short for action recognition")

    idxs = np.linspace(0, len(frames) - 1, FRAMES).astype(int)
    clip = np.array([frames[i] for i in idxs], dtype=np.float32) / 255.0

    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
    return clip.unsqueeze(0)  # (1, 3, T, H, W)


def run_action_recognition(video_path, model_path, return_features=False):
    model = generate_model(num_classes=101)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    clip = load_clip(video_path).to(DEVICE)

    with torch.no_grad():
        if return_features:
            feats = model(clip, return_features=True)
            return feats.squeeze(0)  # (256,)

        logits = model(clip)
        probs = torch.softmax(logits, dim=1)

    class_id = probs.argmax(dim=1).item()
    confidence = probs.max(dim=1).values.item()

    return {
        "class_id": class_id,
        "action": UCF101_CLASSES[class_id],
        "confidence": confidence
    }
