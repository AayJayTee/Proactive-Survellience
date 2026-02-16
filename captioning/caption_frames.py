import torch
import cv2
import json
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class SceneCaptioner:

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # ===============================
        # 1️⃣ GIT Caption Model
        # ===============================
        print("Loading GIT-large captioning model...")

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/git-large-coco"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-large-coco"
        ).to(self.device)

        self.model.eval()

        print("✓ GIT model loaded")

        # ===============================
        # 2️⃣ PEGASUS Summarizer
        # ===============================
        print("Loading PEGASUS summarization model...")

        self.sum_tokenizer = AutoTokenizer.from_pretrained(
            "google/pegasus-xsum"
        )

        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/pegasus-xsum"
        ).to(self.device)

        self.sum_model.eval()

        print("✓ PEGASUS model loaded")

    # =========================================================
    # Frame Captioning
    # =========================================================
    @torch.no_grad()
    def caption_video_frames(self, frames):

        captions = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                max_length=40
            )

            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            captions.append(caption.strip())

        return captions

    # =========================================================
    # Proper Summarization
    # =========================================================
    @torch.no_grad()
    def build_final_caption(
        self,
        frame_captions,
        action_label=None,
        action_confidence=None
    ):

        if not frame_captions:
            return "No visual description available."

        # Remove duplicates
        unique = list(dict.fromkeys(frame_captions))

        text_block = " ".join(unique)

        inputs = self.sum_tokenizer(
            text_block,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        summary_ids = self.sum_model.generate(
            **inputs,
            max_length=60,
            num_beams=4,
            early_stopping=True
        )

        summary = self.sum_tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        if action_label:
            summary += (
                f" The recognized activity is {action_label} "
                f"(confidence {action_confidence:.2f})."
            )

        return summary.strip()


def save_captions(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Captions saved at: {path}")
