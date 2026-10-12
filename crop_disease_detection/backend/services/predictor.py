from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from backend.models.crop_disease_model import CropDiseaseModel

MODEL_CLASSES = [
    "Healthy_Tomato",
    "Early_Blight_Tomato",
    "Late_Blight_Tomato",
    "Leaf_Mold_Tomato",
    "Healthy_Potato",
    "Early_Blight_Potato",
    "Late_Blight_Potato",
]

MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "crop_disease_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model_checkpoint() -> None:
    """Ensure the checkpoint exists; raise a clear error if not."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Model checkpoint not found or empty at {MODEL_PATH}. Train the model first.")


class TransferLearningPredictor:
    """Transfer-learning CNN inference pipeline for crop disease classification."""

    def __init__(self):
        ensure_model_checkpoint()
        self.class_names = MODEL_CLASSES
        self.model = CropDiseaseModel(num_classes=len(MODEL_CLASSES)).to(DEVICE)
        self.model.eval()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint at {MODEL_PATH} is not a valid dict-based PyTorch state. Expected a dict with 'state_dict'.")

        if "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint is missing 'state_dict' at {MODEL_PATH}.")

        state_dict = checkpoint["state_dict"]
        class_names = checkpoint.get("class_names", self.class_names)
        expected_num_classes = int(checkpoint.get("num_classes", len(class_names)))
        self.class_names = list(class_names)

        if (
            len(self.class_names) != expected_num_classes
            or expected_num_classes != len(MODEL_CLASSES)
            or set(self.class_names) != set(MODEL_CLASSES)
        ):
            raise ValueError(
                "Checkpoint class metadata does not match the seven supported tomato/potato classes. "
                f"Found {self.class_names!r} with num_classes={expected_num_classes}."
            )

        if not isinstance(state_dict, dict) or not state_dict:
            raise TypeError(f"Checkpoint state_dict at {MODEL_PATH} must be a non-empty mapping of tensors.")

        cleaned = {}
        for key, value in state_dict.items():
            normalized = key.replace("module.", "").replace("model.", "")
            if normalized.startswith("model."):
                normalized = normalized.replace("model.", "", 1)
            cleaned[normalized] = value

        # Training saves the unwrapped EfficientNet backbone; inference wraps it in CropDiseaseModel.
        current_state = self.model.state_dict()
        for key in list(current_state.keys()):
            if key.startswith("model.") and key.replace("model.", "", 1) in cleaned:
                cleaned[key] = cleaned.pop(key.replace("model.", "", 1))

        mismatched = set(current_state) ^ set(cleaned)
        if mismatched:
            raise RuntimeError(
                f"State dict mismatch while loading {MODEL_PATH}. "
                f"Architecture mismatch: {sorted(list(mismatched))[:15]}"
            )

        self.model.load_state_dict(cleaned, strict=True)

    def predict(self, image: Image.Image) -> Dict[str, object]:
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        rgb_image = image.convert("RGB")
        tensor = preprocess(rgb_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        label = self.class_names[int(predicted_index.item())]
        crop, disease = self._parse_label(label)
        return {
            "crop": crop,
            "disease": disease,
            "confidence": float(confidence.item()),
            "label": label,
        }

    @staticmethod
    def _parse_label(label: str) -> Tuple[str, str]:
        if "Tomato" in label:
            crop = "Tomato"
        elif "Potato" in label:
            crop = "Potato"
        else:
            crop = "Unknown"

        if "Healthy" in label:
            disease = "Healthy"
        elif "Early" in label:
            disease = "Early Blight"
        elif "Late" in label:
            disease = "Late Blight"
        elif "Leaf" in label:
            disease = "Leaf Mold"
        else:
            disease = "Unknown"

        return crop, disease


@lru_cache(maxsize=1)
def get_predictor() -> TransferLearningPredictor:
    return TransferLearningPredictor()
