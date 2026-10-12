from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "ml" / "validation_dataset"
MODEL_PATH = PROJECT_ROOT / "model" / "crop_disease_model.pth"
sys.path.insert(0, str(PROJECT_ROOT))
from ml.training.class_mapping import SOURCE_TO_CLASS


class IndependentValidationDataset(Dataset):
    """View only supported crop classes from Kaggle's separate validation split."""

    def __init__(self, data_dir: Path, class_names: list[str], transform):
        self.image_folder = datasets.ImageFolder(str(data_dir))
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        source_to_model_index = {}
        for source_name, target_name in SOURCE_TO_CLASS.items():
            if target_name in self.class_to_idx:
                target_index = self.class_to_idx[target_name]
                source_to_model_index[source_name] = target_index
                source_to_model_index[target_name] = target_index
        self.samples = [
            (path, source_to_model_index[source_name])
            for path, source_index in self.image_folder.samples
            if (source_name := self.image_folder.classes[source_index]) in source_to_model_index
        ]

        counts = {name: 0 for name in class_names}
        for _, label in self.samples:
            counts[class_names[label]] += 1
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise ValueError(f"Independent validation split is missing checkpoint classes: {missing}")
        print(f"Independent validation examples per class: {counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.image_folder.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on Kaggle's independent validation split.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATASET_DIR,
        help="Kaggle `valid/` split prepared by Download/Download_dataset.py.",
    )
    args = parser.parse_args()
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Trained checkpoint not found: {MODEL_PATH}. Run train_model.py first.")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Independent validation folder not found: {args.data_dir}. "
            "Run Download/Download_dataset.py to prepare Kaggle's separate valid/ split."
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = checkpoint.get("class_names", [])
    if not classes or checkpoint.get("num_classes") != len(classes):
        raise ValueError(f"Checkpoint has invalid class metadata: {MODEL_PATH}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = IndependentValidationDataset(args.data_dir, classes, transform)

    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.models.crop_disease_model import build_crop_disease_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crop_disease_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, batch_labels in loader:
            all_preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
            all_labels.extend(batch_labels.tolist())

    print(f"Independent validation accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds, labels=range(len(classes)), target_names=classes, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds, labels=range(len(classes))))


if __name__ == "__main__":
    main()
