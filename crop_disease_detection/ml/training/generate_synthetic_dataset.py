from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.class_mapping import MODEL_CLASS_NAMES

CLASS_NAMES = list(MODEL_CLASS_NAMES)

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_real_dataset(dataset_root: Path = DATASET_ROOT) -> dict[str, int]:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}. Download a real labeled crop-disease dataset "
            "such as PlantVillage and place it under ml/dataset using class folders."
        )

    class_counts: dict[str, int] = {}
    missing_classes: list[str] = []

    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue

        images = [
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        class_counts[class_name] = len(images)

    if not any(class_counts.values()):
        raise FileNotFoundError(
            "No labeled image files were found in the dataset directory. Replace the synthetic placeholder data with a real dataset "
            "organized as ml/dataset/<ClassName>/<image>.jpg."
        )

    if missing_classes:
        raise FileNotFoundError(
            "Required PlantVillage class folders are missing: "
            f"{', '.join(missing_classes)}. Run Download/Download_dataset.py first."
        )

    for class_name, count in class_counts.items():
        if count == 0:
            raise FileNotFoundError(f"No valid images found in required class folder: {dataset_root / class_name}")
        else:
            print(f"Validated {count} images for {class_name}")

    return class_counts


if __name__ == "__main__":
    validate_real_dataset()
    print("\nExpected real dataset structure:")
    print("ml/dataset/")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}/")
    print("\nUse a real labeled crop-disease dataset such as PlantVillage, then rerun the training script.")
