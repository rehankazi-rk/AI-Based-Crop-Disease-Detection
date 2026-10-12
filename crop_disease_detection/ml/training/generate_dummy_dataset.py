import os
from pathlib import Path
from PIL import Image
import random

# Classes must match CLASS_NAMES in generate_synthetic_dataset.py
CLASS_NAMES = [
    "Healthy_Tomato",
    "Early_Blight_Tomato",
    "Late_Blight_Tomato",
    "Leaf_Mold_Tomato",
    "Healthy_Potato",
    "Early_Blight_Potato",
    "Late_Blight_Potato",
]

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

def create_class_folder(class_name: str, count: int = 5) -> None:
    class_dir = DATASET_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (224, 224), (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ))
        img_path = class_dir / f"img_{i+1}.jpg"
        img.save(img_path, "JPEG")
        print(f"Created {img_path}")

def main():
    if DATASET_DIR.exists():
        print(f"Dataset directory {DATASET_DIR} already exists. Removing it to recreate dummy data.")
        import shutil
        shutil.rmtree(DATASET_DIR)
    for class_name in CLASS_NAMES:
        create_class_folder(class_name, count=5)
    print("Dummy dataset created at", DATASET_DIR)

if __name__ == "__main__":
    main()

