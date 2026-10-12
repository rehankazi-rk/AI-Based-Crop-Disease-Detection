from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.models.crop_disease_model import build_crop_disease_model
from ml.training.generate_synthetic_dataset import validate_real_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "ml" / "dataset"
MODEL_OUTPUT = PROJECT_ROOT / "model" / "crop_disease_model.pth"
SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transforms(training: bool):
    operations = [transforms.Resize((224, 224))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    return transforms.Compose(operations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune an EfficientNet-B0 on PlantVillage images.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16 if not torch.cuda.is_available() else 32)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet initialization weights.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    counts = validate_real_dataset(DATASET_DIR)
    too_small = [name for name, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(f"Need at least two real images per class for a stratified split: {too_small}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    training_data = datasets.ImageFolder(str(DATASET_DIR), transform=make_transforms(training=True))
    validation_data = datasets.ImageFolder(str(DATASET_DIR), transform=make_transforms(training=False))
    if training_data.classes != validation_data.classes:
        raise RuntimeError("Training and validation class mappings differ")
    labels = [label for _, label in training_data.samples]
    train_indices, validation_indices = train_test_split(
        list(range(len(labels))),
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; {len(train_indices):,} train and {len(validation_indices):,} validation images")
    model = build_crop_disease_model(
        num_classes=len(training_data.classes), pretrained=not args.no_pretrained
    ).to(device)

    if args.no_pretrained:
        print("Training all layers from scratch; consider increasing --epochs for useful accuracy.")
    else:
        # Retain pretrained low-level features while adapting the final feature stage and head.
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    train_loader = DataLoader(
        Subset(training_data, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        validation_count = 0
        with torch.inference_mode():
            for images, targets in validation_loader:
                outputs = model(images.to(device))
                targets = targets.to(device)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                validation_count += targets.size(0)
        validation_accuracy = correct / validation_count
        average_loss = total_loss / len(train_indices)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f} "
            f"- validation accuracy: {validation_accuracy:.4f}",
            flush=True,
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            payload = {
                "state_dict": model.cpu().state_dict(),
                "class_names": training_data.classes,
                "num_classes": len(training_data.classes),
                "input_size": 224,
                "normalization": {"mean": MEAN, "std": STD},
                "seed": SEED,
                "validation_accuracy": best_accuracy,
                "epoch": epoch + 1,
            }
            MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = MODEL_OUTPUT.with_suffix(MODEL_OUTPUT.suffix + ".tmp")
            torch.save(payload, temporary_output)
            os.replace(temporary_output, MODEL_OUTPUT)
            model.to(device)

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Saved best checkpoint to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
