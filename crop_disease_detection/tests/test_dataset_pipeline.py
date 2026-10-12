from __future__ import annotations

from pathlib import Path

from PIL import Image

from Download.Download_dataset import prepare_training_dataset, prepare_validation_dataset
from ml.training.class_mapping import MODEL_CLASS_NAMES, SOURCE_TO_CLASS
from ml.training.evaluate_model import IndependentValidationDataset


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), color=color).save(path)


def test_downloader_prepares_train_and_independent_validation_splits(tmp_path):
    source_root = tmp_path / "kaggle"
    train_output = tmp_path / "train-output"
    valid_output = tmp_path / "valid-output"
    for index, source_class in enumerate(SOURCE_TO_CLASS):
        make_image(source_root / "train" / source_class / "train.jpg", (index, 20, 30))
        make_image(source_root / "valid" / source_class / "valid.jpg", (index, 80, 90))

    train_counts = prepare_training_dataset(source_root, train_output)
    valid_counts = prepare_validation_dataset(source_root, valid_output)

    assert set(train_counts) == set(MODEL_CLASS_NAMES)
    assert train_counts == valid_counts
    assert (train_output / "Healthy_Tomato" / "train.jpg").is_file()
    assert (valid_output / "Healthy_Tomato" / "valid.jpg").is_file()
    assert not (valid_output / "Healthy_Tomato" / "train.jpg").exists()


def test_independent_evaluator_maps_only_supported_validation_labels(tmp_path):
    validation_root = tmp_path / "valid"
    for index, source_class in enumerate(SOURCE_TO_CLASS):
        make_image(validation_root / source_class / "leaf.jpg", (index, 100, 120))
    make_image(validation_root / "Apple___healthy" / "ignored.jpg", (255, 0, 0))

    dataset = IndependentValidationDataset(validation_root, list(MODEL_CLASS_NAMES), transform=None)
    assert len(dataset) == len(SOURCE_TO_CLASS)
    assert dataset.class_names == list(MODEL_CLASS_NAMES)
    _, label = dataset[0]
    assert 0 <= label < len(MODEL_CLASS_NAMES)
