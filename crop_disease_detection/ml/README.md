# Crop disease model pipeline

From the project root, prepare the seven supported tomato/potato classes from the
Kaggle New Plant Diseases Dataset:

```powershell
python Download/Download_dataset.py
```

The downloader writes the Kaggle `train/` split into `ml/dataset/` and Kaggle's
separate `valid/` split into `ml/validation_dataset/`. Both local data directories
are ignored by Git.

Train EfficientNet-B0 with ImageNet initialization, light augmentation, a
stratified 80/20 split, and validation-based best-checkpoint selection:

```powershell
python ml/training/train_model.py --epochs 5 --batch-size 16
```

Use `--no-pretrained` only when ImageNet weights cannot be downloaded. CPU-only
machines may take significantly longer; use a smaller batch size if memory is
limited. Evaluate the resulting checkpoint on the independent Kaggle validation
images with:

```powershell
python ml/training/evaluate_model.py
```

The evaluator requires all seven validation classes and reports class-wise
precision/recall, accuracy, and a confusion matrix. It does not create a random
holdout from the training images. The checkpoint is saved to
`model/crop_disease_model.pth` and loaded once per API process.
