from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_crop_disease_model(num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """Return the exact transfer-learning model architecture used for training and inference."""
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return backbone


class CropDiseaseModel(nn.Module):
    """Transfer-learning CNN built on EfficientNet-B0 with a disease classification head."""

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        self.model = build_crop_disease_model(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
