import torch.nn as nn
from torchvision import models


def build_model(num_classes, freeze_backbone=True):
    # EfficientNet-B0 preentrenado en ImageNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False   # no se reentrena la parte convolucional

    # Sustituimos la última capa por una nuestra, con num_classes salidas
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model