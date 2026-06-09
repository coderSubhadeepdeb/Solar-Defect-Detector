import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=3, device='cpu'):
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last ResNet layer (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final layer with our custom classifier
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")

    return model