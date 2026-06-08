import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=3, device='cpu'):
    # Load pretrained ResNet50
    # pretrained=True means it downloads weights trained on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all layers — we don't want to change what ResNet already learned
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer with our own
    # ResNet50's final layer outputs 1000 classes (ImageNet)
    # We replace it with one that outputs 3 classes (our defect types)
    in_features = model.fc.in_features  # this is 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")

    return model