import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_model

# Detect if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Build model
model = build_model(num_classes=3, device=device)

# Test with a fake image tensor
# Simulates one image: batch=1, channels=3, height=224, width=224
fake_image = torch.randn(1, 3, 224, 224).to(device)
output = model(fake_image)

print(f"\nInput shape: {fake_image.shape}")
print(f"Output shape: {output.shape}")
print(f"Raw output scores: {output}")