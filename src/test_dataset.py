import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add src to path so we can import dataset.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders

# Path to your data folder
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

# Load data
train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir)

# Grab one batch
images, labels = next(iter(train_loader))

print(f"\nOne batch contains:")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label values in this batch: {labels[:8]}")
print(f"Which means: {[classes[l] for l in labels[:8]]}")

# Visualize 8 images from the batch
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(8):
    img = images[i].numpy().transpose(1, 2, 0)
    # Unnormalize so image looks correct
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    axes[i].imshow(img)
    axes[i].set_title(classes[labels[i]])
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base, 'sample_batch.png'))
print("\nSample batch image saved as sample_batch.png")