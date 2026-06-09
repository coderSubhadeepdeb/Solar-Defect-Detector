import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from model import build_model

def evaluate():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    model_path = os.path.join(base, 'saved_models', 'best_model.pth')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, classes = get_dataloaders(data_dir)

    # Build model and load saved weights
    model = build_model(num_classes=len(classes), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\nModel loaded from {model_path}")

    # Run model on test set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Per class report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'confusion_matrix.png'))
    print("\nConfusion matrix saved as confusion_matrix.png")

if __name__ == '__main__':
    evaluate()