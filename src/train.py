import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from model import build_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm wraps your loader and shows a progress bar
    loop = tqdm(loader, leave=True)

    for images, labels in loop:
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Step 1: Forward pass
        outputs = model(images)

        # Step 2: Calculate loss
        loss = criterion(outputs, labels)

        # Step 3: Zero old gradients
        optimizer.zero_grad()

        # Step 4: Backward pass
        loss.backward()

        # Step 5: Update weights
        optimizer.step()

        # Track accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        # Update progress bar
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(num_epochs=10, batch_size=32, learning_rate=0.001):
    # Setup
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    save_dir = os.path.join(base, 'saved_models')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get data
    train_loader, val_loader, _, classes = get_dataloaders(data_dir, batch_size)

    # Build model
    model = build_model(num_classes=len(classes), device=device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track best model
    best_val_acc = 0.0

    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("-" * 50)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Best model saved with val accuracy: {val_acc:.2f}%\n")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    train()