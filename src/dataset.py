import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    # Training transforms — includes augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation and test transforms — no augmentation, just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=32):
    train_transform, val_transform = get_transforms()

    # ImageFolder reads your folder structure automatically
    # folder name = class label
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=val_transform
    )

    # DataLoader wraps dataset — handles batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader, train_dataset.classes