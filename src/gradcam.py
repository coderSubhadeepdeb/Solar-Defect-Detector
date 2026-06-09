import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_model

# Class names
classes = ['clean', 'mild_defect', 'severe_defect']

def load_model(model_path, device):
    model = build_model(num_classes=3, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    tensor = transform(img).unsqueeze(0)
    return tensor, img_array

def generate_gradcam(image_path, model, device):
    tensor, img_array = preprocess_image(image_path)
    tensor = tensor.to(device)

    # Target layer — last conv layer in ResNet50
    target_layer = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        # Get model prediction first
        with torch.no_grad():
            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1)[0]
            confidence = probs[pred_idx].item()

        # Generate CAM for predicted class
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    return visualization, classes[pred_idx], confidence

def run_gradcam_on_samples(num_samples=2):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base, 'saved_models', 'best_model.pth')
    data_dir = os.path.join(base, 'data', 'test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_model(model_path, device)

    # 3 classes x 2 samples = 6 images
    # Each image gets 2 columns: original + heatmap
    fig, axes = plt.subplots(6, 2, figsize=(10, 18))

    row = 0
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)[:2]

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)

            visualization, pred_class, confidence = generate_gradcam(
                img_path, model, device
            )

            # Original image on left
            original = Image.open(img_path).convert('RGB').resize((224, 224))
            axes[row][0].imshow(original)
            axes[row][0].set_title(f'Actual: {class_name}', fontsize=9)
            axes[row][0].axis('off')

            # Grad-CAM heatmap on right
            axes[row][1].imshow(visualization)
            axes[row][1].set_title(
                f'Pred: {pred_class} ({confidence*100:.1f}%)',
                fontsize=9
            )
            axes[row][1].axis('off')

            row += 1

    plt.suptitle('Grad-CAM: What the model looks at', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(base, 'gradcam_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGrad-CAM results saved as gradcam_results.png")

if __name__ == '__main__':
    run_gradcam_on_samples()