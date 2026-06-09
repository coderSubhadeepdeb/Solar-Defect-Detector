import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model

# Setup
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base, 'saved_models', 'best_model.pth')
classes = ['clean', 'mild_defect', 'severe_defect']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model once when app starts
model = build_model(num_classes=3, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    # Preprocess
    img = image.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    # Grad-CAM
    target_layer = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    heatmap = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # Build confidence scores for all classes
    scores = {classes[i]: float(probs[i]) for i in range(len(classes))}

    return (
        Image.fromarray(heatmap),
        scores,
        f"Prediction: {classes[pred_idx].replace('_', ' ').title()} ({confidence*100:.1f}% confidence)"
    )

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload Solar Panel Image'),
    outputs=[
        gr.Image(label='Grad-CAM Heatmap'),
        gr.Label(label='Class Probabilities'),
        gr.Textbox(label='Result')
    ],
    title='Solar Panel Defect Detector',
    description='Upload a solar panel image to detect defects. The model will classify it as Clean, Mild Defect, or Severe Defect and show what region it focused on.',
    examples=[
        [os.path.join(base, 'data', 'test', 'clean', os.listdir(os.path.join(base, 'data', 'test', 'clean'))[0])],
        [os.path.join(base, 'data', 'test', 'severe_defect', os.listdir(os.path.join(base, 'data', 'test', 'severe_defect'))[0])]
    ]
)

if __name__ == '__main__':
    demo.launch()