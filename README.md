# Solar Panel Defect Detector

An AI-powered solar panel defect detection system built using deep learning and computer vision. The system classifies solar panel images into three categories and provides visual explanations using Grad-CAM heatmaps.

## Demo
> Live demo link will be added after deployment

## Classes
| Class | Description |
|---|---|
| Clean | No defects detected |
| Mild Defect | Minor defects, slight reduction in efficiency |
| Severe Defect | Major defects, significant damage detected |

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 78.03% |
| Weighted F1 Score | 0.78 |
| Clean F1 | 0.84 |
| Mild Defect F1 | 0.55 |
| Severe Defect F1 | 0.79 |

## Project Structure

## Model Architecture
- Base model: ResNet50 pretrained on ImageNet
- Transfer learning with unfrozen layer3 and layer4
- Custom classifier head: 2048 → 256 → 3 classes
- Dropout (0.4) for regularization

## Training Details
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 30 |
| Loss Function | CrossEntropyLoss with class weights |
| LR Scheduler | ReduceLROnPlateau (patience=5) |

## Dataset
- Source: ELPV Dataset
- Total images: 2624
- Train: 2098 images
- Val: 262 images  
- Test: 264 images

## Setup and Installation

### Clone the repository
```bash
git clone https://github.com/coderSubhadeepdeb/Solar-Defect-Detector.git
cd Solar-Defect-Detector
```

### Create conda environment
```bash
conda create -n solar-defect python=3.10
conda activate solar-defect
```

### Install dependencies
```bash
pip install torch torchvision pandas scikit-learn matplotlib pillow gradio tqdm grad-cam seaborn
```

### Download and organize dataset
```bash
git clone https://github.com/zae-bayern/elpv-dataset.git
python src/organize_dataset.py
```

### Train the model
```bash
python src/train.py
```

### Evaluate the model
```bash
python src/evaluate.py
```

### Run the web app
```bash
python app/app.py
```

## How It Works

1. User uploads a solar panel image
2. Image is resized to 224×224 and normalized
3. ResNet50 model classifies the image into one of three classes
4. Grad-CAM generates a heatmap showing which region the model focused on
5. Result is displayed with confidence score and heatmap overlay

## Grad-CAM Visualization
The system uses Gradient-weighted Class Activation Mapping (Grad-CAM) to explain model predictions. Red/warm regions indicate areas the model focused on when making its decision.

## Tech Stack
- PyTorch — deep learning framework
- torchvision — pretrained models and transforms
- Gradio — web interface
- pytorch-grad-cam — explainability
- scikit-learn — evaluation metrics
- matplotlib — visualization

## Team
Developed as part of internship project at [Your Institution Name]

## License
MIT License

