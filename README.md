# Face Classifier

Binary face detection using transfer learning with ResNet architectures implemented in Rust.

## Overview

This project implements a face vs. non face classifier using pretrained ResNet models (ResNet-18, ResNet-34, ResNet-50) with custom fully connected layers trained for binary classification. The models are trained in Python using PyTorch on Google Colab, and inference is performed in Rust using the tch-rs library.

## Features

- Three ResNet architectures with comparable performance (99.86% validation accuracy)
- Efficient inference in Rust with minimal memory footprint
- Command line interface for real time classification
- Transfer learning workflow: Python training, Rust deployment

## Dataset

Face images sourced from the CelebA dataset:
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Non-face images sourced from the Natural Images dataset (excluding person category):
https://www.kaggle.com/datasets/prasunroy/natural-images

Final balanced dataset: 4,674 images
- 70% training (3,271 images)
- 15% validation (701 images)
- 15% test (702 images)

## Models

### ResNet-18
- Backbone: 44 MB
- FC layer: 9 KB
- Validation accuracy: 99.86%

### ResNet-34
- Backbone: 82 MB
- FC layer: 9 KB
- Validation accuracy: 99.86%

### ResNet-50
- Backbone: 97 MB
- FC layer: 9 KB
- Validation accuracy: 99.86%

## Installation

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install LibTorch (macOS ARM64)
# Download from: https://pytorch.org/get-started/locally/
# Extract to ~/libtorch
```

### Setup

```bash
# Clone repository
git clone https://github.com/tanyabudhiraja/face_classifier.git
cd face_classifier

# Set environment variables (add to ~/.zshrc or ~/.bash_profile)
export LIBTORCH=$HOME/libtorch
export DYLD_LIBRARY_PATH=$HOME/libtorch/lib
```

## Usage

### ResNet-18

```bash
bash scripts/run_resnet18.sh path/to/image.jpg
```

### ResNet-34

```bash
bash scripts/run_resnet34.sh path/to/image.jpg
```

### ResNet-50

```bash
bash scripts/run_resnet50.sh path/to/image.jpg
```

### Example Output

```
Analyzing: photo.jpg

FACE DETECTED
Confidence: 99.82%
```

## Project Structure

```
face_classifier/
├── models/
│   ├── resnet18/
│   │   ├── resnet18.ot              # Pretrained backbone
│   │   └── fc_layer_resnet18_cpu.pt # Trained FC layer
│   ├── resnet34/
│   └── resnet50/
├── src/
│   └── bin/
│       ├── infer_resnet18.rs        # ResNet-18 CLI
│       ├── infer_resnet34.rs        # ResNet-34 CLI
│       └── infer_resnet50.rs        # ResNet-50 CLI
├── scripts/
│   ├── run_resnet18.sh
│   ├── run_resnet34.sh
│   └── run_resnet50.sh
└── training_scripts_for_colab/
    ├── face_classifier_training_resnet18.ipynb
    ├── face_classifier_training_resnet34.ipynb
    └── face_classifier_training_resnet50.ipynb
```

## Training

Models were trained in Google Colab using GPU acceleration. Training notebooks are available in `training_scripts_for_colab/`.

Training configuration:
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 10
- Training time: ~2 minutes per model

## Results

All three architectures achieved similar performance on this binary classification task, with validation accuracy near 99-100%. ResNet-18 provides the best balance between performance and computational efficiency.

