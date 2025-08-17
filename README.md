# ECG Pretrained Models: TemporalNet & ModelGroup_TemporalNet

## Overview
This repository provides pretrained deep learning models for clinical ECG classification. The models are designed for robust feature extraction and classification from multi-lead ECG signals, supporting both single-model and grouped-lead architectures. Pretrained weights are provided for reproducibility and further research.

**Note:** The models in this repository are designed for 8-lead ECGs, specifically using the following leads: I, II, V1, V2, V3, V4, V5, and V6.

## Checkpoints
Pretrained model weights are available in the `checkpoints/` directory:
- `temporalnet.pth.tar`: Weights for the single-model TemporalNet architecture.
- `lead_groupings_temporalnet.pth.tar`: Weights for the grouped-lead ModelGroup_TemporalNet architecture.

## Model Architectures
- **ECG_TemporalNet1D**: A 1D convolutional neural network for ECG signal analysis, supporting both feature extraction and classification.
- **ModelGroup**: Combines two ECG_TemporalNet1D models to process grouped ECG leads, enabling more flexible and robust analysis.

## Usage
The models can be loaded in three modes: **baseline** (random initialization), **pretrained** (for inference), and **finetune** (for transfer learning). Example usage is provided below.

### 1. Loading a Pretrained Model for Inference
```python
import torch
import argparse
from models import ECG_TemporalNet1D, ModelGroup
import parameters

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, choices=['TemporalNet', 'ModelGroup_TemporalNet'], default='TemporalNet')
args = parser.parse_args([])  # Use [] for Jupyter or script usage

# Function to create and load the model
from main import create_model
model = create_model(args, baseline=False, finetune=False)
model.eval()

# Example input (batch_size=16, leads=8, samples=2500)
# The 8 leads should be ordered as: I, II, V1, V2, V3, V4, V5, V6
ecgs = torch.randn(16, 8, 2500)
with torch.no_grad():
    output = model(ecgs)
```

### 2. Loading for Finetuning
```python
model = create_model(args, baseline=False, finetune=True)
# All layers except the final layer are trainable
```

### 3. Loading a Baseline (Randomly Initialized) Model
```python
model = create_model(args, baseline=True)
# No pretrained weights loaded
```

## Requirements
- Python 3.10+
- PyTorch (tested with >=1.10)
- numpy

Install dependencies with:
```bash
pip install -r requirements.txt
```
