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

## Model Modes
The models can be loaded in three distinct modes, depending on your use case:

1. **Baseline** (Randomly Initialized)
   - No pretrained weights are loaded. All model parameters are randomly initialized.
   - Use for training from scratch or as a control.
   - `baseline=True`

2. **Frozen** (Pretrained, Only Final Layer Trainable)
   - Loads pretrained weights. All layers except the final layer are frozen (not trainable); only the final layer is updated during training.
   - Use for transfer learning when you want to adapt only the classifier to new data.
   - `baseline=False`, `finetune=False`

3. **Finetune** (Pretrained, All Parameters Trainable)
   - Loads pretrained weights. All layers except the final layer are unfrozen (trainable); the final layer is randomly initialized and trainable.
   - Use for full transfer learning or domain adaptation.
   - `baseline=False`, `finetune=True`

## Usage
Below are code snippets for loading the model in each mode. The 8-lead input should be ordered as: I, II, V1, V2, V3, V4, V5, V6.

### 1. Baseline Mode (Random Initialization)
```python
from main import create_model
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, choices=['TemporalNet', 'ModelGroup_TemporalNet'], default='TemporalNet')
args = parser.parse_args([])

model = create_model(args, baseline=True)
# All parameters are randomly initialized
```

### 2. Frozen Mode (Pretrained, Only Final Layer Trainable)
```python
model = create_model(args, baseline=False, finetune=False)
# Loads pretrained weights; only the final layer is trainable
```

### 3. Finetune Mode (Pretrained, All Parameters Trainable)
```python
model = create_model(args, baseline=False, finetune=True)
# Loads pretrained weights; all layers except the final layer are trainable
```

### Example Inference
```python
import torch
model.eval()
ecgs = torch.randn(16, 8, 2500)  # 8 leads: I, II, V1-V6
with torch.no_grad():
    output = model(ecgs)
```

## Requirements
- Python 3.10+
- PyTorch (tested with >=1.10)
- numpy

Install dependencies with:
```bash
pip install -r requirements.txt
```
