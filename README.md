# PMDC-Net: Channel-weighted multi-scale dilated feature network for robust retinal vessel segmentation in OCTA

## Overview

PMDC-Net is a novel deep learning architecture designed for robust retinal vessel segmentation in Optical Coherence Tomography Angiography (OCTA) images. The model combines multiple advanced techniques including channel-weighted multi-scale dilated convolutions, attention mechanisms, and feature fusion modules to achieve superior segmentation performance.

## Key Features

### 🏗️ **Architecture Components**

- **PDCM (Parallel Dilated Convolution Module)**: Multi-scale dilated convolutions with different dilation rates (1, 2, 3) for capturing vessels at various scales
- **MSFAM (Multi-Scale Feature Attention Module)**: Attention mechanism with dilated convolutions and channel attention for enhanced feature representation
- **CWM (Channel Weighting Module)**: Channel-wise attention for optimal feature fusion between encoder and decoder paths
- **DAFM (Decoder Attention Fusion Module)**: Spatial attention for precise feature alignment during upsampling

### 🎯 **Key Innovations**

1. **Multi-Scale Dilated Convolutions**: Captures vessels of different sizes and complexities
2. **Channel-Weighted Feature Fusion**: Intelligent weighting of encoder and decoder features
3. **Attention-Guided Decoding**: Spatial and channel attention mechanisms for precise segmentation
4. **Modular Design**: Flexible architecture allowing component-wise activation/deactivation

## Model Architecture

```
Input → PDCM → MSFAM → Encoder Path (4 levels)
                    ↓
Decoder Path ← DAFM ← CWM ← Upsampling
     ↓
Output Segmentation
```

## Usage

### Basic Usage

```python
import torch
from src.models.pmdc_net import PMDC_Net

# Initialize the model
model = PMDC_Net(
    n_channels=1,      # Input channels (grayscale OCTA)
    n_classes=2,        # Background and vessel classes
    bilinear=False,    # Use transposed convolution for upsampling
    use_msfam=True,    # Enable Multi-Scale Feature Attention Module
    use_cwm=True,      # Enable Channel Weighting Module
    use_pdcm=True,     # Enable Parallel Dilated Convolution Module
    use_dafm=True      # Enable Decoder Attention Fusion Module
)

# Example forward pass
input_tensor = torch.randn(1, 1, 512, 512)  # Batch of 1, 1 channel, 512x512 images
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # [1, 2, 512, 512]
```

### Advanced Configuration

```python
# Minimal configuration (faster inference)
model_minimal = PMDC_Net(
    use_msfam=False,
    use_cwm=False,
    use_pdcm=False,
    use_dafm=False
)

# Full configuration (best performance)
model_full = PMDC_Net(
    use_msfam=True,
    use_cwm=True,
    use_pdcm=True,
    use_dafm=True
)
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd PMDC-Net

# Install dependencies
pip install torch torchvision numpy
```

## Datasets

### ROSE Dataset
- **ROSE-1**: 117 OCTA images from 39 subjects
- **Used Subset**: SVC (Superficial Vascular Complex) images
- **Download**: [ROSE Dataset](https://imed.nimte.ac.cn/dataofrose.html)

### OCTA-500 Dataset
- **OCTA-500**: 500 subjects with OCTA imaging under two FOVs
- **Used Projection**: Maximum projection from Inner Limiting Membrane (ILM) to Outer Plexiform Layer (OPL)
- **Features**: OCT/OCTA volumes, projections, text labels, segmentation labels
- **Download**: [OCTA-500 Dataset](https://ieee-dataport.org/open-access/octa-500)

## Input Specifications

- **Input Image Size**: 512 × 512 pixels
- **Input Channels**: 1 (grayscale OCTA images)
- **Output Classes**: 2 (background and vessel)

## Performance

PMDC-Net demonstrates superior performance on retinal vessel segmentation tasks with:
- Enhanced multi-scale feature extraction
- Improved vessel boundary detection
- Robust performance across different OCTA imaging conditions
- Efficient computation with modular design

## Contact

For questions and support, please contact [13640440419@163.com].

## Acknowledgments
- ROSE dataset: [ROSE Dataset](https://imed.nimte.ac.cn/dataofrose.html)
- OCTA-500 dataset: [OCTA-500 Dataset](https://ieee-dataport.org/open-access/octa-500)
- PyTorch framework for deep learning implementation
