- 

  # **PMDC-Net**

  **PMDC-Net: Channel-weighted Multi-scale Dilated Feature Network for Robust Retinal Vessel Segmentation in OCTA**

  ------

  ## 📌 Overview

  This repository provides the official implementation of **PMDC-Net**, a deep learning framework designed for **retinal vessel segmentation in Optical Coherence Tomography Angiography (OCTA)** images.

  Accurate OCTA vessel segmentation is challenging due to:

  - low contrast of capillary vessels,
  - large scale variations between major vessels and microvasculature,
  - noise and projection artifacts,
  - discontinuities in thin vessels.

  To address these challenges, PMDC-Net introduces a set of **complementary modules** within an encoder–decoder architecture to enhance multi-scale representation, feature fusion, and fine vessel preservation.

  The proposed method has been evaluated on multiple public OCTA datasets and demonstrates **robust and consistent performance across datasets**.

  ------

  ## 🧠 Network Architecture

  PMDC-Net follows an **encoder–decoder segmentation framework** enhanced with four carefully designed modules:

  📌 *Overall architecture of PMDC-Net*
  ![](src\images\PMDC-NET.JPG)

  ### 🔹 Key Modules

  #### 1. Parallel Dilated Convolution Module (PDC)

  📌 *Structure of the PDC module*
  ![](src\images\PDC.JPG)

  - Expands receptive fields using parallel dilated convolutions with different dilation rates.
  - Captures both global vessel structures and local capillary details.
  - Enhances multi-scale spatial context while maintaining parameter efficiency.

  ------

  #### 2. Multi-Scale Feature Aggregation Module (MSFA)

  📌 *Structure of the MSFA module*
  ![](src\images\MSFA.JPG)

  - Aggregates contextual features at multiple scales.
  - Introduces channel-wise attention to highlight vessel-relevant features.
  - Mitigates feature imbalance caused by large vessel scale variations.

  ------

  #### 3. Decoder Attention Fusion Module (DAF)

  📌 *Structure of the DAF module*
  ![](src\images\DAF.JPG)

  - Applies spatial attention during skip connections.
  - Suppresses noise and projection artifacts from shallow encoder features.
  - Enhances vessel boundary and continuity in the decoding stage.

  ------

  #### 4. Channel Weighting Module (CW)

  📌 *Structure of the CW module*
  ![](src\images\CW.JPG)

  - Rebalances low-level structural features and high-level semantic features.
  - Prevents fine vessel details from being overwhelmed by strong semantic responses.
  - Improves capillary detection and vessel connectivity.

  ------

  ## 📊 Experimental Results

  PMDC-Net has been evaluated on **three public OCTA datasets**:

  ### 📁 Datasets

  - **ROSE-1**
     https://imed.nimte.ac.cn/dataofrose.html
  - **OCTA-500**
    [OCTA-500 | IEEE DataPort](https://ieee-dataport.org/open-access/octa-500)
  - **ROSSA**
    [OCTA-FRNet/dataset/ROSSA at main · nhjydywd/OCTA-FRNet · GitHub](https://github.com/nhjydywd/OCTA-FRNet/tree/main/dataset/ROSSA)

  For all datasets, en face OCTA images were generated using **maximum intensity projection between the ILM and OPL layers**, which is commonly adopted for vessel visualization.

  ------

  ### 📈 Quantitative Performance

  📌 *Performance comparison on different datasets*

  | ROSE-1                |           |           |           |           |           |           |
  | --------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
  | Method                | AUC(%)    | ACC(%)    | Dice(%)   | IoU(%)    | Se(%)     | Sp(%)     |
  | U-Net(2015)           | 94.95     | 92.43     | 78.69     | 65.15     | 74.14     | 96.79     |
  | Attention U-Net(2018) | 95.07     | 92.56     | 78.90     | 65.46     | 73.75     | 97.04     |
  | CE-Net(2019)          | 94.77     | 92.68     | 79.26     | 65.98     | 74.43     | 97.09     |
  | VesselNet(2019)       | 95.08     | 92.68     | 79.23     | 65.93     | 74.01     | 97.15     |
  | Swin-Unet(2021)       | 95.00     | 92.34     | 78.72     | 65.18     | **75.23** | 96.44     |
  | TransUnet(2021)       | 94.90     | 92.36     | 78.77     | 65.27     | 75.15     | 96.46     |
  | OCT2Former(2023)      | 94.76     | 92.09     | 77.73     | 63.83     | 73.29     | 96.54     |
  | DGNet(2025)           | 94.10     | 91.45     | 76.19     | 61.77     | 73.33     | 95.81     |
  | FRNet V2(2025)        | 94.54     | 91.75     | 77.09     | 62.96     | 73.79     | 96.00     |
  | PMDC-Net(ours)        | **95.31** | **92.90** | **79.93** | **66.90** | 75.12     | **97.20** |

  | OCTA-500_3M           |           |           |           |           |           |           |
  | --------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
  | Method                | AUC(%)    | ACC(%)    | Dice(%)   | IoU(%)    | Se(%)     | Sp(%)     |
  | U-Net(2015)           | 99.73     | 98.83     | 91.11     | 83.74     | 90.74     | 99.40     |
  | Attention U-Net(2018) | 99.74     | 98.85     | 91.20     | 83.88     | 89.53     | **99.52** |
  | CE-Net(2019)          | 99.72     | 98.87     | 91.45     | 84.31     | 91.14     | 99.42     |
  | VesselNet(2019)       | **99.75** | 98.88     | 91.45     | 84.33     | 90.71     | 99.46     |
  | Swin-Unet(2021)       | 99.72     | 98.78     | 90.84     | 83.29     | 90.78     | 99.36     |
  | TransUnet(2021)       | **99.75** | 98.84     | 91.25     | 83.98     | **91.67** | 99.35     |
  | OCT2Former(2023)      | 99.72     | 98.79     | 90.82     | 83.26     | 90.05     | 99.42     |
  | DGNet(2025)           | 99.67     | 98.64     | 89.77     | 81.50     | 89.58     | 99.30     |
  | FRNet V2(2025)        | 99.71     | 98.77     | 90.72     | 83.08     | 90.39     | 99.38     |
  | PMDC-Net(ours)        | **99.75** | **98.89** | **91.54** | **84.47** | 91.08     | 99.44     |

  | OCTA-500_6M           |           |           |           |           |           |           |
  | --------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
  | Method                | AUC(%)    | ACC(%)    | Dice(%)   | IoU(%)    | Se(%)     | Sp(%)     |
  | U-Net(2015)           | 99.35     | 97.84     | 88.70     | 79.82     | 89.03     | 98.76     |
  | Attention U-Net(2018) | 99.36     | 97.86     | 88.77     | 79.93     | 88.66     | 98.82     |
  | CE-Net(2019)          | 99.35     | 97.85     | 88.76     | 79.93     | 88.78     | 98.80     |
  | VesselNet(2019)       | 99.40     | 97.91     | 89.05     | 80.39     | **89.07** | 98.84     |
  | Swin-Unet(2021)       | 99.34     | 97.82     | 88.50     | 79.50     | 87.69     | 98.88     |
  | TransUnet(2021)       | 99.37     | 97.87     | 88.74     | 79.88     | 88.27     | 98.87     |
  | OCT2Former(2023)      | 99.37     | 97.86     | 88.72     | 79.86     | 88.32     | 98.86     |
  | DGNet(2025)           | 99.26     | 97.68     | 87.74     | 78.25     | 88.05     | 98.70     |
  | FRNet V2(2025)        | 99.33     | 97.83     | 88.49     | 79.50     | 87.46     | 98.91     |
  | PMDC-Net(ours)        | **99.42** | **97.94** | **89.13** | **80.50** | 88.68     | **98.91** |

  | ROSSA                 |           |           |           |           |           |           |
  | --------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
  | Method                | AUC(%)    | ACC(%)    | Dice(%)   | IoU(%)    | Se(%)     | Sp(%)     |
  | U-Net(2015)           | 99.42     | 98.13     | 91.33     | 84.24     | 90.76     | 99.00     |
  | Attention U-Net(2018) | 99.45     | 98.21     | 91.53     | 84.54     | 90.12     | 99.17     |
  | CE-Net(2019)          | 99.48     | 98.24     | 91.70     | 84.84     | 91.35     | 99.08     |
  | VesselNet(2019)       | 99.47     | 98.22     | 91.79     | 85.02     | **92.02** | 98.97     |
  | Swin-Unet(2021)       | 99.46     | 98.16     | 91.41     | 84.36     | 90.68     | 99.05     |
  | TransUnet(2021)       | 99.45     | 98.14     | 91.37     | 84.29     | 90.96     | 98.98     |
  | OCT2Former(2023)      | 99.42     | 98.08     | 91.03     | 83.72     | 90.20     | 98.99     |
  | DGNet(2025)           | 99.32     | 98.01     | 90.54     | 82.86     | 89.74     | 99.04     |
  | FRNet V2(2025)        | 99.40     | 98.04     | 90.90     | 83.53     | 90.04     | 98.96     |
  | PMDC-Net(ours)        | **99.50** | **98.32** | **92.11** | **85.54** | 91.14     | **99.22** |

  PMDC-Net consistently achieves competitive or superior performance in terms of **AUC, Dice, IoU, Accuracy, Sensitivity, and Specificity**, demonstrating strong generalization across datasets with varying image quality and vessel characteristics.

  ------

  ### 👁 Qualitative Results

  📌 *Visual comparison of vessel segmentation results*

  ![](src\images\ROSE-1_Result.JPG)

  ![](src\images\OCTA_500_Result.JPG)

  ![](src\images\ROSSA_Result.JPG)

  Compared with existing methods, PMDC-Net:

  - better preserves thin capillaries,
  
- reduces vessel discontinuities,
    - suppresses over-segmentation and noise-induced artifacts.
    
    ------

    ## ⚙️ Implementation Details

    - Framework: **PyTorch**
    
- Input size: **512 × 512**
    - Optimizer: **Adam**
    - Learning rate: **5e-4** with cosine annealing scheduler
    - Loss function: **Weighted combination of Cross-Entropy Loss and Dice Loss**
    - Training strategy: data augmentation with geometric and photometric transformations
    
    

## 🚀 Getting Started

```
git clone https://github.com/your-username/PMDC-Net.git
cd PMDC-Net
```

*(Instructions for environment setup, training, and inference will be updated.)*

------

## 📄 Paper

If you find this work useful, please consider citing our paper:

```
The paper associated with this repository is currently under review at a peer-reviewed journal.

We will update this section with the official publication information and BibTeX citation after acceptance.
```

------

## 🙏 Acknowledgements

This work was conducted in collaboration with **Peking University Third Hospital (PKU Third Hospital)** as part of the project
 *“Early Intelligent Fundus Warning Methods for Obstructive Sleep Apnea (OSA)”*.

We thank the authors of **ROSE-1**, **OCTA-500**, and **ROSSA** datasets for making their data publicly available.

------

## 📬 Contact

For questions, discussions, or collaborations, please contact:

- Fudeng Ding
- Email: dingfudeng@xs.ustb.edu.cn
- Affiliation: University of Science and Technology Beijing

------

## 📌 Notes

- This repository is for **research purposes only**.
- Commercial use requires permission from the authors.
- The code will be continuously updated.