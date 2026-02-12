# Deep Learning Karyotyping Pipeline

![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Deep Learning pipeline for automated chromosome karyotyping.
This project implements a novel **"Triad of Features"** approach (Geometry, Scale, Texture) to classify human chromosomes with high precision, overcoming the limitations of traditional shape-only analysis.

## 🚀 Key Features

*   **Robust Segmentation**: Uses a **Mask R-CNN** (ResNet50-FPN) to detect and isolate individual chromosomes from metaphase spreads, even in overlapping cases.
*   **Advanced Classification**: A **Hybrid ResNet** classifier that integrates:
    *   **Geometric Features**: Aspect Ratio, Centromere Index (CI).
    *   **Texture Features**: Banding pattern intensity and contrast.
    *   **Scale Features**: Relative chromosome area.
*   **Centromere Detection**: Automated algorithm to identify the centromeric constriction and calculate the p/q arm ratio.
*   **Full Pipeline**: From raw image to annotated karyogram.

## 📂 Dataset Citation

This project utilizes the **CIL:54816** dataset of human epithelial cells.
If you use this code or data, please cite the original authors:

> **Chien-Hsing Lu, Chih-En Kuo, Jenn-Jhy Tseng (2022).**
> *CIL:54816, Homo sapiens Linnaeus, 1758, epithelial cell.*
> CIL. Dataset. (RRID:SCR_003510).
> [https://doi.org/10.7295/W9CIL54816](https://doi.org/doi:10.7295/W9CIL54816)

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AI-Assisted-Chromosomes-Karyotyping.git
    cd karyotyping-pipeline
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage

### 1. Training
The pipeline is split into two stages for maximum accuracy.

**Stage 1: Segmentation Training**
Trains the Mask R-CNN to localize chromosomes.
```bash
python3 train_segmentation.py
```
*Outputs: `mask_rcnn_karyotype_best.pth`*

**Stage 2: Classification Training**
Trains the Hybrid Classifier on the extracted feature triad.
```bash
python3 train_classification.py
```
*Outputs: `karyotype_classifier_best.pth`*

**Optimized Execution (Background)**
To run the full training suite in the background:
```bash
nohup ./run_training.sh &
```

### 2. Inference
To analyze a new metaphase spread image:
```bash
python3 inference_pipeline.py
```
This will:
1.  Detect chromosomes using the segmentation model.
2.  Extract the feature triad for each chromosome.
3.  Classify them into 24 distinct types (1-22, X, Y).
4.  Generate an annotated image `inference_result.png`.

### Karyogram Module – Current Limitations

Although this repository includes an automated karyogram construction module, it does not yet produce biologically consistent chromosome numbering (1–22, X, Y).

Why?

Chromosome numbering is fundamentally a relative ranking problem, not a pure image classification task.

In early experiments, the model was trained to directly predict chromosome identities (e.g., A1, C7, D14). However, this approach led to:

Repeated assignment of the same chromosome label

Confusion between morphologically similar chromosomes

Incorrect grouping of X/Y chromosomes

This occurs because:

Many chromosomes share similar size and banding patterns

Chromosome identity depends on relative length within the same metaphase spread

CNN-based classifiers operate on isolated crops without global context

As a result, morphology-only classification is insufficient for reliable karyogram reconstruction.

### Future Improvements

Planned improvements include:

Length-based global ranking

Centromere index computation

Hybrid rule-based + ML grouping

Improved homolog pairing strategy
