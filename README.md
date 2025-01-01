# OptimCLM: Optimizing Clinical Language Models for Predicting Patient Outcomes

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Highlights](#highlights)
4. [Installation](#installation)
5. [Dataset](#dataset)
6. [Methods](#methods)
7. [Results](#results)
8. [Usage Instructions](#usage-instructions)
9. [Training Curves](#training-curves)
10. [Citation](#citation)
11. [License](#license)

---

## Introduction
This repository contains the implementation of **OptimCLM**, a framework for optimizing Clinical Language Models (CLMs) through **knowledge distillation**, **pruning**, and **quantization**. The framework is designed to improve efficiency and facilitate real-world deployment without significant performance loss. The methods focus on key clinical predictive tasks such as:
- **Mortality Prediction**
- **Length of Stay Prediction**
- **Procedure Prediction**
- **Diagnosis Prediction**

### Publication
The research was published in the *International Journal of Medical Informatics*. Access the manuscript [here](https://doi.org/10.1016/j.ijmedinf.2024.105764).

---

## Repository Structure
```
├── Figures/                 # Contains figures used in notebooks and documentation
├── LOS_Ensemble.ipynb       # Notebook for creating and evaluating ensemble models
├── LOS_Optimization.ipynb   # Notebook for model optimization (distillation, pruning, quantization)
├── LICENSE                  # License information
├── README.md                # Repository documentation
```

---

## Highlights
- **Optimized CLMs** for real-world clinical applications using **knowledge distillation**, **pruning**, and **quantization**.
- Achieved **22.88× compression** and **28.7× inference speedup** with <5% performance loss.
- Improved **macro-averaged AUROC** on major clinical outcome prediction tasks.
- Enhanced domain-knowledge transfer through **ensemble learning**.

---

## Installation
### Clone the Repository
```bash
git clone https://github.com/junayed-hasan/Clinical-Language-Model-Distillation-Pruning-Quantization.git
cd Clinical-Language-Model-Distillation-Pruning-Quantization
```

### Install Dependencies
Ensure Python 3.6+ is installed, then run:
```bash
pip install torch torchvision torchaudio transformers==4.15.0 \
    matplotlib==3.4.3 numpy==1.21.2 pandas==1.3.3 \
    scikit-learn==0.24.2 scipy==1.7.1 jupyter==1.0.0
```

---

## Dataset
This project uses the **MIMIC-III clinical database**. Refer to the official repository [clinical-outcome-prediction](https://github.com/bvanaken/clinical-outcome-prediction) for instructions on accessing and preparing the dataset.

---

## Methods
The methodology involves:
1. **Teacher Model Selection**: Domain-specific models (**DischargeBERT**, **COReBERT**) combined in an ensemble.
2. **Knowledge Distillation**: Transfer knowledge to smaller models (**TinyBERT**, **BERT-PKD**).
3. **Model Compression**: Apply pruning and quantization to reduce size and inference latency.

![OptimCLM Framework Architecture](Figures/archi.png)

---

## Results
### Preliminary Results
Macro-averaged AUROC results for various tasks are summarized below:

| Model                | Diagnosis (%) | Procedure (%) | Mortality (%) | Length of Stay (%) |
|----------------------|---------------|---------------|---------------|--------------------|
| COReBERT + DischargeBERT | 85.93 ± 0.072 | 88.67 ± 0.078 | 84.11 ± 0.038 | 73.82 ± 0.017      |

For detailed experimental results, refer to the publication.

---

## Usage Instructions
### Running Notebooks
1. Launch Jupyter:
    ```bash
    jupyter notebook
    ```
2. Open `LOS_Ensemble.ipynb` to:
   - Train and evaluate ensemble models.
3. Open `LOS_Optimization.ipynb` to:
   - Optimize student models through distillation, pruning, and quantization.

---

## Training Curves
The following curves depict training progress for all experiments:
![Training Curves](Figures/training_curves.png)

---

## Citation
If you use this repository or find it helpful, please cite:
```bibtex
@article{hasan2024optimclm,
    title={OptimCLM: Optimizing Clinical Language Models for Predicting Patient Outcomes via Knowledge Distillation, Pruning, and Quantization},
    author={Mohammad Junayed Hasan, Fuad Rahman, Nabeel Mohammed},
    journal={International Journal of Medical Informatics},
    year={2025},
    doi={10.1016/j.ijmedinf.2024.105764}
}
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2024, Mohammad Junayed Hasan 
