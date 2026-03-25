# DCESR: Dual-Channel Emotion- & Semantic-aware Recommender system

![Last Commit](https://img.shields.io/github/last-commit/kkkimsuji/Rec-SSP?style=flat-square)

This repository contains the official implementation of the following paper:
> Pyo, S., **Kim, S.**, Li, X., & Kim, J. Beyond Sentiment: A Dual-Channel Approach to Emotion and Review Summarization for Recommendation. IEEE Access. -Under Review

## Overview

DCESR (Dual-Channel Emotion- & Semantic-aware Recommender) is an advanced recommendation framework designed to bridge the gap between textual feedback and rating prediction. Unlike traditional models, DCESR operates through two distinct analytical channels to capture the full spectrum of user-item interactions. The Semantic Channel utilizes the BART model for abstractive summarization, extracting high-level item attributes and long-term user preferences from noisy review text. Simultaneously, the Emotion Channel employs DistilRoBERTa to distill 7-class emotional probability vectors, capturing the psychological nuances behind user ratings. These heterogeneous features are then adaptively integrated through a Gated Multimodal Unit (GMU), which learns a dynamic weighting mechanism to balance the importance of semantic context and emotional sentiment for each specific prediction. By fusing deep linguistic understanding with fine-grained sentiment analysis, DCESR provides a robust and interpretative solution for modern recommender systems.

## Environment & Requirements

This project is implemented in **Python 3.8+**. To ensure reproducibility, please install the specific versions of the libraries listed below.

### 1. Key Dependencies
| Category | Library | Version | Description |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow` / `Keras` | `2.21.0` / `3.13.2` | Implements the core neural network and Gated Multimodal Unit (GMU). |
| **NLP** | `Transformers` | `5.3.0` | Provides pre-trained BART (Semantics) and DistilRoBERTa (Emotion) models. |
| **NLP Backend** | `PyTorch` | `2.11.0` | Serves as the high-performance backend for Hugging Face transformer models. |
| **Analysis** | `Pandas` | `3.0.1` | Handles data loading, 5-core filtering, and review set aggregation. |
| **Matrix** | `NumPy` | `2.4.3` | Facilitates efficient numerical operations on 768D and 7D embedding vectors. |
| **ML Tools** | `scikit-learn` | `1.8.0` | Manages train/val/test splitting and calculates performance metrics (MAE, RMSE). |

### 2. Utility Libraries
- **`PyYAML` (`6.0.3`)**: Essential for parsing the `config.yaml` to manage hyperparameters and file paths dynamically.
- **`PyArrow` (`23.0.1`)**: Used as the high-performance engine for saving and loading large Parquet data splits.
- **`tqdm` (`4.67.3`)**: Provides real-time visual feedback for long-running embedding extraction processes.
- **`h5py` (`3.14.0`)**: Handles the serialization and storage of trained Keras model weights.
- **`Huggingface Hub` (`1.7.2`)**: Manages the seamless downloading of pre-trained NLP model weights.


## Repository Structure

The repository is organized as follows to ensure a clear workflow from data preprocessing to model evaluation:

```text
DCESR/
├── main.py              # Central orchestrator to run the entire pipeline
├── config.yaml          # Global configurations (hyperparameters, paths, device)
├── requirements.txt     # List of required Python libraries
├── .gitignore           # Specifies files/folders to be ignored by Git
├── README.md            # Project documentation and overview
│
├── model/               # Model Architecture
│   ├── __init__.py
│   └── proposed.py      # DCESR Model (BART + DistilRoBERTa + GMU)
│
├── scr/                 # Source Code Modules
│   ├── __init__.py
│   ├── data_processing.py # Data loading, 5-core filtering, and set aggregation
│   ├── trainer.py         # Training loop, evaluation metrics, and data splitting
│   ├── bart.py            # BART-based semantic feature extraction
│   └── distilroberta.py   # DistilRoBERTa-based emotion feature extraction
│
└── data/                # Data Directory
    ├── raw/
    │   └── SampleData.json.gz # Sample dataset for testing
    └── processed/       # (Generated) Processed .pkl and .parquet files
```

## How to Run
### 1. Installation & Environment Setup
We recommend using a virtual environment to manage dependencies.

```bash
python -m venv .venv

pip install -r requirements.txt
```
### 2. Data Preparation
The model requires the dataset to be placed in the specific directory defined in the structure.
- **Prepare Data**: Place your original dataset (e.g., SampleData.json.gz) into the data/raw/ directory.
- **Automatic Preprocessing**: When you run the main script, the pipeline will automaticall performs.

### 3. Configuration
You can customize hyperparameters and file paths in the centralized config file.
- File Path: ```config.yaml```

### 4. Train and Evaluate
Once the environment and data are ready, execute the following command to start the full workflow (Preprocessing → Training → Evaluation):
```
python main.py
```

## Model Description

The DCESR (Dual-Channel Emotion- & Semantic-aware Recommender) is a deep learning-based recommendation framework designed to predict user ratings by analyzing the multidimensional nature of review texts.

The model operates through a dual-channel architecture. The Semantic Channel utilizes a pre-trained BART encoder to extract 768-dimensional vectors from summarized user and item review sets, capturing high-level attributes and preferences. Simultaneously, the Emotion Channel employs DistilRoBERTa to analyze 7-class emotion probabilities, generating a fine-grained psychological profile for each user and item.

To integrate these diverse features, the model uses a Gated Multimodal Unit (GMU), which adaptively calculates weights to balance the importance of semantic meaning versus emotional sentiment for every specific prediction. Finally, this fused representation is passed through a neural network to output a precise numerical rating.

<img width="1162" height="690" alt="image" src="https://github.com/user-attachments/assets/cdd1e1cf-b494-4ea9-b686-6e0fe2052eb1" />


## Experimental Results
The following table summarizes the performance comparison across three Amazon review datasets. DCESR consistently achieves the lowest error rates in both MAE and RMSE.

<table width="100%" style="border-collapse: collapse; text-align: center; font-family: sans-serif; border: 1px solid #ddd;">
  <thead>
    <tr style="background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;">
      <th rowspan="2" style="padding: 12px; border: 1px solid #ddd;">Model</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Books</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Movies and TV</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Office Products</th>
    </tr>
    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">NCF</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.703</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.902</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.099</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.360</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.638</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.911</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">DeepCoNN</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.525</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.767</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.784</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.086</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.580</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.855</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">NARRE</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.530</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.791</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.780</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.083</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.584</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.858</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">DRRNN</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.498</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.764</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.724</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.081</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.523</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.844</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">AENAR</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.508</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.760</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.750</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.083</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.559</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.852</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">MFNR</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.501</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.777</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.738</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.093</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.526</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.850</td>
    </tr>
    <tr style="background-color: #e6ffed; font-weight: bold;">
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">DCESR (Ours)</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.485</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.745</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.690</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">1.035</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.498</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.809</td>
    </tr>
  </tbody>
</table>
