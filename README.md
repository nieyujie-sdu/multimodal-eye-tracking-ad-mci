# Multimodal Eye-Tracking-Based Classification of AD, MCI, and HC

This repository provides an implementation of a multimodal eye-tracking-based classification framework for Alzheimer's disease (AD), mild cognitive impairment (MCI), and healthy controls (HC).

The code supports binary and three-class classification tasks, modality ablation experiments, and comparisons of different multimodal fusion strategies.

## Features

- **Classification tasks:** AD vs. HC, MCI vs. HC, and AD vs. MCI vs. HC.
- **Multimodal inputs:** eye-tracking heatmaps, eye-movement sequences, and behavioral features.
- **Modality ablation:** select individual modalities or combinations of modalities.
- **Fusion comparisons:** evaluate the proposed fusion strategy and alternative fusion methods.
- **Subject-level cross-validation:** organize data splits at the participant level to avoid placing trials from the same participant in different folds.
- **Data auditing:** check data matching and fold assignments without training a model.
- **Reproducibility:** save experiment configurations, evaluation results, and cross-validation summaries.

## Requirements

- Python 3.10 or a compatible Python version
- PyTorch and a compatible torchvision version
- NumPy
- pandas
- Pillow
- scikit-learn
- openpyxl

Install the required packages in your Python environment:

```bash
pip install numpy pandas torch torchvision Pillow scikit-learn openpyxl
```

For GPU training, install a PyTorch version compatible with your CUDA environment. The exact package versions should be selected and tested for your hardware and software configuration.

## Repository Structure

A typical project layout is shown below:

```text
your-repository/
├── README.md
├── requirements.txt
├── .gitignore
└── code/
    └── train_multimodal_admci.py
```

The dataset should be stored locally and supplied through the `--data-root` argument. **The dataset is not included in this repository.**

The expected dataset directory structure and file-naming conventions must match those implemented in `code/train_multimodal_admci.py`. See the data preparation section below before running the experiments.

## Data Preparation

Before training, prepare the dataset in the directory specified by `--data-root`.

The data files must follow the naming conventions and structural requirements expected by the training script, including participant identifiers, modality-specific files, and task-related data organization.

Refer to the data-loading and validation functions in `code/train_multimodal_admci.py` for the exact directory structure and required spreadsheet columns.

To check data matching and generate fold assignments without model training, run:

```bash
python code/train_multimodal_admci.py \
  --task all \
  --data-root "/path/to/your/dataset" \
  --audit-only
```

Replace `/path/to/your/dataset` with the actual dataset directory on your computer.

## Usage

Run the following commands from the repository root.

### Run all classification tasks

```bash
python code/train_multimodal_admci.py \
  --task all \
  --data-root "/path/to/your/dataset" \
  --output-root "./runs" \
  --device cuda:0
```

### Run a single classification task

For AD vs. HC:

```bash
python code/train_multimodal_admci.py \
  --task adhc \
  --data-root "/path/to/your/dataset" \
  --output-root "./runs" \
  --device cuda:0
```

Other supported task options include:

- `adhc`: AD vs. HC
- `mcihc`: MCI vs. HC
- `admcihc`: AD vs. MCI vs. HC
- `all`: run all configured tasks

Replace the example data path with your local dataset path. Use `--device cpu` if a CUDA-enabled GPU is unavailable.

### Run the architecture smoke test

The smoke test checks the model architecture without requiring the real dataset. To avoid downloading pretrained heatmap-encoder weights, use:

```bash
python code/train_multimodal_admci.py \
  --smoke-test \
  --no-pretrained-heatmap \
  --device cpu
```

## Ablation Experiments

### Modality ablation

Use `--modalities` to select the input modalities.

Sequence and behavioral features:

```bash
python code/train_multimodal_admci.py \
  --task admcihc \
  --data-root "/path/to/your/dataset" \
  --modalities sequence behavior \
  --device cuda:0
```

Heatmap only:

```bash
python code/train_multimodal_admci.py \
  --task adhc \
  --data-root "/path/to/your/dataset" \
  --modalities heatmap \
  --device cuda:0
```

Use the modality names and combinations supported by the script.

### Fusion-strategy comparison

Use `--fusion` to select the fusion strategy:

```text
--fusion concatenation
--fusion gated
--fusion transformer
--fusion proposed
```

The proposed fusion strategy is the default. The alternative strategies retain the selected encoders and classification head, allowing comparisons of fusion methods under the corresponding experimental configuration.

## Outputs

Each run creates a timestamped experiment directory under the specified output directory. Depending on the selected task and execution mode, the outputs may include:

- `data_report.json`: data-matching results and quality-control information.
- `invocation.json`: software-version information and the command configuration.
- `run_config.json`: task-specific experiment configuration.
- `fold_assignments.json`: participant-to-fold assignments.
- Fold-specific split information, training history, selected checkpoints, and test metrics.
- Subject-level prediction probabilities.
- `cv_summary.json`: mean and sample standard deviation across held-out test folds, together with pooled out-of-fold predictions.

The mean and standard deviation across folds and the metrics computed from pooled out-of-fold predictions are **different evaluation summaries** and should not be interpreted as interchangeable.

