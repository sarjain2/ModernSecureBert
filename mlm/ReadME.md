# ModernBERT Pre-training with PyTorch Lightning Fabric

This repository contains a script for pre-training the `answerdotai/ModernBERT-base` model using Masked Language Modeling (MLM) on custom datasets. It leverages PyTorch Lightning Fabric for efficient distributed training across multiple GPUs.

## Table of Contents

-   [Features](#features)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Dataset Preparation](#dataset-preparation)
-   [Usage](#usage)
-   [Command-Line Arguments](#command-line-arguments)
-   [Training Details](#training-details)
-   [Checkpointing](#checkpointing)
-   [Logging](#logging)
-   [Notes](#notes)

## Features

*   **Distributed Training**: Utilizes PyTorch Lightning Fabric's Distributed Data Parallel (DDP) strategy for scalable training on multiple GPUs.
*   **Masked Language Modeling (MLM)**: Pre-trains the ModernBERT model using the MLM objective.
*   **Configurable Hyperparameters**: All key training parameters are exposed as command-line arguments for easy customization.
*   **Checkpointing**: Automatically saves model checkpoints at the end of each epoch, allowing for training resumption.
*   **Gradient Clipping**: Implements gradient norm clipping to stabilize training.
*   **Learning Rate Scheduling**: Uses a linear learning rate scheduler with warmup steps.
*   **Comprehensive Logging**: Logs training progress (loss, learning rate) to both console and a file.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.8+
*   NVIDIA GPUs with compatible drivers (for GPU training)
*   CUDA Toolkit

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Adjust cu121 for your CUDA version
    pip install transformers lightning tqdm pandas pyarrow
    ```
    *Note: Ensure your `torch` installation matches your CUDA version. The example above is for CUDA 12.1.*

4.  **Ensure `dataset.py` is available: and also the datasets are available**

```
python3 primus_load.py 
```

## Run Train script

When you run primus load you will be given many huggingface datasets in local workspace

Change the following lines:

1. Change to your checkpoint path

train_model(checkpoint_path = "no_secure_bert_data_internet_data_pths/checkpoint_epoch_10.pth")

2. Change these paths pased on paths created by primus load respectively

```
primus_seed_pth = "Primus-Seed_dataset/train/data-00000-of-00001.arrow"
primus_fineweb_dir = "Primus-FineWeb_dataset/train"
primus_instruct_pth = "Primus-Instruct_dataset/train/data-00000-of-00001.arrow"
```

# Run Eval Scripts

Parser for eval script

```
 parser = argparse.ArgumentParser(description="Evaluate a model on a given dataset.")

    # Add arguments
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint file (e.g., checkpoint_epoch_10.pth)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Object_prediction.csv", # Default value
        help="Path to the dataset CSV file (e.g., Object_prediction.csv)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=2, # Default value
        help="Number of top predictions to evaluate (e.g., 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu", # Default value based on availability
        help="Device to use for evaluation (e.g., 'cuda' or 'cpu')",
    )
```

Same arguments for other eval scripts