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

4.  **Ensure `dataset.py` is available:**
    The script assumes the existence of a `dataset.py` file containing the `ModernBertDataset` class, which handles loading your specific data. If this file is not provided, you will need to create it based on your data structure. The current script expects datasets compatible with the `ModernBertDataset` class, which takes paths to `Primus-Seed`, `Primus-FineWeb`, and `Primus-Instruct` datasets.

    *Example `ModernBertDataset` (conceptual, you need to implement this based on your data):*
    ```python
    # dataset.py
    import torch
    from torch.utils.data import Dataset
    import datasets # pip install datasets
    import os

    class ModernBertDataset(Dataset):
        def __init__(self, primus_seed_path, primus_fineweb_dir, primus_instruct_path, max_length=1024):
            # Load your datasets here. Example using Hugging Face datasets library:
            self.dataset = datasets.concatenate_datasets([
                datasets.load_dataset("arrow", data_files=primus_seed_path, split="train"),
                datasets.load_dataset("arrow", data_files=[os.path.join(primus_fineweb_dir, f) for f in os.listdir(primus_fineweb_dir) if f.endswith('.arrow')], split="train"),
                datasets.load_dataset("arrow", data_files=primus_instruct_path, split="train")
            ])
            # You might need to process/tokenize the dataset here or in __getitem__
            # For simplicity, assuming 'text' column exists and will be tokenized by DataCollator
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # Return a dictionary with 'input_ids', 'attention_mask' etc.
            # The DataCollatorForLanguageModeling will handle masking and label creation.
            # Assuming your dataset yields a 'text' field.
            text = self.dataset[idx]['text'] # Adjust 'text' to your actual column name
            return {"text": text}

    ```
    *Note: The original code mentions `https://github.com/CVEProject/cvelistV5 (Title, description)`. If your `ModernBertDataset` is designed to process CVE data, ensure its implementation correctly handles the structure of that data.*

## Run Evaluation on provided weights


    --log_interval 100
