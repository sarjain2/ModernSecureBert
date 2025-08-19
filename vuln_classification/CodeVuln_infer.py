import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm 
from dataset import Eval_SentimentVulnerabilityDataset

# Define a mapping for the labels (assuming 0 for non-vulnerable, 1 for vulnerable)
LABEL_MAP = {
    0: "Non-Vulnerable",
    1: "Vulnerable"
}

def run_dual_inference(
    text: str,
    ckpt_path: str,
    true_label: int = None, # Optional ground truth label for comparison
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    if true_label is not None:
        print(f"Ground Truth Label: {true_label} ({LABEL_MAP.get(true_label, 'Unknown')})")
    else:
        print("No Ground Truth Label provided for comparison.")

    # --- Model 1: ModernBERT from Checkpoint ---
    print("\n--- Model 1: ModernBERT (from Checkpoint) ---")
    try:
        modernbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        modernbert_model = AutoModelForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-base",
            num_labels=2,
            attn_implementation="sdpa",
        )

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        modernbert_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        modernbert_model.to(device)
        modernbert_model.eval()

        # Tokenize and infer
        # Note: The original eval script used max_len=1024 for ModernBERT
        enc_modernbert = modernbert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=1024, # Use 1024 for ModernBERT as per original eval script
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits_modernbert = modernbert_model(**enc_modernbert).logits
            pred_id_modernbert = logits_modernbert.argmax(dim=-1).item()

        print(f"Prediction (ModernBERT): {pred_id_modernbert} ({LABEL_MAP.get(pred_id_modernbert, 'Unknown')})")

    except Exception as e:
        print(f"Error loading or inferring with ModernBERT model from checkpoint: {e}")
        print(f"Please ensure '{ckpt_path}' is correct and the model architecture matches 'answerdotai/ModernBERT-base'.")
        pred_id_modernbert = None


    # --- Model 2: CodeBERT (mahdin70/codebert-devign-code-vulnerability-detector) ---
    print("\n--- Model 2: CodeBERT (mahdin70/codebert-devign-code-vulnerability-detector) ---")
    try:
        codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        codebert_model = AutoModelForSequenceClassification.from_pretrained("mahdin70/codebert-devign-code-vulnerability-detector")

        codebert_model.eval()
        codebert_model.to(device)

        # Tokenize and infer
        # Note: The original eval script suggested max_len=512 for CodeBERT
        enc_codebert = codebert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512, # Use 512 for CodeBERT as per original eval script comments
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits_codebert = codebert_model(**enc_codebert).logits
            pred_id_codebert = logits_codebert.argmax(dim=-1).item()

        print(f"Prediction (CodeBERT): {pred_id_codebert} ({LABEL_MAP.get(pred_id_codebert, 'Unknown')})")

    except Exception as e:
        print(f"Error loading or inferring with CodeBERT model: {e}")
        pred_id_codebert = None

    print("\n-----------------------------------------\n")

    return {
        "modernbert_prediction": pred_id_modernbert,
        "codebert_prediction": pred_id_codebert,
        "true_label": true_label
    }

# --- Barcoded Example Usage ---
if __name__ == "__main__":
    CKPT_PATH = "sentiment_classif/checkpoint_epoch_6.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_ds = Eval_SentimentVulnerabilityDataset()
    # Can change this to try whatever model you want
    code, gt = test_ds[100]

    print(code)
    results_1 = run_dual_inference(code, CKPT_PATH, true_label = gt)

    print("\n--- Summary of Results ---")
    print(f"Example 1: {results_1['true_label']}, SecureBert 2.0 Pred: {results_1['modernbert_prediction']}, CodeBERT Pred: {results_1['codebert_prediction']}")
 
