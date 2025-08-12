import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from dataset import Eval_SentimentVulnerabilityDataset  # your dataset class
from tqdm import tqdm


# ---------- Collate ----------
def cls_collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


# ---------- Evaluation ----------
def evaluate(
    ckpt_path: str,
    batch_size: int = 32,
    max_len: int = 1024,
    device: torch.device = torch.device("cuda:0"),
):
    # 1. Dataset and dataloader
    test_ds = Eval_SentimentVulnerabilityDataset()  # adjust if needed
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cls_collate_fn,
    )

    # 2. Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        num_labels=2,
        attn_implementation="sdpa",
    )

    # Load model directly

    # tokenizer = AutoTokenizer.from_pretrained("SynamicTechnologies/CYBERT")
    # model = AutoModelForSequenceClassification.from_pretrained("SynamicTechnologies/CYBERT")
    # 3. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    # 4. Inference loop
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in tqdm(test_dl):
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            labels = labels.to(device)
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # 5. Metrics
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    cm = confusion_matrix(y_true, y_pred)

    print("======= Evaluation (single GPU) =======")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix (rows: true, cols: pred):")
    print(cm)


# ---------- CLI ----------
if __name__ == "__main__":
    evaluate(
        ckpt_path="sentiment_classif/checkpoint_epoch_6.pth", max_len = 1024
    )