import torch
import tqdm
import os
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForMaskedLM
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def evaluate_model_on_dataset(
    checkpoint_path, device="cuda:0"):
    max_seq_length = 1024
    batch_size = 8
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("Cyber-ThreaD/SecureBERT-CyNER")
    raw_dataset = load_dataset("bnsapa/cybersecurity-ner")
    label_list = raw_dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    print(label_list)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_length
        )
        labels = []
        for i, ner_tags_for_example in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            current_labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    current_labels.append(-100)
                elif word_idx != previous_word_idx:
                    current_labels.append(ner_tags_for_example[word_idx])
                else:
                    current_labels.append(-100)
                previous_word_idx = word_idx
            labels.append(current_labels)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply preprocessing to the test split
    processed_test_dataset = raw_dataset["test"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset["test"].column_names
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # DataLoader (no DistributedSampler needed for single GPU)
    test_dataloader = torch.utils.data.DataLoader(
        processed_test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle for evaluation
        collate_fn=data_collator
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        attn_implementation="sdpa",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # model = AutoModelForMaskedLM.from_pretrained(
    #     "answerdotai/ModernBERT-base",
    #     attn_implementation="sdpa",
    #     num_labels=num_labels
    # )

    # Load trained model weights
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device) # Move model to the specified device
    print("Model weights loaded.")

    model = AutoModelForTokenClassification.from_pretrained("Cyber-ThreaD/SecureBERT-CyNER")
    model.to(device)
    # model = AutoModelForTokenClassification.from_pretrained(
    #     "Cyber-ThreaD/SecureBERT-APTNER",
    #     num_labels=num_labels,
    #     id2label=id2label,
    #     label2id=label2id)
    # model.to(device)

    # --- Evaluation Loop ---
    model.eval() # Set model to evaluation mode
    all_predictions = []
    all_true_labels = []
    total_eval_loss = 0

    print("Starting evaluation...")
    progress_bar = tqdm.tqdm(test_dataloader, desc="Evaluating")

    with torch.no_grad(): # Disable gradient calculations for inference
        for batch in progress_bar:
            # Move batch tensors to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            # Get predictions (argmax over the label dimension)
            predictions = torch.argmax(logits, dim=-1)

            # Convert to CPU and numpy for processing with seqeval
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            # Collect predictions and true labels, ignoring -100
            for pred_seq, true_seq in zip(predictions, labels):
                true_labels_filtered = []
                predictions_filtered = []
                for p, t in zip(pred_seq, true_seq):
                    if t != -100: # -100 is the ignore_index
                        true_labels_filtered.append(id2label[t])
                        predictions_filtered.append(id2label[p])
                all_true_labels.append(true_labels_filtered)
                all_predictions.append(predictions_filtered)

    # Calculate overall average loss
    avg_eval_loss = total_eval_loss / len(test_dataloader)

    print("\n--- Evaluation Results ---")
    print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")

    # Generate and print the classification report
    report = classification_report(all_true_labels, all_predictions, digits=4)
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":

    model_checkpoint_path = "final_ner_original/take_2/checkpoint_epoch_20.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate the model
    evaluate_model_on_dataset(
        checkpoint_path=model_checkpoint_path,
        device=device
    )