import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from tqdm import tqdm # Keeping tqdm import, though not strictly used for single inference

# --- Helper Function for Tokenization and Label Alignment (from your eval script) ---
def tokenize_and_align_labels(examples, tokenizer, label_list, max_seq_length):
    """
    Tokenizes text and aligns NER labels to tokens,
    handling word-piece tokenization and special tokens.
    """
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
                current_labels.append(-100) # Special tokens
            elif word_idx != previous_word_idx:
                current_labels.append(ner_tags_for_example[word_idx])
            else:
                current_labels.append(-100) # Subword tokens that are not the first part of a word
            previous_word_idx = word_idx
        labels.append(current_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def run_ner_inference(
    checkpoint_path: str,
    example_index: int = 0, # Index of the example from the test set
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """
    Runs NER inference on a single example using two different models:
    1. A model loaded from a checkpoint (e.g., ModernBERT-based for NER).
    2. The Cyber-ThreaD/SecureBERT-CyNER model.
    """
    print(f"--- Running NER Inference on Device: {device} ---")

    # 1. Load Dataset and Labels
    raw_dataset = load_dataset("bnsapa/cybersecurity-ner")
    label_list = raw_dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Select a single example from the test set
    # Note: The original eval script used max_seq_length = 1024
    MAX_SEQ_LENGTH = 1024 # Consistent with your eval script

    # --- Prepare the selected example ---
    # We need the raw tokens and ner_tags for the ground truth display
    selected_example_raw = raw_dataset["test"][example_index]
    original_tokens = selected_example_raw["tokens"]
    true_ner_tags_ids = selected_example_raw["ner_tags"]
    true_ner_tags_labels = [id2label[tag_id] for tag_id in true_ner_tags_ids]

    print(f"\n--- Selected Example (Index: {example_index}) ---")
    print("Original Text (Tokens):")
    print(original_tokens)
    print("\nGround Truth NER Tags:")
    print(true_ner_tags_labels)
    print("-" * 50)

    # --- Model 1: ModernBERT from Checkpoint ---
    print("\n--- Model 1: ModernBERT (from Checkpoint) ---")
    try:
        modernbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        modernbert_model = AutoModelForTokenClassification.from_pretrained(
            "answerdotai/ModernBERT-base",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            attn_implementation="sdpa",
        )

        # Load checkpoint
        print(f"Attempting to load checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        modernbert_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        modernbert_model.to(device)
        modernbert_model.eval()
        print("Checkpoint loaded successfully.")

        # Tokenize and align labels for inference
        # We process it as a batch of one for consistency with the map function,
        # but the actual inference will be on a single item.
        processed_example_modernbert = tokenize_and_align_labels(
            {"tokens": [original_tokens], "ner_tags": [true_ner_tags_ids]},
            modernbert_tokenizer, label_list, MAX_SEQ_LENGTH
        )

        # Convert to PyTorch tensors for model input
        input_ids = torch.tensor(processed_example_modernbert["input_ids"][0]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(processed_example_modernbert["attention_mask"][0]).unsqueeze(0).to(device)
        word_ids_for_example = processed_example_modernbert.word_ids(batch_index=0) # Get word_ids for this example

        with torch.no_grad():
            outputs = modernbert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy() # Squeeze to remove batch dim

        # Map predictions back to original words, ignoring special tokens and subwords
        predicted_ner_labels_modernbert = []
        for i, (token_id, word_id) in enumerate(zip(input_ids[0].cpu().numpy(), word_ids_for_example)):
            if word_id is not None and word_id < len(original_tokens): # Ensure it corresponds to an original word
                # Check if this is the first subword token for the current word_id
                if i == 0 or word_ids_for_example[i-1] != word_id:
                    predicted_ner_labels_modernbert.append(id2label[predictions[i]])

        # Pad or truncate predictions to match original_tokens length for display
        # This is a heuristic for display, actual seqeval uses filtered lists.
        # For simplicity, we'll just show the aligned predictions.
        # If the number of predicted labels doesn't match original tokens, it's due to tokenization/alignment.
        # We'll display them side-by-side.
        print("\nPredicted NER Tags (ModernBERT):")
        # For better readability, align predictions to original words
        aligned_predictions_modernbert = ["O"] * len(original_tokens) # Initialize with 'O'
        current_word_idx = -1
        for i, word_id in enumerate(word_ids_for_example):
            if word_id is not None and word_id != current_word_idx:
                if word_id < len(original_tokens): # Ensure index is within bounds of original words
                    aligned_predictions_modernbert[word_id] = id2label[predictions[i]]
                current_word_idx = word_id

        # Print aligned predictions
        for token, gt_tag, pred_tag in zip(original_tokens, true_ner_tags_labels, aligned_predictions_modernbert):
            print(f"  Token: '{token}' | GT: {gt_tag:<15} | ModernBERT Pred: {pred_tag}")

    except Exception as e:
        print(f"Error with ModernBERT model from checkpoint: {e}")
        print(f"Please ensure '{checkpoint_path}' is correct and the model architecture matches 'answerdotai/ModernBERT-base'.")
        aligned_predictions_modernbert = None

    print("-" * 50)

    # --- Model 2: SecureBERT-CyNER ---
    print("\n--- Model 2: SecureBERT-CyNER ---")
    try:
        securebert_tokenizer = AutoTokenizer.from_pretrained("Cyber-ThreaD/SecureBERT-CyNER")
        securebert_model = AutoModelForTokenClassification.from_pretrained("Cyber-ThreaD/SecureBERT-CyNER")
        securebert_model.to(device)
        securebert_model.eval()

        # SecureBERT-CyNER should have id2label in its config
        # If not, we can use the one derived from the dataset
        if not hasattr(securebert_model.config, 'id2label') or not securebert_model.config.id2label:
            securebert_model.config.id2label = id2label
            securebert_model.config.label2id = label2id

        # Tokenize the example for SecureBERT-CyNER
        processed_example_securebert = tokenize_and_align_labels(
            {"tokens": [original_tokens], "ner_tags": [true_ner_tags_ids]},
            securebert_tokenizer, label_list, MAX_SEQ_LENGTH
        )

        input_ids = torch.tensor(processed_example_securebert["input_ids"][0]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(processed_example_securebert["attention_mask"][0]).unsqueeze(0).to(device)
        word_ids_for_example = processed_example_securebert.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = securebert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Map predictions back to original words
        aligned_predictions_securebert = ["O"] * len(original_tokens)
        current_word_idx = -1
        for i, word_id in enumerate(word_ids_for_example):
            if word_id is not None and word_id != current_word_idx:
                if word_id < len(original_tokens):
                    aligned_predictions_securebert[word_id] = securebert_model.config.id2label[predictions[i]]
                current_word_idx = word_id

        print("\nPredicted NER Tags (SecureBERT-CyNER):")
        for token, gt_tag, pred_tag in zip(original_tokens, true_ner_tags_labels, aligned_predictions_securebert):
            print(f"  Token: '{token}' | GT: {gt_tag:<15} | SecureBERT-CyNER Pred: {pred_tag}")

    except Exception as e:
        print(f"Error with SecureBERT-CyNER model: {e}")
        aligned_predictions_securebert = None

    print("-" * 50)

# --- Barcoded Example Usage ---
if __name__ == "__main__":
    CKPT_PATH = "final_ner_original/take_2/checkpoint_epoch_20.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # You can change this index to try different examples from the test set
    # For example, try 1, 5, 10, 20, etc.
    EXAMPLE_TO_TEST_INDEX = 107

    run_ner_inference(
        checkpoint_path=CKPT_PATH,
        example_index=EXAMPLE_TO_TEST_INDEX,
        device=DEVICE
    )
 
