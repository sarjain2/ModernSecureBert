import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def run_similarity_inference(
    text1: str,
    text2: str,
    ckpt_path: str,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """
    Calculates and displays the cosine similarity between two texts using two different models:
    1. A custom model loaded from a checkpoint (e.g., ModernBERT-based for contrastive learning).
    2. The 'sentence-transformers/all-MiniLM-L6-v2' model.

    Args:
        text1 (str): The first input text.
        text2 (str): The second input text.
        ckpt_path (str): Path to the checkpoint file for the first model.
        device (torch.device): The device to run the models on (e.g., 'cuda:0' or 'cpu').
    """
    print(f"--- Running Similarity Inference on Device: {device} ---")
    print(f"\nText 1: \"{text1}\"")
    print(f"Text 2: \"{text2}\"")
    print("-" * 50)

    # --- Model 1: ModernBERT (from Checkpoint) ---
    print("\n--- Model 1: ModernBERT (from Checkpoint) ---")
    modernbert_tokenizer = None
    modernbert_model = None
    try:
        modernbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        modernbert_model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base",
            attn_implementation="sdpa",
        )

        # Load checkpoint
        print(f"Attempting to load checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        modernbert_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        modernbert_model.to(device)
        modernbert_model.eval()
        print("Checkpoint loaded successfully.")

        MAX_SEQ_LENGTH = 1024 # As used in your eval script for ModernBERT

        # Encode texts
        encoded_text1 = modernbert_tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)
        encoded_text2 = modernbert_tokenizer(
            text2,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            # Extract [CLS] token embeddings from the last hidden state
            # hidden_states[-1] gets the last layer's hidden states
            # [:, 0, :] gets the [CLS] token (first token) embedding for all items in batch
            emb1_modernbert = modernbert_model(**encoded_text1, output_hidden_states=True).hidden_states[-1][:, 0, :]
            emb2_modernbert = modernbert_model(**encoded_text2, output_hidden_states=True).hidden_states[-1][:, 0, :]

            # Normalize embeddings
            emb1_modernbert = F.normalize(emb1_modernbert, p=2, dim=1)
            emb2_modernbert = F.normalize(emb2_modernbert, p=2, dim=1)

            # Calculate cosine similarity
            # Since embeddings are normalized, dot product is cosine similarity
            similarity_modernbert = torch.sum(emb1_modernbert * emb2_modernbert, dim=1).item()

        print(f"Similarity (ModernBERT from Checkpoint): {similarity_modernbert:.4f}")

    except Exception as e:
        print(f"Error with ModernBERT model from checkpoint: {e}")
        print(f"Please ensure '{ckpt_path}' is correct and the model architecture matches 'answerdotai/ModernBERT-base'.")
        similarity_modernbert = None
    finally:
        # Clean up GPU memory if models were loaded
        if modernbert_model:
            del modernbert_model
        if modernbert_tokenizer:
            del modernbert_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # --- Model 2: SentenceTransformer (all-MiniLM-L6-v2) ---
    print("\n--- Model 2: SentenceTransformer (all-MiniLM-L6-v2) ---")
    st_model = None
    try:
        # Using 'sentence-transformers/all-MiniLM-L6-v2' as it was the final choice in your eval script
        st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        st_model.to(device)
        st_model.eval()

        # Encode texts directly using SentenceTransformer's encode method
        # It typically handles tokenization, batching, and normalization internally
        emb1_st = st_model.encode(text1, convert_to_tensor=True, device=device)
        emb2_st = st_model.encode(text2, convert_to_tensor=True, device=device)

        # Ensure embeddings are normalized (SentenceTransformer usually returns normalized ones)
        emb1_st = F.normalize(emb1_st, p=2, dim=0) # dim=0 because it's a single vector
        emb2_st = F.normalize(emb2_st, p=2, dim=0)

        # Calculate cosine similarity
        similarity_st = torch.dot(emb1_st, emb2_st).item()

        print(f"Similarity (all-MiniLM-L6-v2): {similarity_st:.4f}")

    except Exception as e:
        print(f"Error with SentenceTransformer model: {e}")
        similarity_st = None
    finally:
        # Clean up GPU memory if models were loaded
        if st_model:
            del st_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60 + "\n") # Separator for next example

    return {
        "text1": text1,
        "text2": text2,
        "modernbert_similarity": similarity_modernbert,
        "sentence_transformer_similarity": similarity_st
    }

# --- Barcoded Example Usage ---
if __name__ == "__main__":
    CKPT_PATH = "embedding_sim_weights/checkpoint_epoch_9.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    text_1 = """
    Could you please provide a YARA rule that detects a specific malware variant, Kryptonv03, which has a unique entry point signature? The signature consists of the following bytes: 8B 0C 24 E9 C0 8D 01 ?? C1 3A 6E CA 5D 7E 79 6D B3 64 5A 71 EA. The rule should be designed to match this signature at the entry point of a PE file, and it should include metadata about the author for attribution purposes.
    """

    text_2 = """
    As part of our comprehensive cyber defense strategy, I have formulated a yara rule to protect your environment:

    ```



    rule Kryptonv03
    {
    meta:
    author="malware-lu"
    strings:
    $a0 = { 8B 0C 24 E9 C0 8D 01 ?? C1 3A 6E CA 5D 7E 79 6D B3 64 5A 71 EA }

    condition:
    $a0 at pe.entry_point
    }
    ```

    As a cybersecurity expert, I have generated a YARA rule to detect the Kryptonv03 malware variant based on its unique entry point signature. Here's an in-depth explanation of the rule:

    1. `rule Kryptonv03`: This line defines the name of the rule as "Kryptonv03". It is a standard naming convention for YARA rules, where each rule has a unique identifier to differentiate it from others.

    2. `meta:`: The `meta` block contains metadata about the rule. In this case, we have included the author's name as "malware-lu". This is important for attribution purposes and helps in tracking the origin of the rule.

    3. `strings:`: This section defines the strings that the rule will look for within a file. The string pattern `$a0` is defined with the hexadecimal bytes `8B 0C 24 E9 C0 8D 01 ?? C1 3A 6E CA 5D 7E 79 6D B3 64 5A 71 EA`. The `??` represents a wildcard that can match any byte value. This pattern is the unique entry point signature of Kryptonv03, which we are looking for in the file.

    4. `condition:`: This section specifies the conditions under which the rule will trigger an alert. In this case, it states that the string `$a0` must be found at the "entry_point" of a PE (Portable Executable) file. The entry point is the starting address of the code execution in a PE file, and by checking for the signature there, we can identify if the file contains Kryptonv03 malware.

    The rule is designed to be as specific as possible to minimize false positives while ensuring that it accurately detects the Kryptonv03 variant. The use of wildcards in the string pattern allows for some flexibility in matching byte values, which can be useful when dealing with polymorphic malware that may change its signature over time.

    Remember, this rule is just a starting point and should be used as part of a comprehensive security strategy that includes other detection methods and regular updates to the rule based on new information about Kryptonv03's behavior or evasion techniques.
    """

    # Example 1: Highly similar texts
    print("--- Example 1: Highly Similar Texts ---")
    results1 = run_similarity_inference(
        text1=text_1,
        text2=text_2,
        ckpt_path=CKPT_PATH,
        device=DEVICE
    )

    print("\n--- Summary of All Examples ---")
    print(f"Example 1: Our Model sim: {results1['modernbert_similarity']:.4f}, all MiniLM Sim: {results1['sentence_transformer_similarity']:.4f}")