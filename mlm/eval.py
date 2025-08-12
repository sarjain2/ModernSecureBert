import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import time
from torch.quantization import quantize_dynamic
import argparse


# https://huggingface.co/ehsanaghaei/SecureBERT?library=transformers
#from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
# model = AutoModelForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")

def evaluate_model_on_dataset(
    model_path, dataset_path, top_n=5, device="cuda:0"
):
    """
    Evaluate the model on a dataset with <mask> tokens.

    Parameters:
        model_path (str): Path to the fine-tuned model checkpoint.
        dataset_path (str): Path to the CSV dataset.
        top_n (int): Top N predictions to consider for evaluation.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        accuracy (float): Top-N accuracy over the dataset.
        results (list): List of dictionaries containing evaluation details per row.
    """

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

    # # Load fine-tuned weights
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    # model = quantize_dynamic(
    #     model,  # Model to quantize
    #     {torch.nn.Linear},  # Specify the layers to quantize (e.g., Linear layers)
    #     dtype=torch.qint8  # Quantization data type (8-bit integers)
    # )

    # tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
    # model = AutoModelForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")


    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Evaluation loop
    correct_predictions = 0
    total_predictions = 0
    results = []
    total_time = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["sentence"]
        ground_truth = row["y"]
        # possible_answers = eval(row["y_"])  # Convert string to list
        sentence = sentence.replace("<mask>", "[MASK]")
        # print(sentence)

        # Tokenize input with <mask>
        inputs = tokenizer(
            sentence, return_tensors="pt", add_special_tokens=True
        ).to(device)

        # Find the index of <mask> token
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        end_time = time.time() - start_time
        total_time += end_time

        # Get top N predicted tokens
        try:
            top_n_token_ids = torch.topk(
                predictions[0, mask_token_index], top_n, dim=-1
            ).indices[0].tolist()
        except:
            continue
        top_n_tokens = [tokenizer.decode(token_id).strip() for token_id in top_n_token_ids]

        # print(possible_answers)
        # print(top_n_tokens)
        # assert 1 == 2

        # Check if any of the top N predictions are in the possible answers
        ans = [ground_truth]
        is_correct = any(token in ans for token in top_n_tokens)

        # Update stats
        correct_predictions += int(is_correct)
        total_predictions += 1

        # Save the result for this row
        results.append({
            "sentence": sentence,
            "ground_truth": ground_truth,
            # "possible_answers": possible_answers,
            "top_n_predictions": top_n_tokens,
            "is_correct": is_correct
        })

    # Calculate Top-N Accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print("Total time: {}".format(total_time / len(df)))
    return accuracy, results


if __name__ == "__main__":
    # Path to the model checkpoint
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

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    model_checkpoint_path = args.model_path
    dataset_path = args.dataset_path
    top_n = args.top_n
    device = args.device

    # model_checkpoint_path = "final_base_modernsecurebert_pths/checkpoint_epoch_20.pth"

    # Path to the dataset CSV file
    #dataset_path = "Verb_prediction.csv"
    # dataset_path = "Object_prediction.csv"
    # dataset_path = "SecureBERT-eval-verbs.csv"

    # Number of top predictions to evaluate
    # top_n = 4

    # Device to use for evaluation (e.g., "cuda" or "cpu")
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate the model
    accuracy, results = evaluate_model_on_dataset(
        model_path=model_checkpoint_path,
        dataset_path=dataset_path,
        top_n=top_n,
        device=device
    )

    # Print Top-N Accuracy
    print(f"Top-{top_n} Accuracy: {accuracy:.2%}")

    # Save results to a CSV for inspection
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")