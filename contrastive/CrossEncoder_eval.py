import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dataset import ContrastiveLearningDataset, Eval_ContrastiveDataset
from sentence_transformers import SentenceTransformer


device = "cuda"

def cont_collate_fn(batch):
    text_1, text_2 = zip(*batch)
    return list(text_1), list(text_2)

def evaluate_model(model, tokenizer, dataset, max_seq_length=1024, batch_size=8, device="cuda"):
    model.eval()
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = cont_collate_fn)
    question_embeddings = []
    answer_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            # print(batch)
            # assert 1 == 2
            # questions, answers = zip(*batch)
            questions, answers = batch
            encoded_questions = tokenizer(
                list(questions),
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            ).to(device)
            encoded_answers = tokenizer(
                list(answers),
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            ).to(device)
            question_hs = model(**encoded_questions, output_hidden_states=True).hidden_states[-1][:, 0, :]
            answer_hs = model(**encoded_answers, output_hidden_states=True).hidden_states[-1][:, 0, :]
            question_embeddings.append(torch.nn.functional.normalize(question_hs, dim=-1))
            answer_embeddings.append(torch.nn.functional.normalize(answer_hs, dim=-1))
    question_embeddings = torch.cat(question_embeddings, dim=0)  # Shape: [num_samples, hidden_dim]
    answer_embeddings = torch.cat(answer_embeddings, dim=0)      # Shape: [num_samples, hidden_dim]
    similarity_matrix = torch.matmul(question_embeddings, answer_embeddings.T)  # Shape: [num_samples, num_samples]
    num_samples = similarity_matrix.size(0)
    recall_at_1 = 0
    avg_precisions = []
    for i in tqdm(range(num_samples)):
        # Get similarity scores for current question
        scores = similarity_matrix[i]
        ranked_indices = torch.argsort(scores, descending=True)
        if ranked_indices[0] == i:  
            recall_at_1 += 1
        correct_index = i  # The correct answer index
        rank = (ranked_indices == correct_index).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
        avg_precision = 1.0 / rank  # Precision at the correct rank
        avg_precisions.append(avg_precision)
    recall_at_1 = recall_at_1 / num_samples
    mean_avg_precision = sum(avg_precisions) / num_samples

    return recall_at_1, mean_avg_precision


if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa")
    # model_path = "checkpoint_epoch_9.pth"
    # model_path = "final_base_modernsecurebert_pths/checkpoint_epoch_20.pth"
    model_path = "embedding_sim_weights/checkpoint_epoch_9.pth"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Loading Attack Bert
    # model = SentenceTransformer('basel/ATTACK-BERT')

    # Load dataset
    dataset = Eval_ContrastiveDataset()

    # Evaluate model
    recall_at_1, mean_avg_precision = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device="cuda"
    )

    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"mAP: {mean_avg_precision:.4f}")