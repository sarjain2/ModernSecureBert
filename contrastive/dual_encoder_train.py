from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData, 
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from dataset import Mrr_ContrastiveLearningDataset
##
import math, random, torch, tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from transformers import AutoTokenizer, AutoModel   # <- use base model, not *ForMaskedLM
from transformers import AutoModelForMaskedLM



device = "cuda:0"
word_emb = models.Transformer(
    "answerdotai/ModernBERT-base",
    max_seq_length=512
)
# word_emb = AutoModelForMaskedLM.from_pretrained(
#         "answerdotai/ModernBERT-base",
#         attn_implementation="sdpa"
#     )
# word_emb = AutoModel.from_pretrained("answerdotai/ModernBERT-base", attn_implementation="sdpa")
sd = torch.load(
    "final_base_modernsecurebert_pths/checkpoint_epoch_20.pth",
    map_location=device
)
new_state_dict = {k.replace("model.", ""): v for k, v in sd["model_state_dict"].items()}
word_emb.auto_model.load_state_dict(new_state_dict, strict = False)
pooling = models.Pooling(
    word_emb.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
model = SentenceTransformer(modules=[word_emb, pooling]).to(device)


# 3. Load a train dataset
train_dataset = Mrr_ContrastiveLearningDataset()
print("Length of train dataset: {}".format(len(train_dataset)))
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,     
    shuffle=True,      
    drop_last=True,     
    num_workers=4,      
    pin_memory=True, 
    collate_fn=model.smart_batching_collate   
)
loss = MultipleNegativesRankingLoss(model)

num_epochs   = 3
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="modernbert_dual_encoder_mnr",
    use_amp=True,                 # fp16 if GPU ‑‒> speed/memory
    show_progress_bar=True
)

model.save("final_2")


