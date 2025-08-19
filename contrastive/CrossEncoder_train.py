from dataset import ModernBertDataset, ContrastiveLearningDataset
import torch
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import os
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
torch.set_float32_matmul_precision('high')

# traning command: torchrun --nproc_per_node=4 train.py
# torchrun --nproc_per_node=8 train.py (8 GPUs)
# https://github.com/CVEProject/cvelistV5 (Title, description)

# Make it 8 GPU's
# Train all data after epoch 3

fabric = Fabric(accelerator="gpu", devices=8, strategy=DDPStrategy(find_unused_parameters=True))

def cont_collate_fn(batch):
    text_1, text_2 = zip(*batch)
    return list(text_1), list(text_2)

def train_model(checkpoint_path):
    fabric.launch()
    # torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # if fabric.is_global_zero:  # Only the main process writes to the log file
    log_file = open("cont_training_log.txt", "w")
    # Paths to your datasets
    df = ContrastiveLearningDataset()
    rng = random.Random()
    # Hyperparameters
    max_seq_length = 1024
    batch_size_per_gpu = 8  # Adjust based on memory usage
    print(f"Using per-GPU batch size: {batch_size_per_gpu}")
    # print(f"Total effective batch size: {total_batch_size}")
    num_epochs = 10
    learning_rate = 1e-5
    weight_decay = 0.01
    mlm_prob = 0.15
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cuda:{rank}")
    # Model init
    tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base",
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "answerdotai/ModernBERT-base",
        attn_implementation="sdpa"
    )
    # model = DDP(model, device_ids=[rank], output_device=rank)
    # Use DistributedSampler for the DataLoader
    # sampler = DistributedSampler(df, num_replicas=world_size, rank=rank, shuffle=True)
    sampler = DistributedSampler(df, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        df, batch_size=batch_size_per_gpu, sampler=sampler, collate_fn = cont_collate_fn
    )
    num_training_steps = len(dataloader) * num_epochs
    dataloader = fabric.setup_dataloaders(dataloader)
    # dataloader = torch.utils.data.DataLoader(df, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    # 2 to 5 thousands warmup steeps if checkpoint resume
    print("Has no warmup steps")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    model.train()
    # Load base model
    checkpoint = torch.load(checkpoint_path, map_location=fabric.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for epoch in range(num_epochs):
        epoch_loss = 0
        sampler.set_epoch(epoch)
        skipped = 0
        for batch in tqdm.tqdm(dataloader, disable=not fabric.is_global_zero):
            optimizer.zero_grad()
            text_1, text_2 = batch
            try:
                encoded_text1 = tokenizer(
                    text_1,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt", 
                    add_special_tokens=True
                )
            except:
                skipped += 1
                continue
            try:
                encoded_text2 = tokenizer(
                    text_2,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt", 
                    add_special_tokens=True
                )
            except:
                skipped += 1
                continue
            # get hidden states
            token_1_hs = model(**encoded_text1, output_hidden_states=True)
            token_2_hs = model(**encoded_text2, output_hidden_states=True)
            embedding1 = token_1_hs.hidden_states[-1][:, 0, :] 
            embedding2 = token_2_hs.hidden_states[-1][:, 0, :]  
            embedding1 = torch.nn.functional.normalize(embedding1, dim=-1)
            embedding2 = torch.nn.functional.normalize(embedding2, dim=-1)
            similarity_matrix = torch.matmul(embedding1, embedding2.T) 
            temperature = 0.07
            logits = similarity_matrix / temperature
            batch_size = logits.shape[0]
            labels = torch.arange(batch_size).to(fabric.device)  # Ground truth: 0, 1, ..., batch_size-1
            # Compute contrastive loss (cross-entropy between logits and labels)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            

            fabric.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        if fabric.is_global_zero:
            log_message = f"Epoch Number: {epoch + 1} | Average Epoch Loss: {avg_epoch_loss} | Skipped: {skipped}\n"
            print(log_message.strip())  # Print to console
            log_file.write(log_message)  # Write to log file

            # save_path = f"model_epoch_{epoch + 1}.pth"
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": avg_epoch_loss,
            }, checkpoint_path)
            # torch.save(model.state_dict(), save_path)  # Save the model weights
            print(f"Model saved to {checkpoint_path}")
    log_file.close()
    # torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train_model(checkpoint_path = "no_secure_bert_data_internet_data_pths/checkpoint_epoch_10.pth")
    # world_size = 4  # Number of GPUs
    # torch.multiprocessing.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)