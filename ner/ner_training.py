from dataset import ModernBertDataset, ContrastiveLearningDataset, NerDataset
import torch
import tqdm
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling, DataCollatorForTokenClassification
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

fabric = Fabric(accelerator="gpu", devices=1, strategy=DDPStrategy(find_unused_parameters=True))



def train_model(checkpoint_path):
    fabric.launch()
    # torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # if fabric.is_global_zero:  # Only the main process writes to the log file
    log_file = open("ner_training_log.txt", "w")
    # Paths to your datasets
    df = NerDataset()
    rng = random.Random()
    # Hyperparameters
    max_seq_length = 1024
    batch_size_per_gpu = 8  # Adjust based on memory usage
    print(f"Using per-GPU batch size: {batch_size_per_gpu}")
    # print(f"Total effective batch size: {total_batch_size}")
    num_epochs = 20
    learning_rate = 1e-5
    weight_decay = 0.01
    mlm_prob = 0.15
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device(f"cuda:{rank}")
    # Model init
    tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base",
    )
    # model = AutoModelForMaskedLM.from_pretrained(
    #     "answerdotai/ModernBERT-base",
    #     attn_implementation="sdpa",
    #     num_labels=df.num_labels
    # )
    model = AutoModelForTokenClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        attn_implementation="sdpa",
        num_labels=df.num_labels)
    base_pth = "final_base_modernsecurebert_pths/checkpoint_epoch_20.pth"
    checkpoint = torch.load(base_pth, map_location=fabric.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict = False)
    # model = DDP(model, device_ids=[rank], output_device=rank)
    # Use DistributedSampler for the DataLoader
    # sampler = DistributedSampler(df, num_replicas=world_size, rank=rank, shuffle=True)
    sampler = DistributedSampler(df, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        df, batch_size=batch_size_per_gpu, sampler=sampler, collate_fn = data_collator
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
    for epoch in range(num_epochs):
        epoch_loss = 0
        sampler.set_epoch(epoch)
        skipped = 0
        for batch in tqdm.tqdm(dataloader, disable=not fabric.is_global_zero):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss  
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
    train_model(checkpoint_path = "final_base_modernsecurebert_pths/checkpoint_epoch_20.pth")
    # world_size = 4  # Number of GPUs
    # torch.multiprocessing.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)