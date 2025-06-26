from dataset import ModernBertDataset
import torch
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import os
from lightning.fabric import Fabric
torch.set_float32_matmul_precision('high')

# traning command: torchrun --nproc_per_node=4 train.py
# torchrun --nproc_per_node=8 train.py (8 GPUs)
# https://github.com/CVEProject/cvelistV5 (Title, description)

# Make it 8 GPU's
# Train all data after epoch 3

fabric = Fabric(accelerator="gpu", devices=8, strategy="ddp")

def train_model(checkpoint_path = None):
    fabric.launch()
    # torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    # if fabric.is_global_zero:  # Only the main process writes to the log file
    log_file = open("training_log.txt", "w")
    # Paths to your datasets
    primus_seed_pth = "Primus-Seed_dataset/train/data-00000-of-00001.arrow"
    primus_fineweb_dir = "Primus-FineWeb_dataset/train"
    primus_instruct_pth = "Primus-Instruct_dataset/train/data-00000-of-00001.arrow"
    df = ModernBertDataset(primus_seed_pth, primus_fineweb_dir, primus_instruct_pth)
    rng = random.Random()
    # Hyperparameters
    max_seq_length = 1024
    batch_size_per_gpu = 16  # Adjust based on memory usage
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
    model = AutoModelForMaskedLM.from_pretrained(
        "answerdotai/ModernBERT-base",
        attn_implementation="sdpa"
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True, 
        mlm_probability=mlm_prob 
    )
    # model = DDP(model, device_ids=[rank], output_device=rank)
    # Use DistributedSampler for the DataLoader
    # sampler = DistributedSampler(df, num_replicas=world_size, rank=rank, shuffle=True)
    sampler = DistributedSampler(df, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        df, batch_size=batch_size_per_gpu, sampler=sampler, collate_fn=data_collator
    )
    num_training_steps = len(dataloader) * num_epochs
    dataloader = fabric.setup_dataloaders(dataloader)
    # dataloader = torch.utils.data.DataLoader(df, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    # 2 to 5 thousands warmup steeps if checkpoint resume
    print("Has warmup steps")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=3000, num_training_steps=num_training_steps)
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    model.train()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=fabric.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        sampler.set_epoch(epoch)
        for batch in tqdm.tqdm(dataloader, disable=not fabric.is_global_zero):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(fabric.device)
            attention_mask = batch["attention_mask"].to(fabric.device)
            labels = batch["labels"].to(fabric.device)
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = model_outputs.loss
            # loss.backward()
            fabric.backward(loss)
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        if fabric.is_global_zero:
            log_message = f"Epoch Number: {epoch + 1} | Average Epoch Loss: {avg_epoch_loss}\n"
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