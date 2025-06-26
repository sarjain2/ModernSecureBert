from datasets import Dataset
import re
import glob
import os
import tqdm
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
import json

class ModernBertDataset():
    def __init__(self, primus_seed_pth, primus_fineweb_dir, primus_instruct_pth):
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        seed_dataset = Dataset.from_file(primus_seed_pth)
        primus_fineweb_pths = glob.glob(os.path.join(primus_fineweb_dir, "*.arrow"))
        fineweb_datasets = [Dataset.from_file(pth) for pth in primus_fineweb_pths]
        instruct_datasets = Dataset.from_file(primus_instruct_pth)
        ds_reasoning_datasets = load_from_disk("Primus-Reasoning_dataset")
        cybergpt_dataset = load_from_disk("CyberGPT_dataset")
        code_vul_dpo_dataset = load_from_disk("Code_Vulnerability_Security_DPO_dataset")
        sbt_txt_files = glob.glob("/teamspace/studios/this_studio/secure_modern_bert/sbt_data/*.txt")
        self.txt_data = list()
        # Load Secure Bert data:
        for file_path in tqdm.tqdm(sbt_txt_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm.tqdm(f):
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        info = data.get("text", "")
                    self.txt_data.append(info)
        # Load Primus seed dataset 
        for idx in tqdm.tqdm(range(len(seed_dataset))):
            example = self.clean_text(seed_dataset[idx]["content"])
            self.txt_data.append(example)
        # Load all the fineweb datasets
        for d_idx in range(len(fineweb_datasets)):
            for idx in tqdm.tqdm(range(len(fineweb_datasets[d_idx]))):
                example = self.clean_text(fineweb_datasets[d_idx][idx]["content"])
                self.txt_data.append(example)
        # # Load the instruct data right now
        for idx in tqdm.tqdm(range(len(instruct_datasets))):
            example = self.clean_text(instruct_datasets[idx]["messages"][0]["content"])
            self.txt_data.append(example)
        # Load the reasoning data right now
        for idx in tqdm.tqdm(range(len(ds_reasoning_datasets["train"]))):
            example = self.clean_text(ds_reasoning_datasets["train"][idx]["messages"][0]["content"] + " " + ds_reasoning_datasets["train"][idx]["messages"][1]["content"])
            self.txt_data.append(example)
        # Load the CyberGPT dataset
        for idx in tqdm.tqdm(range(len(cybergpt_dataset["train"]))):
            example = self.clean_text(str(cybergpt_dataset['train'][idx]))
            self.txt_data.append(example)
        # Load the Code Vulnerability Dataset
        for idx in tqdm.tqdm(range(len(code_vul_dpo_dataset["train"]))):
            example = self.clean_text(code_vul_dpo_dataset['train'][idx]["chosen"])
            self.txt_data.append(example)

    def clean_text(self, text):
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  
        text = re.sub(r"\*(.*?)\*", r"\1", text)      
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\n+", "\n", text) 
        text = re.sub(r"\s{2,}", " ", text)  
        text = text.strip()  
        return text
    def __getitem__(self, idx):
        curr_text = self.txt_data[idx]
        encoded = self.tokenizer(
            curr_text,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        # Convert tensors to lists for compatibility with DataCollator
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        # return curr_text
    def __len__(self):
        return len(self.txt_data)

if __name__ == "__main__":
    primus_seed_pth = "Primus-Seed_dataset/train/data-00000-of-00001.arrow"
    primus_fineweb_dir = "Primus-FineWeb_dataset/train"
    primus_instruct_pth = "Primus-Instruct_dataset/train/data-00000-of-00001.arrow"
    # primus_reasoning_pth = "Primus-Reasoning_dataset/train/data-00000-of-00001.arrow"
    df = ModernBertDataset(primus_seed_pth, primus_fineweb_dir, primus_instruct_pth)
    print(len(df))
    print(df[106])

## some docs about the Hugging Face:
# encoded = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_seq_length,
#             return_tensors="pt", 
#         )