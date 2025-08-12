from datasets import load_dataset
import re
import matplotlib.pyplot as plt
import tqdm
from collections import Counter

# Login using e.g. `huggingface-cli login` to access this dataset

# infosec = load_dataset("pAILabs/infosec-security-qa")
# tiamz = load_dataset("Tiamz/cybersecurity-instruction-dataset")
# heimdall = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1")
# cr = load_dataset("jcordon5/cybersecurity-rules")
# ner_2 = load_dataset("PranavaKailash/CyNER2.0_augmented_dataset")
vuln = load_dataset("DetectVul/devign")




def clean_text(text):
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

datasets = [
    # ("Infosec", infosec),
    # ("Tiamz", tiamz),
    # ("heimdall", heimdall)
    # ("cr", cr)
    # ("augmented_ner", ner_2)
    ("vuln", vuln)
]

   

for name, dataset in datasets:
    print(f"\n--- {name} ---")
    # plot_text_length_distribution(dataset, split="train") 
    dataset.save_to_disk(f"{name}_dataset")