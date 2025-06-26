from datasets import load_dataset
import re
import matplotlib.pyplot as plt
import tqdm
from collections import Counter

# Login using e.g. `huggingface-cli login` to access this dataset
ds_seed = load_dataset("trendmicro-ailab/Primus-Seed", "default")
ds_fineweb = load_dataset("trendmicro-ailab/Primus-FineWeb")
ds_reasoning = load_dataset("trendmicro-ailab/Primus-Reasoning")
ds_instruct = load_dataset("trendmicro-ailab/Primus-Instruct")
ds_dpo = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
ds_gpt = load_dataset("Nitral-AI/Cybersecurity-ShareGPT")


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
    ("Primus-Seed", ds_seed),
    ("Primus-FineWeb", ds_fineweb),
    ("Primus-Reasoning", ds_reasoning),
    ("Primus-Instruct", ds_instruct),
    ("Code_Vulnerability_Security_DPO", ds_dpo)
]

datasets = [
    ("Primus-Seed", ds_seed),
    ("Primus-FineWeb", ds_fineweb),
    ("Primus-Reasoning", ds_reasoning),
    ("Primus-Instruct", ds_instruct),
    ("Code_Vulnerability_Security_DPO", ds_dpo),
    ("CyberGPT", ds_gpt)
]

def plot_text_length_distribution(dataset, split="train", gpt = None, dpo = None):  # Added new function
    """
    Calculate the length of the text content and plot its distribution.
    """
    lengths = []
    if split in dataset:  
        for example in tqdm.tqdm(dataset[split]):
            if gpt:
                text = clean_text(str(example["conversations"]))
            elif dpo:
                text = clean_text(example["chosen"])
            else:
                try: 
                    text = clean_text(example["content"])
                except:
                    text = clean_text(example["messages"][0]["content"])
            lengths.append(len(text.split())) 
    else:
        first_split = list(dataset.keys())[0]
        for example in dataset[first_split]:
            try:
                text = clean_text(example["content"])
            except:
                text = clean_text(example["messages"][0]["content"])
            lengths.append(len(text.split()))  
    print(f"Number of examples: {len(lengths)}")
    print(f"Average length: {sum(lengths) / len(lengths):.2f} words")
    print(f"Minimum length: {min(lengths)} words")
    print(f"Maximum length: {max(lengths)} words")
    plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    plt.xlabel("Text Length (words)")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution")
    plt.show(block=True)

CODE_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)
def count_code_percentages(dataset, gpt = None, dpo = None):
    chars = Counter(code = 0, text = 0)
    for example in tqdm.tqdm(dataset["train"]):
        if gpt:
            text = str(example["conversations"])
        else:
            try: 
                text = example["content"]
            except:
                text = example["messages"][0]["content"]
        pos = 0
        for m in CODE_BLOCK.finditer(text):
            start, end = m.span()
            chars["text"] += start - pos
            chars["code"] += end - start
            pos = end
        chars["text"] += len(text) - pos
    total = chars["code"] + chars["text"] or 1
    pct = {k: 100.0 * v / total for k , v in chars.items()}
    return dict(chars), pct
        

for name, dataset in datasets:
    print(f"\n--- {name} ---")
    # plot_text_length_distribution(dataset, split="train") 
    dataset.save_to_disk(f"{name}_dataset")
    
    # if "train" in dataset:
    #     example = dataset["train"][1]
    #     text = clean_text(example["content"])
    # else:
    #     first_split = list(dataset.keys())[0]
    #     example = dataset[first_split][0]
    #     text = clean_text(example["content"])
    # print(text)
# print("Primus Seed: ")
# print(count_code_percentages(ds_seed))
# print("Primus FineWeb: ")
# print(count_code_percentages(ds_fineweb))
# print("Primus Reasoning")
# print(count_code_percentages(ds_reasoning))
# print("Primus Instruct")
# print(count_code_percentages(ds_instruct))
# print("CyberSecurity ShareGPT")
# print(count_code_percentages(ds_gpt, gpt = True))

# plot_text_length_distribution(ds_gpt, gpt = True)

# print("Code Vuln DPO")

# plot_text_length_distribution(ds_dpo, dpo = True)