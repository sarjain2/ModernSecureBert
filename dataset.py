from datasets import Dataset
import re
import glob
import os
import tqdm
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
import json
import torch
from datasets import load_dataset
import random
from sentence_transformers import SentenceTransformer, InputExample, models, losses

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


class ContrastiveLearningDataset():
    def __init__(self):
        self.txt_data = list()
        infosec = load_from_disk("Infosec_dataset")
        heimdall_dataset = load_from_disk("heimdall_dataset")
        tiamz_dataset = load_from_disk("Tiamz_dataset")
        for idx in tqdm.tqdm(range(len(infosec['train']))):
            self.txt_data.append((infosec['train'][idx]['question'], infosec['train'][idx]['answer']))
        for idx in tqdm.tqdm(range(len(heimdall_dataset['train']))):
            self.txt_data.append((heimdall_dataset['train'][idx]['user'], heimdall_dataset['train'][idx]['assistant']))
        for idx in tqdm.tqdm(range(len(tiamz_dataset['train']))):
            self.txt_data.append((tiamz_dataset['train'][idx]['instruction'], tiamz_dataset['train'][idx]['answer']))
    def __getitem__(self, idx):
        curr_txt = self.txt_data[idx]
        return curr_txt[0], curr_txt[1]
    def __len__(self):
        return len(self.txt_data)

class Mrr_ContrastiveLearningDataset():
    def __init__(self):
        self.txt_data = list()
        infosec = load_from_disk("Infosec_dataset")
        heimdall_dataset = load_from_disk("heimdall_dataset")
        tiamz_dataset = load_from_disk("Tiamz_dataset")
        for idx in tqdm.tqdm(range(len(infosec['train']))):
            self.txt_data.append((infosec['train'][idx]['question'], infosec['train'][idx]['answer']))
        for idx in tqdm.tqdm(range(len(heimdall_dataset['train']))):
            self.txt_data.append((heimdall_dataset['train'][idx]['user'], heimdall_dataset['train'][idx]['assistant']))
        for idx in tqdm.tqdm(range(len(tiamz_dataset['train']))):
            self.txt_data.append((tiamz_dataset['train'][idx]['instruction'], tiamz_dataset['train'][idx]['answer']))
        
        self.top = len(self.txt_data)
    def __getitem__(self, idx):
        curr_txt = self.txt_data[idx]
        return InputExample(texts=[curr_txt[0], curr_txt[1]])
        # ran = list(range(0, self.top))
        # ran.remove(idx)
        # neg_idx = random.choice(ran)
        # _, neg = self.txt_data[neg_idx]
        # # anchor, positive, negative
        # return curr_txt[0], curr_txt[1], neg 
    def __len__(self):
        return len(self.txt_data)

class NerDataset():
    def __init__(self, mode = "train"):
        self.txt_data = list()
        self.ner_tags = list()
        ner_dataset = load_from_disk("NER_dataset")
        second_ner_dataset = load_from_disk("augmented_ner_dataset")

        # raw_dataset = load_dataset("PranavaKailash/CyNER2.0_augmented_dataset")
        # self.num_labels = raw_dataset["train"].features["ner_tags"].feature.num_classes
        # label_list = raw_dataset["train"].features["ner_tags"].feature.names
        # print(label_list)

        # raw_dataset = load_dataset("bnsapa/cybersecurity-ner")
        # self.num_labels = raw_dataset["train"].features["ner_tags"].feature.num_classes
        # label_list = raw_dataset["train"].features["ner_tags"].feature.names
        # print(label_list)
        # assert 1 == 2

        second_to_first = {0: 10, 1: 0, 2: 0, 3:1, 4: 2, 5: 3, 6: 10, 7: 4, 8: 10, 9: 5, 10: 10, 11 : 6, 12: 7, 13 : 8, 14: 10, 15: 9, 16: 10}

        if mode == "train":
            for idx in tqdm.tqdm(range(len(ner_dataset["train"]))):
                self.txt_data.append(ner_dataset['train'][idx]['tokens'])
                self.ner_tags.append(ner_dataset['train'][idx]['ner_tags'])
            # This is where do the mapping
            for idx in tqdm.tqdm(range(len(second_ner_dataset["train"]))):
                self.txt_data.append(second_ner_dataset['train'][idx]['tokens'])
                nt = second_ner_dataset['train'][idx]['ner_tags']
                nt = [second_to_first[n] for n in nt]
                self.ner_tags.append(nt)
            for idx in tqdm.tqdm(range(len(second_ner_dataset["validation"]))):
                self.txt_data.append(second_ner_dataset['validation'][idx]['tokens'])
                nt = second_ner_dataset['validation'][idx]['ner_tags']
                nt = [second_to_first[n] for n in nt]
                self.ner_tags.append(nt)
            # for idx in tqdm.tqdm(range(len(second_ner_dataset["test"]))):
            #     self.txt_data.append(second_ner_dataset['test'][idx]['tokens'])
            #     nt = second_ner_dataset['test'][idx]['ner_tags']
            #     nt = [second_to_first[n] for n in nt]
            #     self.ner_tags.append(nt)
            raw_dataset = load_dataset("bnsapa/cybersecurity-ner")
            self.num_labels = raw_dataset["train"].features["ner_tags"].feature.num_classes
            label_list = raw_dataset["train"].features["ner_tags"].feature.names
            label2id = {label: i for i, label in enumerate(label_list)}
        else:
            for idx in tqdm.tqdm(range(len(ner_dataset["test"]))):
                self.txt_data.append(ner_dataset['test'][idx]['tokens'])
                self.ner_tags.append(ner_dataset['test'][idx]['ner_tags'])
            raw_dataset = load_dataset("bnsapa/cybersecurity-ner")
            self.num_labels = raw_dataset["test"].features["ner_tags"].feature.num_classes
            label_list = raw_dataset["train"].features["ner_tags"].feature.names
            label2id = {label: i for i, label in enumerate(label_list)}
        # text_data = open("train.txt", "r")
        # curr_ner_tags = list()
        # curr_txt_data = list()
        # for line in text_data:
        #     line_split = line.split()
        #     curr_txt_data.append(line_split[0])
        #     curr_id = label2id[line_split[1]]
        #     curr_ner_tags.append(curr_id)
        #     if line_split[0] == ".":
        #         self.txt_data.append(curr_txt_data)
        #         self.ner_tags.append(curr_ner_tags)
        #         curr_txt_data = list()
        #         curr_ner_tags = list()
        #         print(self.txt_data[-1])
        #         print(self.ner_tags[-1])
        #         assert 1 == 2
        # assert 1 == 2

            # curr_ner_tags.append(line_split[1])


        assert len(self.ner_tags) == len(self.txt_data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",
            )
        self.tokenized_inputs = self.tokenizer(
            self.txt_data,
            is_split_into_words=True,
            truncation=True,
            max_length=1024, 
            # return_offsets_mapping=True # Not strictly needed for this method, but useful for debugging
        )
        self.labels = []
        # Iterate through each example in the batch
        for i, ner_tags_for_example in enumerate(self.ner_tags):
            word_ids = self.tokenized_inputs.word_ids(batch_index=i)
            current_labels = []
            previous_word_idx = None
            # Iterate through each token's word_id
            for word_idx in word_ids:
                if word_idx is None:
                    current_labels.append(-100)
                elif word_idx != previous_word_idx:
                    current_labels.append(ner_tags_for_example[word_idx])
                else:
                    current_labels.append(-100)
            previous_word_idx = word_idx
            self.labels.append(current_labels)
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_inputs['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.tokenized_inputs['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    def __len__(self):
        return len(self.txt_data)


# Create a Contrastive Evaluation Set
class Eval_ContrastiveDataset():
    def __init__(self):
        self.txt_data = list()
        cr = load_from_disk("cr_dataset")
        for idx in tqdm.tqdm(range(len(cr['train']))):
            self.txt_data.append((cr['train'][idx]['instruction'], cr['train'][idx]['output']))
    def __getitem__(self, idx):
        curr_txt = self.txt_data[idx]
        return curr_txt[0], curr_txt[1]
    def __len__(self):
        return len(self.txt_data)

class SentimentVulnerabilityDataset():
    def __init__(self):
        self.txt_data = list()
        vuln = load_from_disk("vuln_dataset")
        for idx in tqdm.tqdm(range(len(vuln['train']))):
            self.txt_data.append((vuln['train'][idx]['normalized_func'], vuln['train'][idx]['target']))
    def __getitem__(self, idx):
        curr_txt = self.txt_data[idx]
        return curr_txt[0], curr_txt[1]
    def __len__(self):
        return len(self.txt_data)

class Eval_SentimentVulnerabilityDataset():
    def __init__(self):
        self.txt_data = list()
        vuln = load_from_disk("vuln_dataset")
        for idx in tqdm.tqdm(range(len(vuln['test']))):
            self.txt_data.append((vuln['test'][idx]['normalized_func'], vuln['test'][idx]['target']))
    def __getitem__(self, idx):
        curr_txt = self.txt_data[idx]
        return curr_txt[0], curr_txt[1]
    def __len__(self):
        return len(self.txt_data)

# @dataclass
# class FlowRecord:
#     id: int
#     dur: float
#     proto: str
#     service: str
#     state: str
#     spkts: int
#     dpkts: int
#     sbytes: int
#     dbytes: int
#     rate: float
#     sttl: int
#     dttl: int
#     sload: float
#     dload: float
#     sloss: int
#     dloss: int
#     sinpkt: float
#     dinpkt: float
#     sjit: float
#     djit: float
#     swin: int
#     stcpb: float
#     dtcpb: float
#     dwin: int
#     tcprtt: float
#     synack: float
#     ackdat: float
#     smean: int
#     dmean: int
#     trans_depth: int
#     response_body_len: int
#     ct_srv_src: int
#     ct_state_ttl: int
#     ct_dst_ltm: int
#     ct_src_dport_ltm: int
#     ct_dst_sport_ltm: int
#     ct_dst_src_ltm: int
#     is_ftp_login: int
#     ct_ftp_cmd: int
#     ct_flw_http_mthd: int
#     ct_src_ltm: int
#     ct_srv_dst: int
#     is_sm_ips_ports: int
#     gt: str    # attack_cat

# class FlowTemplateDataset(Dataset):
#     template = (
#         "proto: {proto} | service: {service} | state: {state} | spkts: {spkts} | "
#         "dpkts: {dpkts} | sbytes: {sbytes} | dbytes: {dbytes} | rate: {rate} | sttl: {sttl} | "
#         "dttl: {dttl} | sload: {sload} | dload: {dload} | sloss: {sloss} | dloss: {dloss} | "
#         "sinpkt: {sinpkt} | dinpkt: {dinpkt} | sjit: {sjit} | djit: {djit} | swin: {swin} | "
#         "stcpb: {stcpb} | dtcpb: {dtcpb} | dwin: {dwin} | tcprtt: {tcprtt} | synack: {synack} | "
#         "ackdat: {ackdat} | smean: {smean} | dmean: {dmean} | trans_depth: {trans_depth} | "
#         "response_body_len: {response_body_len} | ct_srv_src: {ct_srv_src} | ct_state_ttl: {ct_state_ttl} | "
#         "ct_dst_ltm: {ct_dst_ltm} | ct_src_dport_ltm: {ct_src_dport_ltm} | ct_dst_sport_ltm: {ct_dst_sport_ltm} | "
#         "ct_dst_src_ltm: {ct_dst_src_ltm} | is_ftp_login: {is_ftp_login} | ct_ftp_cmd: {ct_ftp_cmd} | "
#         "ct_flw_http_mthd: {ct_flw_http_mthd} | ct_src_ltm: {ct_src_ltm} | ct_srv_dst: {ct_srv_dst} | "
#         "is_sm_ips_ports: {is_sm_ips_ports}"
#     )

#     field_names = [
#         "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
#         "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
#         "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
#         "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
#         "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
#         "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label"
#     ]

#     # All template fields except the last two
#     template_fields = field_names[2:-2]  # skips id, dur, then stops before attack_cat, label

#     def __init__(self, csv_path):
#         self.rows = []
#         with open(csv_path, newline='') as csvfile:
#             reader = csv.reader(csvfile)
#             header = next(reader)
#             for row in reader:
#                 self.rows.append(row)

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, idx):
#         row = self.rows[idx]
#         # Map field names to their values
#         record = dict(zip(self.field_names, row))
#         # Prepare the template dict for formatting
#         template_dict = {k: record[k] for k in self.template_fields}
#         # Compose template string
#         template_str = self.template.format(**template_dict)
#         # Use second last column as gt/label
#         gt = record["attack_cat"]
#         return template_str, gt




if __name__ == "__main__":
    primus_seed_pth = "Primus-Seed_dataset/train/data-00000-of-00001.arrow"
    primus_fineweb_dir = "Primus-FineWeb_dataset/train"
    primus_instruct_pth = "Primus-Instruct_dataset/train/data-00000-of-00001.arrow"
    # primus_reasoning_pth = "Primus-Reasoning_dataset/train/data-00000-of-00001.arrow"
    # df = ModernBertDataset(primus_seed_pth, primus_fineweb_dir, primus_instruct_pth)
    # print(len(df))
    # print(df[106])
    # df = Eval_ContrastiveDataset()
    # print(df[0])
    # df = NerDataset()
    df = SentimentVulnerabilityDataset()
    print(df[0])

## some docs about the Hugging Face:
# encoded = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_seq_length,
#             return_tensors="pt", 
#         )