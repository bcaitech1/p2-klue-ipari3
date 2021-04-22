import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset

# Dataset
class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        
        data = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        data["labels"] = self.labels[idx]
        return data
    
    def __len__(self):
        return len(self.labels)


# tokenization for bert
# tip! 다양한 종류의 tokenizer와 special token들을 시도를 해볼 수 있다.
def tokenized_dataset(dataset, tokenizer):
    special_tokens_dict = {'additional_special_tokens': ["[ENT]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    concat_entity = []
    for e01, e02 in zip(dataset["entity_01"], dataset["entity_02"]):
        concat_entity.append("".join([e01, "[ENT]", e02]))
    

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences

# Load tsv file
def load_data(dataset_dir):
    # Load
    with open("/opt/ml/input/data/label_type.pkl", "rb") as f:
        label_type = pickle.load(f)

    dataset = pd.read_csv(dataset_dir, delimiter="\t", header=None)

    for label, logit in label_type.items():
        if len(dataset[dataset[8] == label]) < 5:
            label_type[label] = 0
    
    # Preprocess
    dataset = preprocess_dataset(dataset, label_type)
    return dataset

# tsv(pickle) to dataframe(pandas)
def preprocess_dataset(dataset, label_type):
    label = []
    for rel in dataset[8]:
        if rel == "blind":
            label.append(100)
        else:
            label.append(label_type[rel])

    out_dataset = pd.DataFrame({
        "sentence": dataset[1],
        "entity_01": dataset[2],
        "entity_02": dataset[5],
        "label": label,
    })
    return out_dataset