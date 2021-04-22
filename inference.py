import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, integrations
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from dataset import *
from utils import *

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset["label"].values

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []
    
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
  
    return np.array(output_pred).flatten()

def main(config, args):
    """
        주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    # config
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model_name = config.get("model_name", None)
    Model = config.get("Model", None)
    Tokenizer = config.get("Tokenizer", None)

    # tokenizer
    tokenizer = Tokenizer.from_pretrained(model_name)

    # model
    model = Model.from_pretrained(args.model_dir)
    model.to(device)

    # datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # prediction
    pred_answer = inference(model, test_dataset, device)
    assert len(pred_answer) == len(test_dataset)

    output = pd.DataFrame(pred_answer, columns=["pred"])
    output.to_csv(args.output_dir, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    output_root = "./prediction"
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    ## *********** CHANGE SETTINGS HERE *********** ##
    # change config
    model_name = "monologg/koelectra-base-v3-discriminator"
    config = {
        "model_name": model_name,
        "Model": ElectraForSequenceClassification,
        "Tokenizer": ElectraTokenizer,
    }
    
    # change parser
    output_dir = increment_path("./prediction/submission", name="15epoch_to_13epoch") + ".csv"
    model_dir = "/opt/ml/code/results/exp15_token_and_minor_label/checkpoint-7319"
    ## ******************************************** ##

    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model_dir", type=str, default=model_dir)
    parser.add_argument("--output_dir", type=str, default=output_dir)

    args = parser.parse_args()
    print(args)
    main(config, args)
  
