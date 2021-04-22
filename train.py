import os
import logging
import wandb
import pickle
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, integrations
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, ElectraForSequenceClassification
# from transformers import BertForSequenceClassification, BertConfig
# from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

from dataset import *
from utils import *


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def train(config, training_args):
    ## SETTINGS
    device = torch.device("cuda:0")
    seed = config["seed"]
    seed_everything(seed)

    data_path = config.get("data_path", None)
    model_name = config.get("model_name", None)
    Model = config.get("Model", None)
    Tokenizer = config.get("Tokenizer", None)

    ## DATASET
    train_dataset = load_data(data_path)
    train_label = train_dataset["label"].values

    # tokenization
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    #tokenized_eval = tokenized_dataset(valid_dataset, tokenizer)
  
    # tokenized dataset for pytorch
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_eval_dataset = RE_Dataset(tokenized_eval, train_label)

    # model
    model = Model.from_pretrained(model_name, num_labels=42)
    model.resize_token_embeddings(len(tokenizer)) # update vocab_size
    model.to(device)

    trainer = Trainer(
        model=model,            # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        # eval_dataset=RE_eval_dataset,             # evaluation dataset
        # compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()

def main():
    model_name = "monologg/koelectra-base-v3-discriminator" 
    # model_name = "xlm-roberta-base"#"bert-base-multilingual-cased"
    config = {
        "seed": 41,
        "data_path": "/opt/ml/input/data/train/train_aug.tsv",
        "model_name": model_name,
        "Model": ElectraForSequenceClassification, #ElectraForQuestionAnswering,
        "Tokenizer": ElectraTokenizer,
        # "Model": XLMRobertaForSequenceClassification,
        # "Tokenizer": XLMRobertaTokenizer,
    }

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    infix = "/exp"
    exp_name = "data_aug"
    training_args = TrainingArguments(
        output_dir=increment_path("./results", infix, exp_name),    # output directory
        save_total_limit=3,              # number of total save model.
        save_steps=500,                 # model saving step.
        save_strategy="epoch",
        learning_rate=5e-5,               # learning_rate
        num_train_epochs=10,              # total number of training epochs
        dataloader_num_workers=4,
        per_device_train_batch_size=16,  # batch size per device during training
        # per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=increment_path("./logs", infix, exp_name),  # directory for storing logs
        logging_steps=100,              # log saving step.
        # evaluation_strategy="steps", # evaluation strategy to adopt during training
        # eval_steps = 500,            # evaluation step.
    )

    wandb.init(project="Pstage2", group="single_model", name=exp_name, reinit=True)
    integrations.WandbCallback()
    integrations.TensorBoardCallback()

    train(config, training_args)

if __name__ == '__main__':
    main()