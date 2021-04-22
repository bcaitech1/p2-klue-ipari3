import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import re
from glob import glob
from pathlib import Path

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def increment_path(path, infix="", name=""):
    path += infix
    dirs = glob(f"{path}*")
    stem = Path(path).stem
    matches = [re.search(rf"{stem}(\d+)", d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 1
    return "".join([path, f"{n}_", name])

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average="micro")

    metrics = {
        "accuracy": acc,
        "f1": f1,
        # "precision": precision,
        # "recall": recall,
        # "support": support
    }
    return metrics