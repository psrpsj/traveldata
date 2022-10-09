import os
import pandas as pd
import torch
import wandb

from argument import TrainingCat1Arguments, TrainCat1ModelArguments
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed


def compute_meterics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    return {"accuracy": acc}


def train_cat1():
    parser = HfArgumentParser((TrainingCat1Arguments, TrainCat1ModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("### Training Model for Cat 1 ###")
    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    data = pd.read_csv(os.path.join(model_args.data_path, "train.csv"))
    desc = data["overview"]
    label = data["cat1"]
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    set_seed(training_args.seed)
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
