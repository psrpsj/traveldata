import os
import pandas as pd
import torch
import wandb

from argument import (
    TrainingCat1NLPArguments,
    TrainCat1NLPModelArguments,
    TrainingCat2NLPArguments,
    TrainCat2NLPModelArguments,
    TrainingCat3NLPArguments,
    TrainCat3NLPModelArguments,
)
from dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from trainer import CustomTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from utils import label_to_num, preprocess_nlp


def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    f1 = f1_score(label, preds, average="weighted")
    return {"accuracy": acc, "f1_score": f1}


def train_cat1_nlp(data):
    parser = HfArgumentParser((TrainingCat1NLPArguments, TrainCat1NLPModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("### Training NLP Model for Cat 1 ###")
    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    set_seed(training_args.seed)
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    model_config.num_labels = 6
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    wandb.init(
        entity="psrpsj",
        project="traveldata",
        name=model_args.project_cat1_name,
        tags=[model_args.model_name],
    )
    wandb.config.update(training_args)

    train_dataset, valid_dataset = train_test_split(
        data, test_size=0.2, stratify=data["cat1"], random_state=42
    )
    train_dataset["cat1"] = preprocess_nlp(train_dataset["cat1"], 1)
    valid_dataset["cat1"] = preprocess_nlp(valid_dataset["cat1"], 1)
    train = CustomDataset(train_dataset["overview"], train_dataset["cat1"], tokenizer)
    valid = CustomDataset(valid_dataset["overview"], valid_dataset["cat1"], tokenizer)

    trainer = CustomTrainer(
        loss_name=model_args.loss_name,
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
    )

    print("--- CAT1 NLP TRAINING ---")
    trainer.train()
    model.save_pretrained(
        os.path.join(training_args.output_dir, model_args.project_cat1_name)
    )
    wandb.finish()
    print("--- CAT1 NLP TRAINING FINISH ---")


def train_cat2_nlp(data):
    parser = HfArgumentParser((TrainingCat2NLPArguments, TrainCat2NLPModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("### Training NLP Model for Cat 2 ###")
    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    special_tokens_dict = {"additional_special_tokens": ["[RELATION]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    set_seed(training_args.seed)
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    model_config.num_labels = 18
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    wandb.init(
        entity="psrpsj",
        project="traveldata",
        name=model_args.project_cat2_name,
        tags=[model_args.model_name],
    )
    wandb.config.update(training_args)

    train_dataset, valid_dataset = train_test_split(
        data, test_size=0.2, stratify=data["cat2"], random_state=42
    )
    train_dataset["overview"] = (
        train_dataset["overview"] + "[RELATION]" + train_dataset["cat1"]
    )
    valid_dataset["overview"] = (
        valid_dataset["overview"] + "[RELATION]" + valid_dataset["cat1"]
    )
    train_dataset["cat2"] = label_to_num(train_dataset["cat2"], 2)
    valid_dataset["cat2"] = label_to_num(valid_dataset["cat2"], 2)

    train = CustomDataset(train_dataset["overview"], train_dataset["cat2"], tokenizer)
    valid = CustomDataset(valid_dataset["overview"], valid_dataset["cat2"], tokenizer)

    trainer = CustomTrainer(
        loss_name=model_args.loss_name,
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
    )

    print("--- CAT2 NLP TRAINING ---")
    trainer.train()
    model.save_pretrained(
        os.path.join(training_args.output_dir, model_args.project_cat2_name)
    )
    wandb.finish()
    print("--- CAT2 NLP TRAINING FINISH ---")


def train_cat3_nlp(data):
    parser = HfArgumentParser((TrainingCat3NLPArguments, TrainCat3NLPModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("### Training NLP Model for Cat 3 ###")
    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    special_tokens_dict = {"additional_special_tokens": ["[RELATION]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    set_seed(training_args.seed)
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    model_config.num_labels = 128
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    wandb.init(
        entity="psrpsj",
        project="traveldata",
        name=model_args.project_cat3_name,
        tags=[model_args.model_name],
    )
    wandb.config.update(training_args)

    train_dataset, valid_dataset = train_test_split(
        data, test_size=0.2, stratify=data["cat3"], random_state=42
    )

    train_dataset["overview"] = (
        train_dataset["overview"]
        + "[RELATION]"
        + train_dataset["cat1"]
        + "[RELATION]"
        + train_dataset["cat2"]
    )
    valid_dataset["overview"] = (
        valid_dataset["overview"]
        + "[RELATION]"
        + valid_dataset["cat1"]
        + "[RELATION]"
        + valid_dataset["cat2"]
    )
    train_dataset["cat3"] = label_to_num(train_dataset["cat3"], 3)
    valid_dataset["cat3"] = label_to_num(valid_dataset["cat3"], 3)

    train = CustomDataset(train_dataset["overview"], train_dataset["cat3"], tokenizer)
    valid = CustomDataset(valid_dataset["overview"], valid_dataset["cat3"], tokenizer)

    trainer = CustomTrainer(
        loss_name=model_args.loss_name,
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
    )

    print("--- CAT3 NLP TRAINING ---")
    trainer.train()
    model.save_pretrained(
        os.path.join(training_args.output_dir, model_args.project_cat3_name)
    )
    wandb.finish()
    print("--- CAT3 NLP TRAINING FINISH ---")


def main():
    dataset = pd.read_csv("./data/train.csv")
    dataset = preprocess_nlp(dataset)
    if not os.path.exists("./output/cat1_nlp"):
        train_cat1_nlp(dataset)
    if not os.path.exists("./output/cat2_nlp"):
        train_cat2_nlp(dataset)
    train_cat3_nlp(dataset)


if __name__ == "__main__":
    main()
