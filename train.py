import os
import pandas as pd
import torch
import wandb

from argument import TrainingCat1Arguments, TrainCat1ModelArguments
from dataset import CustomDataset
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from trainer import CustomTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)


def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    return {"accuracy": acc}


def train_cat1():
    parser = HfArgumentParser((TrainingCat1Arguments, TrainCat1ModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("### Training NLP Model for Cat 1 ###")
    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    data = pd.read_csv(os.path.join(model_args.data_path, "train.csv"))
    label = preprocessing.LabelEncoder()
    label.fit(data["cat1"])
    data["cat1"] = label.transform(data["cat1"])
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
        tags=model_args.model_name,
    )
    wandb.config.update(training_args)

    train_dataset, valid_dataset = train_test_split(
        data, test_size=0.2, stratify=data["cat1"], random_state=42
    )
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
    model.save_pretrained(training_args.output_dir + model_args.project_cat1_name)
    wandb.finish()
    print("--- CAT1 NLP TRAINING FINISH ---")


def main():
    train_cat1()


if __name__ == "__main__":
    main()
