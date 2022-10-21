import albumentations as A
import numpy as np
import os
import pandas as pd
import torch
import wandb

from albumentations.pytorch.transforms import ToTensorV2
from argument import (
    TrainNLPModelArguments,
    TrainingNLPArguments,
)
from dataset import CustomCVDataset, CustomNLPDataset
from loss import create_criterion
from model import CNN
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def train_nlp(data: pd.DataFrame, cat_num: int):
    parser = HfArgumentParser((TrainingNLPArguments, TrainNLPModelArguments))
    (training_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if cat_num == 1:
        model_args.project_name += "_cat1_nlp"
        model_args.target_label = "cat1"
    elif cat_num == 2:
        model_args.project_name += "_cat2_nlp"
        model_args.target_label = "cat2"
        data["overview"] = data["overview"] + "[RELATION]" + data["cat1"]
    elif cat_num == 3:
        model_args.project_name += "_cat3_nlp"
        model_args.target_label = "cat3"
        data["overview"] = (
            data["overview"] + "[RELATION]" + data["cat1"] + "[RELATION]" + data["cat2"]
        )

    print(f"### Training NLP Model for {model_args.target_label} ###")
    print(f"Current model is {model_args.model_name}")
    print(f"Current device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    special_tokens_dict = {"additional_special_tokens": ["[RELATION]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    set_seed(training_args.seed)

    # K-Fold
    if model_args.k_fold:
        print("---- START K-FOLD ----")
        fold = 1
        k_fold = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, valid_index in k_fold.split(
            data["overview"], data[model_args.target_label]
        ):
            wandb.init(
                entity="psrpsj",
                project="traveldata",
                name=model_args.project_name + "_kfold_" + str(fold),
                tags=[model_args.model_name],
            )
            wandb.config.update(training_args)

            print(f"---- Fold {fold} start ----")
            output_dir = os.path.join(
                training_args.output_dir,
                model_args.project_name + "_kfold",
                "fold" + str(fold),
            )

            train_view, valid_view = (
                data["overivew"][train_index],
                data["overview"][valid_index],
            )
            train_label, valid_label = (
                data[model_args.target_label][train_index],
                data[model_args.target_label][valid_index],
            )
            train_label = label_to_num(train_label, cat_num)
            valid_label = label_to_num(valid_label, cat_num)

            train = CustomNLPDataset(train_view, train_label, tokenizer)
            valid = CustomNLPDataset(valid_view, valid_label, tokenizer)

            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name
            )
            if cat_num == 1:
                model_config.num_labels = 6
            elif cat_num == 2:
                model_config.num_labels = 18
            elif cat_num == 3:
                model_config.num_labels = 128

            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name, config=model_config
            )
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model.train()

            trainer = CustomTrainer(
                loss_name=model_args.loss_name,
                model=model,
                args=training_args,
                train_dataset=train,
                eval_dataset=valid,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            model.save_pretrained(output_dir)
            wandb.finish()
            fold += 1
            print(f"---- Fold {fold} finish ----")

    else:
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name
        )
        if cat_num == 1:
            model_config.num_labels = 6
        elif cat_num == 2:
            model_config.num_labels = 18
        elif cat_num == 3:
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
            name=model_args.project_name,
            tags=[model_args.model_name],
        )
        wandb.config.update(training_args)
        train_dataset, valid_dataset = train_test_split(
            data, test_size=0.2, stratify=data[model_args.target_label], random_state=42
        )

        train_dataset[model_args.target_label] = label_to_num(
            train_dataset[model_args.target_label], cat_num
        )
        valid_dataset[model_args.target_label] = label_to_num(
            valid_dataset[model_args.target_label], cat_num
        )

        train = CustomNLPDataset(
            train_dataset["overview"], train_dataset[model_args.target_label], tokenizer
        )
        valid = CustomNLPDataset(
            valid_dataset["overview"], valid_dataset[model_args.target_label], tokenizer
        )

        trainer = CustomTrainer(
            loss_name=model_args.loss_name,
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            compute_metrics=compute_metrics,
        )

        print(f"---{model_args.target_label.upper()} NLP TRAINING ---")
        trainer.train()
        model.save_pretrained(
            os.path.join(training_args.output_dir, model_args.project_name)
        )
        wandb.finish()
        print(f"---{model_args.target_label.upper()} NLP TRAINING FINISH ---")


def train_cv():
    train = pd.read_csv("./data/train.csv")
    CFG = {
        "IMG_SIZE": 128,
        "EPOCHS": 5,
        "LEARNING_RATE": 3e-4,
        "BATCH_SIZE": 64,
        "SEED": 42,
    }
    train["cat1"] = label_to_num(train["cat1"], 1)
    train_data, valid_data = train_test_split(
        train, test_size=0.2, stratify=train["cat1"], random_state=CFG["SEED"]
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_transform = A.Compose(
        [
            A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )
    train_dataset = CustomCVDataset(
        train_data["img_path"], train_data["cat1"], train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)
    valid_dataset = CustomCVDataset(
        valid_data["img_path"], valid_data["cat1"], train_transform
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False
    )
    criterion = create_criterion("focal")

    best_score = 0
    best_model = None
    model = CNN()
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=False
    )

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []

        for img, label in tqdm(iter(train_loader)):
            img = img.float().to(device)
            label = label.to(device)
            optimizer.zero_grad()
            model_pred = model(img)
            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        epoch_loss = np.mean(train_loss)

        # validation
        model.eval()
        model_preds = []
        true_labels = []
        val_loss = []

        with torch.no_grad():
            for img, label in tqdm(iter(valid_loader)):
                img = img.float().to(device)
                label = label.to(device)

                model_pred = model(img)
                loss = criterion(model_pred, label)
                val_loss.append(loss.item())

                model_preds += model_pred.argmax(-1).detach().cpu().numpy().tolist()
                true_labels += label.detach().cpu().numpy().tolist()

        test_score = f1_score(true_labels, model_preds, average="weighted")
        val_loss, val_score = np.mean(val_loss), test_score

        print(
            f"Epoch{epoch}: Train_loss:{epoch_loss:.5f} Val_loss:{val_loss:.5f} Val_score:{val_score:.5f}"
        )

        if scheduler is not None:
            scheduler.step()

        if best_score < val_score:
            best_score = val_score
            best_model = model

    torch.save(best_model.state_dict(), "./output/cv_baseline")
    return best_model


def main():
    dataset = pd.read_csv("./data/train.csv")
    dataset = preprocess_nlp(dataset, train=True)
    if not os.path.exists("./output/cat1_nlp"):
        train_nlp(dataset, 1)
    if not os.path.exists("./output/cat2_nlp"):
        train_nlp(dataset, 2)
    train_nlp(dataset, 3)


if __name__ == "__main__":
    main()
