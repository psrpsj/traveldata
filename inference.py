import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from argument import TrainCat1NLPModelArguments, TrainCat2NLPModelArguments
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
from utils import num_to_label


def inference_cat1_nlp(dataset):
    parser = HfArgumentParser(TrainCat1NLPModelArguments)
    (model_args,) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    dataset["cat1"] = [-1] * len(dataset)
    test_dataset = CustomDataset(dataset["overview"], dataset["cat1"], tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_path = os.path.join("./output/", model_args.project_cat1_name)
    model = AutoModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    output_prob = []
    output_pred = []

    print("### Inference for CAT1 NLP ###")
    for data in tqdm(dataloader):
        output = model(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            token_type_ids=data["token_type_ids"].to(device),
        )
        logit = output[0]
        prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
        logit = logit.detach().cpu().numpy()
        result = np.argmax(logit, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)

    pred_answer = np.concatenate(output_pred).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    output = pd.DataFrame(
        {
            "id": dataset["id"],
            "img_path": dataset["img_path"],
            "overview": dataset["overview"],
            "cat1": pred_answer,
        }
    )
    output["cat1"] = num_to_label(output["cat1"], 1)
    output.to_csv("./data/test_fix.csv", index=False)
    print("### Inference for CAT1 NLP Finish! ###")
    return output


def inference_cat2_nlp(dataset):
    parser = HfArgumentParser(TrainCat2NLPModelArguments)
    (model_args,) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_avaliable() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    special_tokens_dict = {"additional_special_tokens": ["[RELATION]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    overview_fix = dataset["overview"] + "[RELATION]" + dataset["cat1"]
    dataset["cat2"] = [-1] * len(dataset)
    test_dataset = CustomDataset(overview_fix, dataset["cat2"], tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_path = os.path.join("./output/", model_args.project_cat2_name)
    model = AutoModel.from_pretrained(model_path)
    model.resize_token_embedding(len(tokenizer))
    model.to(device)
    model.eval()

    output_prob = []
    output_pred = []
    print("### Inference for CAT2 NLP ###")
    for data in tqdm(dataloader):
        output = model(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            token_type_ids=data["token_type_ids"].to(device),
        )
        logit = output[0]
        prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
        logit = logit.detach().cpu().numpy()
        result = np.argmax(logit, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)

    pred_answer = np.concatenate(output_pred).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    output = pd.DataFrame(
        {
            "id": dataset["id"],
            "img_path": dataset["img_path"],
            "overview": dataset["overview"],
            "cat1": dataset["cat1"],
            "cat2": pred_answer,
        }
    )
    output["cat2"] = num_to_label(output["cat2"], 2)
    output.to_csv("./data/test_fix.csv", index=False)
    print("### Inference for CAT2 NLP Finish! ###")
    return output


def main():
    dataset = pd.read_csv("./data/test.csv")
    if "cat1" not in dataset.columns.tolist():
        dataset = inference_cat1_nlp(dataset)
    if "cat2" not in dataset.columns.tolist():
        dataset = inference_cat2_nlp(dataset)


if __name__ == "__main__":
    main()
