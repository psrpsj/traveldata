import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from argument import (
    TrainNLPModelArguments,
    TrainCat1NLPModelArguments,
    TrainCat2NLPModelArguments,
    TrainCat3NLPModelArguments,
)
from dataset import CustomNLPDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
from utils import num_to_label, preprocess_nlp


def inference_nlp(dataset: pd.DataFrame, cat_num: int) -> pd.DataFrame:
    parser = HfArgumentParser(TrainNLPModelArguments)
    (model_args,) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    special_tokens_dict = {"additional_special_tokens": ["[RELATION]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    overview_fix = dataset["overview"]

    if cat_num == 1:
        model_args.target_label = "cat1"
        model_args.project_name += "_cat1_nlp"
    elif cat_num == 2:
        model_args.target_label = "cat2"
        model_args.project_name += "_cat2_nlp"
        overview_fix = dataset["overview"] + "[RELATION]" + dataset["cat1"]
    elif cat_num == 3:
        model_args.target_label = "cat3"
        model_args.project_name += "_cat3_nlp"
        overview_fix = (
            dataset["overview"]
            + "[RELATION]"
            + dataset["cat1"]
            + "[RELATION]"
            + dataset["cat2"]
        )

    dataset[model_args.target_label] = [-1] * len(dataset)
    test_dataset = CustomNLPDataset(
        overview_fix, dataset[model_args.target_label], tokenizer
    )
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_path = os.path.join("./output/", model_args.project_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    output_prob = []
    output_pred = []

    print(f"### Inference for {model_args.target_label.upper()} NLP ###")
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
    dataset[model_args.target_label] = num_to_label(pred_answer, cat_num)
    dataset.to_csv("./data/test_fix.csv", index=False)
    if cat_num == 3:
        submission = pd.DataFrame({"id": dataset["id"], "cat3": dataset["cat3"]})
        submission.to_csv("./output/submission.csv", index=False)

    print(f"### Inference for {model_args.target_label.upper()} NLP Finish! ###")
    return dataset


def main():
    dataset = pd.read_csv("./data/test.csv")
    dataset = preprocess_nlp(dataset)
    if "cat1" not in dataset.columns.tolist():
        dataset = inference_nlp(dataset, 1)
    if "cat2" not in dataset.columns.tolist():
        dataset = inference_nlp(dataset, 2)
    inference_nlp(dataset, 3)


if __name__ == "__main__":
    main()
