import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import torch

from konlpy.tag import Okt
from tqdm import tqdm


def label_to_num(data: pd.Series, cat_num: int) -> pd.Series:
    if cat_num == 1:
        pkl_1 = open("cat1_label.pkl", "rb")
        label_1 = pickle.load(pkl_1)
        pkl_1.close()
        return label_1.transform(data)
    elif cat_num == 2:
        pkl_2 = open("cat2_label.pkl", "rb")
        label_2 = pickle.load(pkl_2)
        pkl_2.close()
        return label_2.transform(data)
    elif cat_num == 3:
        pkl_3 = open("cat3_label.pkl", "rb")
        label_3 = pickle.load(pkl_3)
        pkl_3.close()
        return label_3.transform(data)


def num_to_label(data: pd.Series, cat_num: int) -> pd.Series:
    if cat_num == 1:
        pkl_1 = open("cat1_label.pkl", "rb")
        label_1 = pickle.load(pkl_1)
        pkl_1.close()
        return label_1.inverse_transform(data)
    elif cat_num == 2:
        pkl_2 = open("cat2_label.pkl", "rb")
        label_2 = pickle.load(pkl_2)
        pkl_2.close()
        return label_2.inverse_transform(data)
    elif cat_num == 3:
        pkl_3 = open("cat3_label.pkl", "rb")
        label_3 = pickle.load(pkl_3)
        pkl_3.close()
        return label_3.inverse_transform(data)


def preprocess_nlp(dataset: pd.DataFrame, train: bool) -> pd.DataFrame:
    f = open("./data/stopword.txt", encoding="UTF-8")
    line = f.readlines()
    stopwords = []

    for l in line:
        l = l.replace("\n", "")
        stopwords.append(l)

    print("### Start Preprocess for Overview ###")
    drop_list = []
    for idx in tqdm(range(len(dataset))):
        to_fix = dataset["overview"][idx]
        to_fix = " ".join(to_fix.split())
        to_fix_sub = re.sub("<.+?>", "", to_fix)
        if len(to_fix) == 0:
            to_fix_sub = to_fix
        to_fix = re.sub("[^ 가-힣0-9a-zA-Z]", "", to_fix_sub)

        # Stopword
        okt = Okt()
        to_fix_morphs = okt.morphs(to_fix)
        fixed_list = [w for w in to_fix_morphs if w not in stopwords]
        to_fix_finish = " ".join(fixed_list)
        if train and len(to_fix_finish) == 0:
            drop_list.append(idx)
        if not train and len(to_fix_finish) == 0:
            to_fix_finish = to_fix
        dataset["overview"][idx] = to_fix_finish

    if train:
        dataset.drop(drop_list)
    return dataset


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
