import pandas as pd
import pickle
import re


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


def num_to_label(data, cat_num):
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


def preprocess_nlp(dataset):
    for idx in range(len(dataset)):
        to_fix = dataset["overview"][idx]
        to_fix = re.sub("<.+?>", "", to_fix)
        to_fix = re.sub("[^ 가-힣0-9a-zA-Z]", "", to_fix)
        dataset["overview"][idx] = to_fix
    return dataset
