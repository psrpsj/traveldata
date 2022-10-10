import pickle


def label_to_num(data, cat_num):
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
