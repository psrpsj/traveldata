import cv2
import torch

from torch.utils.data import Dataset


class CustomNLPDataset(Dataset):
    def __init__(self, reviews, label, tokenizer):
        self.label = label.tolist()
        self.tokenized_sentence = tokenizer(
            reviews.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = torch.tensor(self.label[idx])
        return encoded

    def __len__(self):
        return len(self.label)


class CustomCVDataset(Dataset):
    def __init__(self, img_path_list, transform):
        self.img_path_list = img_path_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

    def __len__(self):
        return len(self.img_path_list)
