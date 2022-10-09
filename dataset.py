from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, reviews, label, tokenizer):
        self.label = label.tolist()
        self.tokenized_sentence = tokenizer(
            reviews.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
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
