from unicodedata import bidirectional
import numpy
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    BertPreTrainedModel,
    SequenceClassifierOutput,
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            # [100, 3, ]
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        output = self.layer(x)
        output = torch.flatten(output, start_dim=1)
        return output


class BERTwithLSTM(BertPreTrainedModel):
    def __init__(self, model, config, *args, **kwargs):
        super().__init__(config=config)

        self.bert = AutoModelForSequenceClassification.from_pretrained(model)
        self.lstm = nn.LSTM(1024, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256 * 2, 30)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh
        self.linear2 = nn.Linear(30, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_output, (h, c) = self.lstm(output[0])
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        linear_output = self.linear(hidden.view(-1, 256 * 2))
        x = self.tanh(linear_output)
        outputs = SequenceClassifierOutput(logits=x)
        return outputs
