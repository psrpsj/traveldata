import os
import torch
import wandb

from sklearn.metrics import accuracy_score

def compute_meterics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    return {"accuracy" : acc}

def train_cat1():
    