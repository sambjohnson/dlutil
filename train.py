# ---- torch imports --- #
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# --- general system / math imports --- #
from typing import Dict, Any

import pandas as pd
import numpy as np

import os
import sys
import pickle
import struct
from array import array
import numpy as np

from .utils import get_device

def train_loop(dataloader, model, loss_fn, optimizer):
    device = get_device()
    size = len(dataloader.dataset)
    # ensure model is set to training mode in case previously in evaluation mode
    model.train()
    for batch, (X_cpu, y_cpu) in enumerate(dataloader):
        # Compute prediction and loss
        X = X_cpu.to(device) # put both model and data on gpu (if available)
        y = y_cpu.to(device)
        pred = model(X)
        # this is required for models with multiple returns; by assumption,
        # the first return of a tuple is the prediction; can only squeeze a tensor
        if isinstance(pred, tuple):
            pred = tuple([torch.squeeze(p) for p in pred])
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    device = get_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # set model to evaluation mode to control dropout and batchnorm layers, etc.
    model.eval()

    # note: no_grad() ensures gradients are not computed as they won't be used;
    # it is only valuable for efficiency and will not alter the training result
    with torch.no_grad():
        # evaluate loss by looping once over entire testing set
        # ensure no gradients are computed
        for X_cpu, y_cpu in dataloader:
            X = X_cpu.to(device) # put both model and data on gpu (if available)
            y = y_cpu.to(device)

            pred = model(X)
            # this is required for models with multiple returns; by assumption,
            # the first return of a tuple is the prediction; can only squeeze a tensor
            if isinstance(pred, tuple):
                pred = tuple([torch.squeeze(p) for p in pred])
            else:
                pred = torch.squeeze(pred)
            test_loss += loss_fn(pred, y).item()
    
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def train(model, loss_fn, optimizer, trainl, testl, epochs=None):
    """ Trains model by minimizing loss_fn using optimizer.
        Trains on trainl data and tests occassionally on testl data.
        Loops over training dataset for # determined by epochs.
        No returns; the model is trained in-place.
    """
    # training happens here; can take a long time
    if epochs is None:
        epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainl, model, loss_fn, optimizer)
        test_loop(testl, model, loss_fn)
    print("Done!")
