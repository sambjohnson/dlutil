import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

# these are required for defining the regession ImageFolder
from typing import Dict, Any
from torchvision import datasets

import pandas as pd
import numpy as np
import os


class ToFloat(object):
    """ Converts the datatype in sample to torch.float32 datatype.
        - helper function to be used as transform (typically from uint8 to float)
        - useful because inputs must have the same datatype as weights of the n.n.
    """

    def __call__(self, target):
        target_tensor = torch.tensor(target)
        return target_tensor.to(torch.float32)


class ToRGB(object):
    """ Converts a 1-channel tensor into 3 (equal) channels
        for ease of use with pretrained vision models.
    """
    
    def __call__(self, image):
        image = torch.tensor(image)
        image = image.repeat(3, 1, 1)
        # print(image.shape) # for testing
        return image


class CustomImageDataset(Dataset):
    """ Custom dataset, like ImageFolder, works with arbitrary image labels.
        Labels should be a .csv in the format:
            image1filename.png, image1label
            image2filename.png, image2label
            ...
        Useful for regression; circumvents ImageFolder classification scheme
        which requires that images be sorted into subfolders corresponding to class names.
        If dealing with multidimensional labels, they can be specified by column location
        in label .csv file
        Args:
            annotations_file: pathlike string to the .csv with annotations
            img_dir: pathlike string to the image_directory
            transform: transform for images (X's)
            target_transform: transform for targets (y's); if y is vector, target_transform can
                take vector inputs. No checks for consistency between y and this transform;
                that is contingent on the user.
            label_idx: an index (or list of integer indices) indicating which column(s) of .csv
                contain y values. By default, this is 1 -- the second column of the .csv.
        Returns:
            custom dataset where X is an image in img_dir and y is a corresponding label from
            annotations_file. A dataset that is well-suited to regression, which will return
            X, y on each iteration, consistent with torch's expectations.
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, label_idx=1):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_idx = label_idx

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, self.label_idx]  # note: iloc works with lists of indices
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_split_indices(dataset, ratio):
    """ Create random split into train and test sets according to ratio.
    """
    nsamples = len(dataset)
    indices = list(range(nsamples))
    ntest = nsamples // ratio
    test_indices = list(np.random.choice(indices, size=ntest, replace=False))
    train_indices = list(set(indices) - set(test_indices))
    return train_indices, test_indices


def get_train_test_split(dataset, ratio):
    """ Function to automatically (randomly) split a dataset
        into a train and test set, and return those in the format
            (trainset, testset)
        The ratio should be the ratio of the total size to the test size,
        e.g., ratio=10 will make a testet with 1/10th of the overall data.
    """
    train_indices, test_indices = get_split_indices(dataset, ratio)
    train = torch.utils.data.Subset(dataset, train_indices)
    test = torch.utils.data.Subset(dataset, test_indices)
    return train, test
