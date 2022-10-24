import torch
from .utils import _make_agg_matrix, aggregate_classes, _get_nclasses_orig
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

# these are required for defining the custom datasets
from typing import Dict, Any
from torchvision import datasets

import pandas as pd
import numpy as np
import os
import collections


def get_split_indices(dataset, ratio):
    """ Create random split of indices into train and test indices.
        Arguments:
            dataset: a torch Dataset object
            ratio: the ratio of total amount of data to test set
              (e.g., ratio=10 implies that the test set will be
              about 1/10th of the total dataset size.)
        Returns:
            A tuple of indices: train_indices, test_indices (as lists)
    """
    nsamples = len(dataset)
    indices = list(range(nsamples))
    ntest = nsamples // ratio
    test_indices = list(np.random.choice(indices, size=ntest, replace=False))
    train_indices = list(set(indices) - set(test_indices))
    return train_indices, test_indices


def get_train_test_datasets(dataset, ratio, index_pair=None):
    """ Splits a dataset into train and test datasets.
        Arguments:
            dataset: a torch Dataset object
            ratio: the ratio of total amount of data to test set
                (e.g., ratio=10 implies that the test set will be
                about 1/10th of the total dataset size.)
            index_pair: (optional) if supplied, a pair of indices
                (train_inds, test_inds) that specify which indices
                of the dataset to sort into train and test datasets.
                If not supplied, split is made randomly according to ratio.
    """
    if index_pair is None:
        index_pair = get_split_indices(dataset, ratio)
    else:
        assert isinstance(index_pair, collections.Sequence) and len(index_pair) == 2
    train_indices, test_indices = index_pair
    ds_train = torch.utils.data.Subset(dataset, train_indices)
    ds_test = torch.utils.data.Subset(dataset, test_indices)
    return ds_train, ds_test


class ToFloat(object):
    """ Converts the datatype in sample to torch.float32 datatype.
        - helper function to be used as transform (typically from uint8 to float)
        - useful because inputs must have the same datatype as weights of the n.n.
    """

    def __call__(self, target):
        target_tensor = torch.tensor(target)
        return target_tensor.to(torch.float32)


class ToLong(object):

    def __call__(self, target):
        target_tensor = torch.tensor(target)
        return target_tensor.to(torch.int64)


class ToOneHot(object):

    def __init__(self, nclasses=-1):
        super.__init__()
        self.nclasses = nclasses

    def __call__(self, target):
        return F.one_hot(target, num_classes=self.nclasses)


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
    """

    def __init__(self, annotations_file, img_dir, transform=None,
                 target_transform=None, npy=False):
        """If dealing with multidimensional labels, they can be specified by column location
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
    def __init__(self, annotations_file, img_dir, transform=None,
                 target_transform=None, label_idx=1,
                 npy=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.npy = npy
        self.label_idx = label_idx

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # is the image saved as an .np array?
        if self.npy:
            image = np.load(img_path)
        else:
            image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, self.label_idx]  # note: iloc works with lists of indices
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomUnetDataset(Dataset):
    """ Custom dataset, like ImageFolder, designed for Unets.
        Designed for inputs of the form (X, y) where X and y
        are stored in separate directories.
        Arguments:
            xdir: directory containing X images (should be either
                image objects that can be opened by PIL or else
                np arrays, saved as e.g., .npy files.
            ydir: directory containing y images (should be either
                image objects that can be opened by PIL or else'
                np arrays, saved as e.g., .npy files.
                Thees should be of shape (px_x, px_y) or
                (px_x, px_y, ch) (if one-hot). They are the
                correct class labels corresponding to each
                pixel of the corresponding X image.
            mapping_file: a .csv of format:
                (xfilename, yfilename) that associates
                each file in xdir to a file in ydir.
                X is the unlabeled image; y contains the
                ground truth labels (pixelwise), either as an integer
                or as a one-hot vector, for each pixel.
            format: (optional: default is 'numpy') By default, assumes
                x and y are stored as numpy arrays representing image
                pixel values; any other value will ve assumed to be
                an image in a format open-able by PIL.
        Returns:
            a CustomUnetDataset object, capable of iterating
            through X, y pairs in typical Dataset fashion.
    """

    def __init__(self,
                 xdir,
                 ydir,
                 mapping_file,
                 transform=None,
                 target_transform=None,
                 format='numpy',
                 onehot=True,
                 nclasses=None,
                 agg_dict=None,
                 return_metadata=None
                ):
        self.xy_pairs = pd.read_csv(mapping_file)  # reads .csv into pd dataframe
        self.xdir = xdir
        self.ydir = ydir
        self.mapping_file = mapping_file
        self.mapping_df = pd.read_csv(mapping_file)
        self.transform = transform
        self.target_transform = target_transform
        self.format = format
        self.onehot = onehot
        self.nclasses = nclasses
        self.agg_dict = agg_dict
        if self.agg_dict is not None:
            n_orig = _get_nclasses_orig(agg_dict)
            self.agg_matrix = _make_agg_matrix(n_orig, agg_dict)

    def __len__(self):
        return len(self.mapping_df)

    def __getitem__(self, idx):

        xpath = os.path.join(self.xdir, self.mapping_df.iloc[idx, 0])
        ypath = os.path.join(self.ydir, self.mapping_df.iloc[idx, 1])
        if self.format == 'numpy':
            x = np.load(xpath)
            y = np.load(ypath)
        else:
            x = read_image(xpath)
            y = read_image(ypath)
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        if self.onehot:
            # note: permutation is necessary to move channel axis to axis 1.
            # meanwhile, squeeze removes an unnecessary axis (how did that get there??)
            # note: indices would be (0, 3, 1, 2) for batch of ys
            y = torch.permute(torch.squeeze(F.one_hot(y.to(torch.int64),
                                                      num_classes=self.nclasses)), (2, 0, 1))
        if self.agg_dict is not None:
            y = aggregate_classes(y, self.agg_matrix)

        return x, y.float()


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

