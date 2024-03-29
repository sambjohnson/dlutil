# ---- torch imports --- #
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# --- general system imports --- #
from typing import Dict, Any

import pandas as pd
import numpy as np

import os
import sys
import pickle
import struct
import copy
from array import array

# --- image processing imports --- #
import png
from PIL import Image
from PIL import ImageOps

import matplotlib.pyplot as plt
import numpy as np

# --- utility functions --- #


def get_device():
    """ Returns gpu if available, cpu otherwise."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_requires_grad(module, allfalse=False):
    """ Checks whether all of a module's parameters
        have requires_grad = True. Returns False if not.
        If allfalse=True, instead returns True only if
        all module parameters do NOT require grad.
        
        Carefully understand allfalse=True behavior.
    """
    assert isinstance(module, nn.Module)
    rg = [p.requires_grad for p in module.parameters()]
    ret = all(r == (not allfalse) for r in rg)
    return ret


def set_unfreeze_(model, submodules_to_unfreeze):
    """ In-place unfreezes only specified submodules
        given in list (submodules_to_unfreeze);
        freezes all other parts of model.
        Useful for transfer learning with pretrained models
        and doing combined feature extraction / finetuning.
    """
    model.requires_grad_(False)  # freezes entire model
    for subm in submodules_to_unfreeze:
        subm.requires_grad_(True)  # unfreezes just specified sm's
    return


def make_pretrained_state_dict(model_state, pretrained_state):
    """ Helper function to load copy a portion of a model's state
        from another model with (partially) overlapping architecture.
        Args:
            model_state: a PyTorch-style OrderedDict with the target state
                of the model to initialize
            pretrained_state: a PyTorch-style OrderedDict with the loaded
                (presumably pretrained) parameters from a partially comparable
                model.
        Returns:
            warmstart_state: a state_dict that is equal to an updated version
            model_state, where keys that are also present in pretrained_state
            are updated to their values in that state. E.g., pretrained_state
            may be the parameters of a large, pretrained model, and model_state
            may be the state of an architecture that only uses some of that
            model's first layeres.

    """
    warmstart_params = copy.deepcopy(model_state)
    for k, v in pretrained_state.items():
        warmstart_params[k] = v
    return warmstart_params


def get_last_fns(model_load_dir, return_optim=True):
    """
        Selects the latest model and load files
        from a given directory, returning them
        as a tuple of filenames.
    """
    import os
    fns = os.listdir(model_load_dir)
    opt = []
    if return_optim:
        opt = sorted([fn for fn in fns if 'optim' in fn])[-1]
    mod = sorted([fn for fn in fns if 'model' in fn])[-1]
    return mod, opt


def make_warmstart_state_dict(model,
                              pretrained_params,
                              discard_keywords=None):
    """
        Creates a state dict suitable for initializing a model whose
        architecture partially overlaps with that of another, already-trained
        model.

        Note: to save space, this function WILL modify the dictionary
            pretrained_params *IN PLACE* by popping some elements!

        Arguments:
            model : the PyTorch model to initialize
            pretrained_params : the (possibly incompatible) state dict of
                another model, which is to be partially pasted onto the model's
                own parameters.
            discard_keywords : keywords
        Returns:
            params : the joined parameters ready to be loaded into new model
    """
    if discard_keywords is None:
        discard_keywords = ['conv_last']
    discard_keys = []
    for keyword in discard_keywords:
        discard_keys += [k for k in list(pretrained_params.keys()) if
                         keyword in k]
    for k in discard_keys:
        pretrained_params.pop(k, None)
    params = make_pretrained_state_dict(model.state_dict(), pretrained_params)
    return params


def imshow(inp, title=None, normalize=False, figsize=None,
           cmap=None):
    """ Imshow for Tensor. Visualize images in a grid. """
    inp = inp.numpy().transpose((1, 2, 0))
    
    if figsize is None:
        figsize = (20, 10)

    if normalize:
        # normalization may be required in some cases, but not here
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(inp, cmap=cmap)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def _get_nclasses_orig(agg_dict):
    return max([max(l) for l in list(agg_dict.values())]) + 1


def make_agg_dict(n_total, n_distinct):
    """
        Create a dictionary representing a partition of the set
        {0, 1, ..., n_total - 1}.
        The first n_distinct elements remain in their own partition,
        and the remaining elements are grouped into a single partition.
    """
    d = {}
    for i in range(n_distinct):
        d[i] = [i]
    d[n_distinct] = list(range(n_distinct, n_total))
    return d


def _make_agg_matrix(n_orig_classes, agg_dict, dtype=torch.int64):
    """
        Internal method to create matrix which collapses one-hot
        encoded classes into a smaller number of aggregate classes
        by matrix multiplication. For example:
        Args:
            n_orig_classes: (int) number of classes in input
            agg_dict: a dictionary mapping the output classes to
                desired groups of input classes, e.g. lists. Example:
                agg_dict = {0: [0], 1: [1, 2, 3], 2: [4, 5]}
        Returns:
            a matrix of zeros and ones that will transform a one-hot
            encoded class into the appropriate aggregate class. E.g.,
            with the above example agg_dict, we would have
                agg_matrix = [[1, 0, 0,],
                              [0, 1, 0],
                              [0, 1, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 1]]
                so that:
                    [1, 0, 0, 0, 0, 0] * agg_matrix = [1, 0, 0]  # class 0 -> class 0
                    [0, 0, 0, 1, 0, 0] * agg_matrix = [0, 1, 0]  # class 3 -> class 1
                    [0, 0, 0, 0, 1, 0] * agg_matrix = [0, 0, 1]  # class 4 -> class 2
    """
    n_agg_classes = len(set(agg_dict.keys()))
    agg_matrix = torch.zeros(n_orig_classes, n_agg_classes, dtype=dtype)
    for k, v in agg_dict.items():
        agg_matrix[v, k] = 1
    return agg_matrix


def aggregate_classes(y, agg_matrix):
    assert len(y.shape) == 3 or len(y.shape) == 4
    if len(y.shape) == 3:
        inds_permute = (1, 2, 0)
        inds_unpermute = (2, 0, 1)
    elif len(y.shape) == 4:
        inds_permute = (0, 2, 3, 1)
        inds_unpermute = (0, 3, 1, 2)
    return torch.permute(torch.matmul(torch.permute(y, inds_permute), agg_matrix), inds_unpermute)


# helper functions for visualization of onehot-encoded labels
def from_onehot(y, batch=True):
    axis = 1 if batch else 0
    return torch.argmax(y, dim=axis)


def y_vis_sample(y):
    y_collapsed = from_onehot(y, batch=False)
    return torch.unsqueeze(y_collapsed, dim=0)




