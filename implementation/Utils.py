import os
from numpy.random import permutation
import numpy as np
from shutil import copytree
from torch import nn


def get_dense_block(input_size, hidden_sizes, activation=nn.ReLU):
    hiddens = []
    simple_past_layer = input_size
    for layer_size in hidden_sizes:
        hiddens.append(nn.Linear(simple_past_layer, layer_size))
        hiddens.append(activation())
        simple_past_layer = layer_size

    hiddens.pop(-1)
    return nn.Sequential(*hiddens)


def train_dev_split(raw_path, out_path, train_part):
    family_dirs = os.listdir(raw_path)
    n_families = len(family_dirs)
    train_size = int(n_families * train_part)
    permuted_indices = permutation(np.arange(n_families))
    train_indices = permuted_indices[:train_size]
    if not os.path.isdir(os.path.join(out_path, 'train')):
        os.mkdir(os.path.join(out_path, 'train'))
    if not os.path.isdir(os.path.join(out_path, 'dev')):
        os.mkdir(os.path.join(out_path, 'dev'))
    for ind, item in enumerate(family_dirs):
        partition = 'train' if ind in train_indices else 'dev'
        copytree(os.path.join(raw_path, item), os.path.join(out_path, partition, item))
