import os
import pickle
from numpy.random import permutation
import numpy as np
from shutil import copytree
from torch import nn
import torch
from matplotlib import pyplot as plt
from ignite.engine import Engine, create_supervised_trainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def simple_concatenation(x1, x2):
    return torch.cat([x1, x2], dim=1)


# Feature combination naming conventions: starting on the outside and going in
def difference_squared(x1, x2):
    return (x1 ** 2) - (x2 ** 2)


def squared_difference(x1, x2):
    res = x1-x2
    return res ** 2


def multification(x1, x2):
    return x1*x2


def summation(x1, x2):
    return x1+x2


def sqrt_difference(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sign(x1 - x2) * torch.sqrt(torch.abs(x1 - x2))


def sqrt_sum(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sign(x1 + x2) * torch.sqrt(torch.sign(x1 + x2))


def difference_sqrt(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sign(x1) * torch.sqrt(torch.abs(x1)) - torch.sign(x2) * torch.sqrt(torch.abs(x2))


def sum_sqrt(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sign(x1) * torch.sqrt(torch.abs(x1)) + torch.sign(x2) * torch.sqrt(torch.abs(x2))


def difference(x1: torch.Tensor, x2: torch.Tensor):
    return x1 - x2


feature_combination_list = [difference_squared, squared_difference, multification, summation, sqrt_difference, sqrt_sum,
                            difference_sqrt, sum_sqrt, difference]


def plot_metric(values, title, y_label, index_scale=1, **kwargs):
    fig = plt.figure()
    plt.cla()
    # plt.scatter(range(len(values)), values, **kwargs)
    plt.plot(range(0, len(values) * index_scale, index_scale), values, **kwargs)
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.title(title)
    if not os.path.isdir(os.path.join(PROJECT_ROOT, 'figs')):
        os.mkdir(os.path.join(PROJECT_ROOT, 'figs'))

    plt.savefig(os.path.join(PROJECT_ROOT, "figs", "{}_plot.png".format(title.replace(" ", "_"))))
    plt.close(fig)


def load_checkpoint(model_class, experiment_dir, checkpoint_name, device):
    with open(os.path.join(experiment_dir, 'model.config'), 'rb') as config_file:
        config = pickle.load(config_file)

    state_dicts = torch.load(os.path.join(experiment_dir, checkpoint_name), map_location=torch.device(device))

    model = model_class.load_from_config_dict(config)
    model.load_state_dict(state_dicts['model'])
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer.load_state_dict(state_dicts['optimizer'])
    loss_func = nn.CrossEntropyLoss()
    loss_func.load_state_dict(state_dicts['loss_func'])
    train_engine = create_supervised_trainer(model, optimizer, loss_func, device, non_blocking=True)
    train_engine.load_state_dict(state_dicts['train_engine'])
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=5e-3, gamma=0.9,
                                                     last_epoch=train_engine.state.epoch, cycle_momentum=False)
    lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])
    return model, optimizer, loss_func, lr_scheduler, train_engine
