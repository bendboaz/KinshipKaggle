from typing import List, Any
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from facenet_pytorch import InceptionResnetV1

from implementation.Utils import get_dense_block

# Load feature extractors (VGGFace, FaceNet, ResNet on Imagenet)

# A (generic?) tool to take feature extractors, a combination function and create an end-to-end model.


class PairCombinationModule(nn.Module):
    def __init__(self, combination_list, input_size, dropout_prob=0.5) -> None:
        super().__init__()
        self.combinations = combination_list
        self.weights = [nn.Parameter(torch.ones(1)/(combination(torch.ones(1), torch.ones(1)) + 1e-7),
                                     requires_grad=True)
                        for combination in self.combinations]
        for ind, weight in enumerate(self.weights):
            self.register_parameter(f"combination{ind}_weight", weight)

        self.dropout = nn.Dropout(dropout_prob)
        self.input_size = input_size

    def forward(self, x1, x2):
        masks = self.dropout(torch.ones(len(self.combinations)))
        comb_maps = [(mask * wi * combination(x1, x2))
                     for wi, combination, mask in zip(self.weights, self.combinations, masks)]
        return torch.cat(comb_maps, dim=1)

    def output_size(self):
        return self.input_size * len(self.combinations)

    def get_configuration(self):
        return OrderedDict(
            combination_list=self.combinations,
            input_size=self.input_size
        )

    @classmethod
    def load_from_config_dict(cls, dictionary):
        return cls(**dictionary)


class KinshipClassifier(nn.Module):
    FACENET_OUT_SIZE = 512

    def __init__(self, combination_module, combination_size, simple_fc_sizes: List[int], custom_fc_sizes: List[int],
                 final_fc_sizes: List[int]) -> None:
        super().__init__()
        self.combination_module = combination_module
        self.combination_size = combination_size
        self.simple_fc_sizes = simple_fc_sizes
        self.custom_fc_sizes = custom_fc_sizes
        self.final_fc_sizes = final_fc_sizes

        self.facenet = InceptionResnetV1(pretrained='vggface2')
        for param in self.facenet.parameters(recurse=True):
            param.requires_grad = False

        self.facenet.last_linear = nn.Linear(1792, self.FACENET_OUT_SIZE)
        self.post_facenet_activation = nn.ReLU()

        self.simple_fc = get_dense_block(self.FACENET_OUT_SIZE * 2, simple_fc_sizes, nn.ReLU)

        self.custom_fc = get_dense_block(self.combination_size, custom_fc_sizes, nn.ReLU)

        self.before_classification_activation = nn.ReLU()
        self.final_bn = nn.BatchNorm1d(simple_fc_sizes[-1] + custom_fc_sizes[-1])

        self.classification_fc = get_dense_block(simple_fc_sizes[-1] + custom_fc_sizes[-1], final_fc_sizes + [2],
                                                 nn.ReLU)

    def forward(self, inputs, **kwargs):
        img1_batch = inputs[:, 0].squeeze(1)
        img2_batch = inputs[:, 1].squeeze(1)

        img1_features = self.post_facenet_activation(self.facenet(img1_batch))
        img2_features = self.post_facenet_activation(self.facenet(img2_batch))

        simple_branch = torch.cat([img1_features, img2_features], 1)
        simple_branch = self.before_classification_activation(self.simple_fc(simple_branch))

        custom_branch = self.combination_module(img1_features, img2_features)
        custom_branch = self.custom_fc(custom_branch)
        custom_branch = self.before_classification_activation(custom_branch)

        concat_vector = torch.cat([simple_branch, custom_branch], dim=1)
        concat_vector = self.final_bn(concat_vector)

        classification = self.classification_fc(concat_vector)
        return classification

    def get_configuration(self):
        return OrderedDict(
            comb_module_dict=self.combination_module.get_configuration(),
            combination_size=self.combination_size,
            simple_fc_sizes=self.simple_fc_sizes,
            custom_fc_sizes=self.custom_fc_sizes,
            final_fc_sizes=self.final_fc_sizes,
        )

    @classmethod
    def load_from_config_dict(cls, dictionary):
        comb_module = PairCombinationModule.load_from_config_dict(dictionary['comb_module_dict'])
        del dictionary['comb_module_dict']
        return cls(comb_module, **dictionary)
