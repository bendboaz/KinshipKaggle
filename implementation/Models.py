from typing import List, Any
from collections import OrderedDict
from itertools import chain

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
                 final_fc_sizes: List[int], fc_dropout=0.6) -> None:
        super().__init__()
        self.combination_module = combination_module
        self.combination_size = combination_size
        self.simple_fc_sizes = simple_fc_sizes
        self.custom_fc_sizes = custom_fc_sizes
        self.final_fc_sizes = final_fc_sizes
        self.fc_dropout_prob = fc_dropout
        self.do_simple_fc = len(simple_fc_sizes) > 0
        self.do_custom_fc = len(custom_fc_sizes) > 0

        self.facenet = InceptionResnetV1(pretrained='vggface2')
        for param in self.facenet.parameters(recurse=True):
            param.requires_grad = False

        submodules_to_unfreeze = [self.facenet.block8]
        for param in chain(*map(lambda x: x.parameters(), submodules_to_unfreeze)):
            param.requires_grad = True

        self.facenet.last_linear = nn.Linear(1792, self.FACENET_OUT_SIZE)
        self.post_facenet_activation = nn.ReLU()
        self.post_facenet_dropout = nn.Dropout(self.fc_dropout_prob)

        if self.do_simple_fc:
            self.simple_fc = get_dense_block(self.FACENET_OUT_SIZE * 2, simple_fc_sizes, nn.ReLU,
                                             dropout_prob=self.fc_dropout_prob)

        if self.do_custom_fc:
            self.custom_fc = get_dense_block(self.combination_size, custom_fc_sizes, nn.ReLU,
                                             dropout_prob=self.fc_dropout_prob)

        self.pre_classification_activation = nn.ReLU()
        self.final_bn = nn.BatchNorm1d(simple_fc_sizes[-1] + custom_fc_sizes[-1])

        self.classification_fc = get_dense_block(simple_fc_sizes[-1] + custom_fc_sizes[-1], final_fc_sizes + [2],
                                                 nn.ReLU, dropout_prob=self.fc_dropout_prob)

    def forward(self, inputs, **kwargs):
        img1_batch = inputs[:, 0].squeeze(1)
        img2_batch = inputs[:, 1].squeeze(1)

        img1_features = self.post_facenet_activation(self.facenet(img1_batch))
        img2_features = self.post_facenet_activation(self.facenet(img2_batch))

        branch_outputs = []

        if self.do_simple_fc:
            simple_branch = torch.cat([img1_features, img2_features], 1)
            simple_branch = self.pre_classification_activation(self.simple_fc(simple_branch))
            branch_outputs.append(simple_branch)

        if self.do_custom_fc:
            custom_branch = self.combination_module(img1_features, img2_features)
            custom_branch = self.custom_fc(custom_branch)
            custom_branch = self.pre_classification_activation(custom_branch)
            branch_outputs.append(custom_branch)

        concat_vector = torch.cat(branch_outputs, dim=1)
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
        return cls(combination_module=comb_module, **dictionary)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample
    and a negative sample.
    Taken from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletNetwork(KinshipClassifier):
    FACENET_OUT_SIZE = 512

    def __init__(self, triplet_margin=1.0, facenet_unfreeze_depth=1, **kwargs):
        super(TripletNetwork, self).__init__(**kwargs)
        self.unfreezing_depth = facenet_unfreeze_depth
        self.triplet_margin = triplet_margin

        self.triplet_loss = TripletLoss(self.triplet_margin)

        # Freezing and unfreezing layers
        # (overriding the unfreezing scheme in KinshipClassifier):
        facenet_layers_order = [
            self.facenet.last_linear,
            self.facenet.block8,
            self.facenet.repeat_3,
            self.facenet.mixed_7a,
            self.facenet.repeat_2,
            self.facenet.mixed_6a,
        ]

        if self.unfreezing_depth > len(facenet_layers_order):
            raise NotImplementedError(f'Can only unfreeze '
                                      f'{len(facenet_layers_order)} layers, '
                                      f'requested {self.unfreezing_depth}.')

        for param in self.facenet.parameters():
            param.requires_grad = False

        unfrozen_submodules = facenet_layers_order[:self.unfreezing_depth]
        for param in chain(*map(lambda x: x.parameters(),
                                unfrozen_submodules)):
            param.requires_grad = True

    def forward(self, input: torch.Tensor, triplet=True, classify=False):
        """
        Forward through the network with either triplet loss,
        classification scores or both.
        :param input: A batch of either triplets or pairs,
            shape (N, 3, H, W, C) or (N, 2, H, W, C).
        :param triplet: Flag specifying whether to compute
            triplet loss.
            If this is set, each batch element must be
            a triplet of images.
        :param classify: Flag specifying whether to compute
            classification scores for the images.
            Can classify triplets (for each triplet t,
                        classifies (t[0], t[1]) and (t[0], t[2])).
            Can also classify simple pairs.
        :return: A tuple [triplet_loss], [classification_scores]:
            - triplet_loss is of shape () (scalar tensor).
            - classification_scores is of shape
                (N, 1, num_classes) or (N, 2, num_classes)
                (depending on whether pairs or
                triplets were classified).
        """
        output = ()
        if input.ndim == 4:
            # No batch dimension, manually add one:
            input = input.unsqueeze(dim=0)
        if triplet:
            if input.shape[1] != 3:
                raise ValueError(f'For triplet loss calculation, every batch '
                                 f'element needs to contain 3 elements. '
                                 f'Got {input.shape[1]}')

            anchor, positive, negative = input[:, 0], input[:, 1], input[:, 2]
            anchor_features = self.facenet(anchor)
            positive_features = self.facenet(positive)
            negative_features = self.facenet(negative)

            triplet_loss_value = self.triplet_loss(
                                                   anchor_features,
                                                   positive_features,
                                                   negative_features
                                                  )
            output = output + (triplet_loss_value,)

        if classify:
            if input.shape[1] == 3:
                pairs = [
                         (input[:, 0], input[:, 1]),
                         (input[:, 0], input[:, 2]),
                        ]
            elif input.shape[1] == 2:
                pairs = [
                         (input[:, 0], input[:, 1])
                        ]
            else:
                raise ValueError(f'Length of dimension 1 should be 2 or 3, '
                                 f'got {input.shape[1]}')

            pairs = [torch.stack(pair, dim=1) for pair in pairs]
            classification_scores = map(
                super(TripletNetwork, self).forward,
                pairs
            )
            classification_scores = torch.cat(list(map(
                                            lambda x: x.unsqueeze(1),
                                            classification_scores
                                                      )),
                                              dim=1)
            output = output + (classification_scores,)

        return output  # ([triplet_loss], [classification_scores])

    def get_configuration(self):
        config_dict = super().get_configuration()
        config_dict['triplet_margin'] = self.triplet_margin
        config_dict['facenet_unfreeze_depth'] = self.unfreezing_depth
        return config_dict

