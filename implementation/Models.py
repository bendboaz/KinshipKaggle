from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from facenet_pytorch import InceptionResnetV1

from implementation.Utils import get_dense_block

# Load feature extractors (VGGFace, FaceNet, ResNet on Imagenet)

# A (generic?) tool to take feature extractors, a combination function and create an end-to-end model.


class KinshipClassifier(nn.Module):
    FACENET_OUT_SIZE = 512

    def __init__(self, combination_func, combination_size, simple_fc_sizes: List[int], custom_fc_sizes: List[int],
                 final_fc_sizes: List[int], normalize_features=True) -> None:
        super().__init__()
        self.combination_func = combination_func
        self.combination_size = combination_size

        self.facenet = InceptionResnetV1(pretrained='vggface2', normalize_features=normalize_features)
        for param in self.facenet.parameters(recurse=True):
            param.requires_grad = False

        for param in self.facenet.last_linear.parameters():
            param.requires_grad = True

        self.simple_fc = get_dense_block(self.FACENET_OUT_SIZE * 2, simple_fc_sizes, nn.ReLU)
        self.custom_fc = get_dense_block(self.combination_size, custom_fc_sizes, nn.ReLU)

        self.final_bn = nn.BatchNorm1d(simple_fc_sizes[-1] + custom_fc_sizes[-1])
        self.classification_fc = get_dense_block(simple_fc_sizes[-1] + custom_fc_sizes[-1], final_fc_sizes + [2],
                                                 nn.ReLU)

    def forward(self, inputs, **kwargs):
        img1_batch = inputs[:, 0].squeeze(1)
        img2_batch = inputs[:, 1].squeeze(1)

        img1_features = F.relu(self.facenet(img1_batch))
        img2_features = F.relu(self.facenet(img2_batch))

        simple_branch = torch.cat([img1_features, img2_features], 1)
        simple_branch = F.relu(self.simple_fc(simple_branch))

        custom_branch = self.combination_func(img1_features, img2_features)
        custom_branch = F.relu(self.custom_fc(custom_branch))

        concat_vector = torch.cat([simple_branch, custom_branch], dim=1)
        concat_vector = self.final_bn(concat_vector)

        classification = self.classification_fc(concat_vector)
        return classification
