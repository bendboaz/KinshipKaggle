import os
from typing import Any, List, Callable, Optional, Iterable
from enum import Enum
import pickle
import re
from argparse import ArgumentParser
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from ignite.utils import to_onehot
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from implementation.DataHandling import KinshipDataset
from implementation.Models import KinshipClassifier, TripletNetwork
from implementation.Utils import ALL_MODELS


class NetworkEnsemble(nn.Module):
    class DecisionMechanism(Enum):
        VOTING = 0
        AVG_POOLING = 1

    def __init__(self, model_paths: List[str], decision: DecisionMechanism = DecisionMechanism.VOTING,
                 num_classes: int = 2, model_score: Callable[[str], float] = None,
                 model_selector: Callable[[Iterable[float]], List[int]] = None) -> None:
        super().__init__()
        self.decision_mechanism = decision
        self.num_classes = num_classes
        self.models = []

        selected_models = model_paths
        if model_score is not None and model_selector is not None:
            selected_indices = model_selector(map(model_score, model_paths))
            selected_models = [model_paths[idx] for idx in sorted(selected_indices)]

        for path in selected_models:
            print(f"Loading model from {path}")
            model_config_path = os.path.join(path, 'model.config')
            model_params_path = os.path.join(path, 'best_models')
            # If no models are present, OS error will raise:
            param_file = os.path.join(model_params_path, os.listdir(model_params_path)[0])

            with open(model_config_path, 'rb') as config_file:
                config_dict = pickle.load(config_file)
                model_class = (TripletNetwork
                               if 'triplet_margin' in config_dict
                               else KinshipClassifier)
                model = model_class.load_from_config_dict(config_dict)

            parameters = torch.load(param_file)
            model.load_state_dict(parameters)
            self.models.append(model)
            self.add_module(os.path.split(path)[1], model)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: Any, **kwargs: Any):
        scores = torch.stack([model.classify_pairs(input) for model in self.models], dim=0)
        if self.decision_mechanism == self.DecisionMechanism.VOTING:
            predictions = scores.max(-1)[1]
            prediction_scores = to_onehot(predictions, self.num_classes).sum(dim=0).transpose(0, 1).type(torch.float)
        elif self.decision_mechanism == self.DecisionMechanism.AVG_POOLING:
            prediction_scores = torch.mean(scores, 0)
        else:
            raise NotImplementedError()

        return prediction_scores


def accuracy_extractor(model_path: str):
    best_models_path = os.path.join(model_path, 'best_models')
    params_file_name = os.listdir(best_models_path)[0]
    acc_regex = re.compile(r'\w*accuracy=(\d+\.\d+)\.pth')
    score_str = acc_regex.match(params_file_name).group(1)
    return float(score_str)


def threshold_selector(scores: Iterable[float], threshold: float = 0.0):
    return list(map(lambda pair: pair[0], filter(lambda ind_score: ind_score[1] >= threshold, enumerate(scores))))


def topk_selector(scores: Iterable[float], k: int = 1):
    return list(map(lambda x: x[0], sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]))


def test_ensemble(data_path: str, model_paths: List[str], threshold: Optional[float], top_k: Optional[int],
                  sample_set: str, decision: NetworkEnsemble.DecisionMechanism, device=None, batch_size=50,
                  num_workers=4):
    device = torch.device(device if device is not None else 'cpu')
    processed_path = os.path.join(data_path, 'processed')

    dataset_name = f'{sample_set}_dataset.pkl'
    dataset_path = os.path.join(data_path, dataset_name)
    raw_path = os.path.join(processed_path, sample_set)

    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    dataset = KinshipDataset.get_dataset(dataset_path, raw_path, relationships_path, data_augmentation=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    selector = None
    if threshold is not None:
        selector = partial(threshold_selector, threshold=threshold)
    if top_k is not None:
        if threshold is not None:
            raise Warning('Both threshold and top-k filtering requested.')
        selector = partial(topk_selector, k=top_k)

    ensemble = NetworkEnsemble(model_paths, decision, 2, accuracy_extractor, selector)

    evaluator = create_supervised_evaluator(ensemble, {'accuracy': Accuracy()}, device=device, non_blocking=True)
    ProgressBar(persist=False).attach(evaluator)

    evaluator.run(dataloader)
    return evaluator.state.metrics['accuracy']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiments_dir', type=str)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--decision_type', nargs='*', choices=['AVG_POOLING', 'VOTING'])
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    full_paths = list(map(lambda exp: os.path.join(args.experiments_dir, exp), ALL_MODELS))
    data_path = args.data_path
    if data_path is None:
        data_path = os.path.dirname(args.experiments_dir)

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    for decision_method in args.decision_type:
        score = test_ensemble(data_path, full_paths, args.threshold, args.top_k, 'dev',
                              NetworkEnsemble.DecisionMechanism[decision_method], device, args.batch_size,
                              args.num_workers)
        print(f"Score for {decision_method}: {score:.5f}")
