import os
from argparse import ArgumentParser
from functools import partial

import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from implementation.DataHandling import KinshipDataset
from implementation.ensemble import NetworkEnsemble, accuracy_extractor, topk_selector, threshold_selector, ALL_MODELS


def test_collate_fn(samples):
    image_pairs, names = zip(*samples)
    return torch.stack(list(image_pairs), dim=0), list(names)


def infer(data_path, models_list, decision_type, threshold, top_k, batch_size, device, num_workers):
    device = torch.device('cpu' if device is None else device)

    raw_test_path = os.path.join(data_path, 'raw', 'test')
    test_pickle_path = os.path.join(data_path, 'test.pkl')
    sample_submssion_path = os.path.join(data_path, 'raw', 'sample_submission.csv')
    test_set = KinshipDataset.get_test_dataset(test_pickle_path, raw_test_path, sample_path=sample_submssion_path)
    results = {'img_pair': [], 'is_related': []}

    selector = None
    if threshold is not None:
        selector = partial(threshold_selector, threshold=threshold)
    if top_k is not None:
        if threshold is not None:
            raise Warning('Both threshold and top-k filtering requested.')
        selector = partial(topk_selector, k=top_k)

    model = NetworkEnsemble(models_list, decision_type, 2, accuracy_extractor, selector)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
                        collate_fn=test_collate_fn, pin_memory=True)
    model.to(device, non_blocking=True)
    print(f"Number of test pairs: {len(test_set)}")
    for pairs, names in tqdm(loader, total=len(loader)):
        pairs = pairs.to(device, non_blocking=True)
        results['img_pair'].extend(names)
        predictions = model(pairs)
        predictions = torch.max(predictions, -1)[1].type(torch.int)
        results['is_related'].extend(predictions.tolist())

    results_df = pd.DataFrame.from_dict(results, orient='columns')
    assert set(results_df['is_related'].to_list()) == {0, 1}
    return results_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('experiments_path', type=str)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--decision_type', type=str, nargs='*', choices=['AVG_POOLING', 'VOTING'])
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()
    relevant_models = ALL_MODELS if args.single_model is None else [args.single_model]
    full_paths = list(map(lambda exp: os.path.join(args.experiments_path, exp), relevant_models))
    data_path = args.data_path
    if data_path is None:
        data_path = os.path.dirname(args.experiments_path)

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    for decision_method in args.decision_type:
        results_df = infer(args.data_path, full_paths, NetworkEnsemble.DecisionMechanism[decision_method],
                           args.threshold, args.top_k, args.batch_size, device, args.num_workers)
        results_df.to_csv(os.path.join(args.save_dir, f'{decision_method}.csv'), index=False)
