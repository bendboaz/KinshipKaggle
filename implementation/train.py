from argparse import ArgumentParser
from typing import Union

import torch

from implementation.Models import KinshipClassifier, PairCombinationModule
from implementation.Utils import PROJECT_ROOT, feature_combination_list
from implementation.Training import finetune_model
from implementation.DataHandling import KinshipDataset


if __name__ == "__main__":
    parser = ArgumentParser(description="A script to train your neural network, new one or from checkpoint.")
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory (needs to contain \'processed\' and \'raw\' directories, as'
                             'well as \'{train/dev/mini}_dataset.pkl\'')
    parser.add_argument('--simple_fc', nargs='+', default=None, type=int,
                        help='Layer sizes for the simple branch of the network.')
    parser.add_argument('--custom_fc', nargs='+', default=None, type=int,
                        help='Layer sizes for the custom (combinations) branch of the network.')
    parser.add_argument('--final_fc', nargs='*', default=None, type=int,
                        help='Layer sizes for the final classification.')
    combinations = parser.add_mutually_exclusive_group()
    combinations.add_argument('--all_combs', action='store_true', help='Flag to use all pair combinations.')
    combinations.add_argument('--comb_filter', nargs='*', default=None, type=int,
                              help='List of combinations to use (if not using --all_combs).')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use only CPU in training. Otherwise use GPU is available.')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='Starting lr for the cyclic scheduling.')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Top lr for the cyclic scheduling.')
    parser.add_argument('--lr_gamma', type=float, default=1.0,
                        help='Parameter by which to decay max_lr each cycle.')
    parser.add_argument('--lr_cycle', type=float, default=1.0,
                        help='Length of each up-down cycle.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay coefficient for AdamW optimizer.')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers to load the batches.')
    parser.add_argument('--no_augmentation', action='store_false',
                        help='Specify this to not use data augmentations.')
    parser.add_argument('--train_ds', type=str, choices=['train', 'dev', 'mini'],
                        help='Name of dataset to use for training.')
    parser.add_argument('--val_ds', type=str, choices=['train', 'dev', 'mini'],
                        help='Name of dataset to use for validation.')
    parser.add_argument('--log_every_iters', type=int, default=1,
                        help='Frequency of loss logging for the convergence graphs.')
    parser.add_argument('--save_every_iters', type=int, default=-1,
                        help='Frequency of checkpoints.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (Used for the checkpoint directory). '
                             'MUST BE DIFFERENT FROM checkpoint_exp.')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Name of checkpoint to load. Must exist in the directory specified by \'checkpoint_exp\'.')
    parser.add_argument('--checkpoint_exp', type=str, default=None,
                        help='Name of experiment from which to load the checkpoint.')
    args = parser.parse_args()

    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() and not args.use_cpu \
        else torch.device('cpu')
    combination_list = feature_combination_list
    if not args.all_combs:
        combination_list = [combination_list[i] for i in sorted(args.comb_filter)]

    combination_module = PairCombinationModule(combination_list, KinshipClassifier.FACENET_OUT_SIZE, 0.7)
    _, _ = finetune_model(KinshipClassifier, PROJECT_ROOT, args.batch_size, num_workers=8, device=device,
                          base_lr=args.base_lr, max_lr=args.max_lr, lr_gamma=args.lr_gamma,
                          lr_decay_iters=args.lr_cycle, n_epochs=args.n_epochs, weight_decay=args.weight_decay,
                          simple_fc_layers=args.simple_fc, custom_fc_layers=args.custom_fc,
                          final_fc_layers=args.final_fc, combination_module=combination_module,
                          combination_size=combination_module.output_size(), data_augmentation=args.no_augmentation,
                          train_ds_name=args.train_ds, dev_ds_name=args.val_ds,
                          pin_memory=True, non_blocking=True, logging_rate=args.log_every_iters, loss_func=None,
                          saving_rate=args.save_every_iters, experiment_name=args.experiment_name,
                          checkpoint_exp=args.checkpoint_exp, checkpoint_name=args.checkpoint_name,
                          data_path=args.data_dir)
