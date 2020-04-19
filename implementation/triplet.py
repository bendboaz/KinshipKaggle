from typing import Dict
import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from ignite.utils import convert_tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from ignite.engine import Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.engines import common
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from implementation.Models import TripletNetwork, PairCombinationModule
from implementation.DataHandling import KinshipTripletDataset
from implementation.Utils import simple_concatenation, load_checkpoint, \
    PROJECT_ROOT, feature_combination_list


def triplet_prep_batch(batch, device=None, non_blocking=False):
    anchors, positives, negatives = batch
    return torch.stack([
        convert_tensor(anchors, device=device, non_blocking=non_blocking),
        convert_tensor(positives, device=device, non_blocking=non_blocking),
        convert_tensor(negatives, device=device, non_blocking=non_blocking)
        ], dim=1)


def create_triplet_trainer(model, optimizer, loss_fn, clip_val=None,
                          device=None, non_blocking=False,
                          prepare_batch=triplet_prep_batch,
                          output_transform=lambda x, y, y_pred, loss: loss.item(),
                          classify=True):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        clip_val (Optional[float]): value for gradient norm clipping.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        outputs = model(batch, triplet=True, classify=classify)
        labels = torch.tensor([0, 1], device=device).repeat(batch.shape[0], 1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        if clip_val is not None:
            clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), clip_val)
        optimizer.step()
        y_pred = outputs[1] if classify else None
        return output_transform(batch, labels, y_pred, loss)

    return Engine(_update)


def triplet_train(project_path, data_path, model_kwargs: Dict,
                  with_classification=True, train_ds=None, dev_ds=None,
                  batch_size=50, num_workers=0, device=None,
                  optimizer_params: Dict = None, n_epochs=1, patience=-1,
                  data_augmentation=True, grad_clip_val=None,
                  weight_reg_coef=0.0, log_every_iters=-1, save_every_iters=-1,
                  experiment_name=None, checkpoint_name=None,
                  checkpoint_exp=None, hof_size=1, verbose=True):
    if device is None:
        device = torch.device('cpu')

    if train_ds is None:
        train_ds = 'train'

    if dev_ds is None:
        dev_ds = 'dev'

    model = TripletNetwork(**model_kwargs)
    if data_path is None:
        data_path = os.path.join(project_path, 'data')

    processed_path = os.path.join(data_path, 'processed')

    partition_names = {'train': train_ds, 'dev': dev_ds}
    dataset_names = {partition: f"{name}_triplet_dataset.pkl"
                     for partition, name in partition_names.items()}
    dataset_paths = {partition: os.path.join(data_path, dataset_names[partition])
                     for partition in dataset_names}
    raw_paths = {partition: os.path.join(processed_path, partition_names[partition])
                 for partition in dataset_paths}

    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    datasets = {partition: KinshipTripletDataset.get_dataset(
                                                    dataset_paths[partition],
                                                    raw_paths[partition],
                                                    relationships_path,
                                                    data_augmentation and (partition == 'train')
                                                    )
                for partition in raw_paths}
    max_iteration_sizes = {'train': int(2e5), 'dev': int(1e4)}
    samplers = {partition: RandomSampler(
                                         datasets[partition],
                                         replacement=True,
                                         num_samples=max_iteration_sizes[partition]
                                        )
                for partition in datasets}

    dataloaders = {partition: DataLoader(
                                         datasets[partition],
                                         sampler=samplers[partition],
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True
                                        )
                   for partition in datasets}
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer, lr_sched = get_optimizer_and_lr_sched(
                                                     optimizer_params,
                                                     params_to_train,
                                                     len(dataloaders['train'])
                                                    )
    if with_classification:
        unregularized_loss_func = nn.CrossEntropyLoss()

        def regularized_loss(y, y_pred):
            params_to_regularize = torch.cat(model.combination_module.weights)
            l1_loss = nn.L1Loss()(params_to_regularize, torch.zeros_like(params_to_regularize))
            if l1_loss == float('nan') or l1_loss == float('inf') or l1_loss == float('-inf'):
                raise ValueError(f"Regularization loss is {l1_loss}!")
            return unregularized_loss_func(y, y_pred) + weight_reg_coef * l1_loss

        def aggregate_loss_func(outputs, labels):
            """
            For the output of TripletNetwork.forward
            with triplet and classify = True.
            :param outputs: tuple of (triplet_loss, scores).
                triplet_loss is a scalar tensor.
                scores is a tensor of shape (N, 2, num_classes),
            :param labels: tensor of shape (N, 2).
            :return: Scalar tensor with the total loss.
            """
            return outputs[0] + regularized_loss(outputs[1], labels)
        loss_func = aggregate_loss_func
    else:
        loss_func = lambda x: x

    train_engine = create_triplet_trainer(model, optimizer, loss_func,
                                          clip_val=grad_clip_val,
                                          device=device, non_blocking=True,
                                          classify=with_classification)

    if checkpoint_exp is not None and checkpoint_name is not None and verbose:
        experiment_dir = os.path.join(project_path, 'experiments', checkpoint_exp)
        model, optimizer, loss_func, lr_scheduler, train_engine = (
            load_checkpoint(TripletNetwork, experiment_dir,
                            checkpoint_name, device, loss_func=loss_func))

        train_engine.state.max_epochs += n_epochs

    eval_metrics = {'aggregate_loss': Loss(lambda x: x, output_transform=lambda output: output['loss'])}
    if with_classification:
        eval_metrics['accuracy'] = Accuracy()

    def _eval_process_func(engine, batch):
        model.eval()
        batch = triplet_prep_batch(batch, device=device, non_blocking=True)
        with torch.no_grad():
            output = model(batch, triplet=True, classify=with_classification)
            y_pred = output[1] if with_classification else None
            y = torch.tensor([0, 1], device=device).repeat(batch.shape[0], 1)
            return dict(triplet_loss=output[0], y_pred=y_pred, y=y)

    eval_engine = Engine(_eval_process_func)
    Loss(lambda x: x, output_transform=lambda output: output['triplet_loss'])\
        .attach(eval_engine, 'triplet_loss')
    if with_classification:
        Accuracy(output_transform=lambda output: (output['y_pred'].view(-1, 2),
                                                  output['y'].view(-1)))\
            .attach(eval_engine, 'accuracy')

    if patience >= 0:
        common.add_early_stopping_by_val_score(patience, eval_engine, train_engine, 'accuracy')

    to_save = None
    output_path = None

    if experiment_name is not None:
        experiment_path = os.path.join(project_path, 'experiments', experiment_name)
        if not os.path.isdir(experiment_path):
            os.makedirs(experiment_path)

        with open(os.path.join(experiment_path, 'model.config'), 'wb+') as config_file:
            pickle.dump(model.get_configuration(), config_file)

        if hof_size > 0:
            best_models_dir = os.path.join(experiment_path, 'best_models')
            if not os.path.isdir(best_models_dir):
                os.makedirs(best_models_dir)

            common.save_best_model_by_val_score(best_models_dir, eval_engine, model, 'accuracy', n_saved=hof_size,
                                                trainer=train_engine, tag='acc')

        if save_every_iters > 0 and verbose:
            to_save = {'model': model,
                       'optimizer': optimizer,
                       'lr_scheduler': lr_sched,
                       'train_engine': train_engine}
            output_path = experiment_path

    common.setup_common_training_handlers(train_engine, to_save=to_save, save_every_iters=save_every_iters,
                                          output_path=output_path, lr_scheduler=lr_sched, with_pbars=True,
                                          with_pbar_on_iters=True, log_every_iters=log_every_iters, device=device)

    eval_pbar = ProgressBar(persist=False, desc="Evaluation")
    eval_pbar.attach(eval_engine)

    print("Running on:", device)
    train_engine.run(dataloaders['train'], max_epochs=n_epochs)
    return model


def get_optimizer_and_lr_sched(optimizer_params: Dict, model_params,
                               default_cycle: int):
    base_lr = optimizer_params.get('base_lr', 1e-6)
    max_lr = optimizer_params.get('max_lr', 1e-3)
    weight_decay = optimizer_params.get('weight_decay', 1e-3)
    lr_decay_iters = optimizer_params.get('lr_decay_iters', default_cycle)
    lr_gamma = optimizer_params.get('lr_gamma', 1.0)
    if lr_decay_iters <= 1.0:
        lr_decay_iters = int(default_cycle * lr_decay_iters)

    optimizer = optim.AdamW(model_params, lr=base_lr, weight_decay=weight_decay)

    stepsize_up = lr_decay_iters // 2
    stepsize_down = lr_decay_iters - stepsize_up
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=stepsize_up,
                                               step_size_down=stepsize_down, mode='exp_range', gamma=lr_gamma,
                                               cycle_momentum=False)
    return optimizer, lr_scheduler


if __name__ == '__main__':
    project_path = PROJECT_ROOT
    data_path = os.path.join('/', 'home', 'boaz.ben-dov', 'gdrive', 'Colab Notebooks', 'KinshipKaggle', 'data')
    feature_combinations = [feature_combination_list[idx] for idx in [0, 1, 3, 8]]
    comb_module = PairCombinationModule(feature_combinations, TripletNetwork.FACENET_OUT_SIZE)
    model_kwargs = dict(combination_module=comb_module,
                        combination_size=comb_module.output_size(),
                        simple_fc_sizes=[1024], custom_fc_sizes=[1024],
                        final_fc_sizes=[], triplet_margin=0.5,
                        facenet_unfreeze_depth=2)
    train_ds = 'train'
    dev_ds = 'dev'
    batch_size = 50
    num_workers = 8
    device = torch.device(torch.cuda.current_device()
                          if torch.cuda.is_available() else 'cpu')
    optimizer_params = dict(base_lr=1e-6, max_lr=1e-3, lr_gamma=0.8,
                            weight_decay=1e-2, lr_decay_iters=0.6)
    n_epochs = 10
    patience = 3
    data_augmentation = True
    weight_reg_coef = 1e-2
    log_every_iters = 5
    save_every_iters = 1000
    experiment_name = 'triplet_1'

    triplet_train(project_path, data_path, model_kwargs,
                  with_classification=True, train_ds=train_ds, dev_ds=dev_ds,
                  batch_size=batch_size, num_workers=num_workers, device=device,
                  optimizer_params=optimizer_params, n_epochs=n_epochs, patience=patience,
                  data_augmentation=True, grad_clip_val=None,
                  weight_reg_coef=weight_reg_coef, log_every_iters=log_every_iters,
                  save_every_iters=save_every_iters, experiment_name=experiment_name,
                  hof_size=1)