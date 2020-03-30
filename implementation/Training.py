import os
import pickle
from typing import Optional, Union

import torch
from torch import optim
from torch.utils.data import DataLoader

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.engines import common

from implementation.Models import KinshipClassifier, PairCombinationModule
from implementation.DataHandling import KinshipDataset
from implementation.Utils import *

# class opt_parameters:
#     def __init__(self, ):


def finetune_model(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
                   device=None, base_lr=1e-4, max_lr=1e-3, lr_gamma=0.9,
                   lr_decay_iters: Optional[Union[int, float]] = None, weight_decay=0.0,
                   loss_func=None, n_epochs=1, patience=-1, data_augmentation=True, weight_reg_coef=0.0,
                   combination_module=simple_concatenation, combination_size=KinshipClassifier.FACENET_OUT_SIZE * 2,
                   simple_fc_layers=None, custom_fc_layers=None, final_fc_layers=None, train_ds_name=None,
                   dev_ds_name=None, logging_rate=-1, saving_rate=-1, experiment_name=None, checkpoint_name=None,
                   hof_size=1, checkpoint_exp=None, data_path=None, verbose=True):
    if device is None:
        device = torch.device('cpu')

    if loss_func is None:
        loss_func = torch.nn.CrossEntropyLoss()

    if simple_fc_layers is None:
        simple_fc_layers = [1024]

    if custom_fc_layers is None:
        custom_fc_layers = [1024]

    if final_fc_layers is None:
        final_fc_layers = []

    if train_ds_name is None:
        train_ds_name = 'train'

    if dev_ds_name is None:
        dev_ds_name = 'dev'

    if checkpoint_exp is None:
        checkpoint_exp = experiment_name

    model = model_class(combination_module, combination_size, simple_fc_layers, custom_fc_layers, final_fc_layers)

    if data_path is None:
        data_path = os.path.join(project_path, 'data')
    processed_path = os.path.join(data_path, 'processed')

    partition_names = {'train': train_ds_name, 'dev': dev_ds_name}
    dataset_names = {partition: f"{name}_dataset.pkl" for partition, name in partition_names.items()}
    dataset_paths = {partition: os.path.join(data_path, dataset_names[partition]) for partition in dataset_names}
    raw_paths = {partition: os.path.join(processed_path, partition_names[partition]) for partition in dataset_paths}

    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    datasets = {partition: KinshipDataset.get_dataset(dataset_paths[partition], raw_paths[partition],
                                                      relationships_path, data_augmentation and (partition == 'train'))
                for partition in raw_paths}

    dataloaders = {partition: DataLoader(datasets[partition], batch_size=batch_size, shuffle=(partition == 'train'),
                                         num_workers=num_workers, pin_memory=pin_memory) for partition in datasets}

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.AdamW(params_to_train, lr=base_lr, weight_decay=weight_decay)
    if lr_decay_iters is None:
        lr_decay_iters = len(dataloaders['train'])
    elif lr_decay_iters <= 1.0:
        lr_decay_iters = int(len(dataloaders['train']) * lr_decay_iters)
    else:
        lr_decay_iters = int(lr_decay_iters)
    stepsize_up = lr_decay_iters // 2
    stepsize_down = lr_decay_iters - stepsize_up
    assert lr_decay_iters > 0
    assert stepsize_up + stepsize_down == lr_decay_iters
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=stepsize_up,
                                               step_size_down=stepsize_down, mode='exp_range', gamma=lr_gamma,
                                               cycle_momentum=False)

    regularized_loss = lambda y, y_pred: loss_func(y, y_pred) + \
                                         weight_reg_coef * sum(map(torch.abs, model.combination_module.weights))

    train_engine = create_supervised_trainer(model, optimizer, loss_fn=regularized_loss, device=device,
                                             non_blocking=non_blocking)

    if checkpoint_exp is not None and checkpoint_name is not None and verbose:
        experiment_dir = os.path.join(project_path, 'experiments', checkpoint_exp)
        model, optimizer, loss_func, lr_scheduler, train_engine = load_checkpoint(model_class, experiment_dir,
                                                                                  checkpoint_name, device)

    eval_engine = create_supervised_evaluator(model, metrics=dict(accuracy=Accuracy(), cross_entropy=Loss(loss_func)),
                                              device=device, non_blocking=non_blocking)

    metrics = {}

    if logging_rate > 0 and verbose:
        metrics['ce_history'] = []
        metrics['smoothed_loss_history'] = []
        beta = 0.98
        avg_loss = 0.0

        @train_engine.on(Events.ITERATION_COMPLETED(every=logging_rate))
        def log_iteration_training_metrics(engine):
            nonlocal metrics
            metrics['ce_history'].append(engine.state.output)

        @train_engine.on(Events.ITERATION_COMPLETED(every=logging_rate))
        def log_smoothed_lr(engine: Engine):
            nonlocal avg_loss
            avg_loss = (avg_loss * beta) + (engine.state.output * (1 - beta))
            metrics['smoothed_loss_history'].append(
                avg_loss / (1 - (beta ** (len(metrics['smoothed_loss_history']) + 1))))

        figs_path = os.path.join(PROJECT_ROOT, 'figs')
        if not os.path.isdir(figs_path):
            os.mkdir(figs_path)

        figs_path = os.path.join(figs_path, experiment_name)
        if not os.path.isdir(figs_path):
            os.mkdir(figs_path)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_metrics(engine):
            plot_metric(metrics['smoothed_loss_history'], f"Smoothed loss epoch #{engine.state.epoch}",
                        "Cross Entropy", index_scale=logging_rate, figs_path=figs_path)

    if patience >= 0:
        common.add_early_stopping_by_val_score(patience, eval_engine, train_engine, 'accuracy')

    # Replaced by setup_common_training_handlers
    # nan_terminate = TerminateOnNan()
    # train_engine.add_event_handler(Events.ITERATION_COMPLETED, nan_terminate)

    if verbose:
        @train_engine.on(Events.EPOCH_COMPLETED)
        def print_training_metrics(engine):
            print(f"Finished epoch {engine.state.epoch}")
            if train_ds_name == dev_ds_name:
                print(f"Epoch {engine.state.epoch}: CE = {engine.state.output}")
                metrics['final_dev_loss'] = engine.state.output
                return
            eval_engine.run(dataloaders['dev'])
            metrics['final_dev_loss'] = eval_engine.state.metrics['cross_entropy']
            print(f"Epoch {engine.state.epoch}: CE = {eval_engine.state.metrics['cross_entropy']}, "
                  f"Acc = {eval_engine.state.metrics['accuracy']}")

    # Replaced by setup_common_training_handlers
    # @train_engine.on(Events.ITERATION_COMPLETED)
    # def change_lr(engine):
    #     lr_scheduler.step()

    to_save = None
    output_path = None

    if saving_rate > 0 and verbose:
        if experiment_name is None:
            print("Warning: saving rate specified but experiment name is None")
            exit()

        experiment_path = os.path.join(project_path, 'experiments', experiment_name)
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

        with open(os.path.join(experiment_path, 'model.config'), 'wb+') as config_file:
            pickle.dump(model.get_configuration(), config_file)

        to_save = {'model': model,
                   'optimizer': optimizer,
                   'loss_func': loss_func,
                   'lr_scheduler': lr_scheduler,
                   'train_engine': train_engine}
        output_path = experiment_path

        best_models_dir = os.path.join(output_path, 'best_models')
        if not os.path.isdir(best_models_dir):
            os.mkdir(best_models_dir)
        common.save_best_model_by_val_score(best_models_dir, eval_engine, model, 'accuracy', n_saved=hof_size,
                                            trainer=train_engine, tag='acc')

        # Replaced by setup_common_training_handlers
        # checkpointer = ModelCheckpoint(experiment_path, 'iter', n_saved=50,
        #                                global_step_transform=lambda engine, _:
        #                                f"{engine.state.epoch}-{engine.state.iteration}", require_empty=False)
        # train_engine.add_event_handler(Events.ITERATION_COMPLETED(every=saving_rate), checkpointer, to_save)

    common.setup_common_training_handlers(train_engine, to_save=to_save, save_every_iters=saving_rate,
                                          output_path=output_path, lr_scheduler=lr_scheduler, with_pbars=True,
                                          with_pbar_on_iters=True, log_every_iters=1, device=device)

    # Replaced by setup_common_training_handlers
    # train_pbar = ProgressBar()
    # train_pbar.attach(train_engine)
    #
    eval_pbar = ProgressBar(persist=False, desc="Evaluation")
    eval_pbar.attach(eval_engine)

    print("Running on:", device)
    train_engine.run(dataloaders['train'], max_epochs=n_epochs)
    if not verbose:
        eval_engine.run(dataloaders['dev'])
        metrics['final_dev_score'] = eval_engine.state.metrics['accuracy']
    return model, metrics


def find_lr(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
            device=None, min_lr=1e-10, max_lr=1e+1, lr_increase=1.006, loss_func=None, beta=0.98,
            data_augmentation=True, combination_module=simple_concatenation,
            combination_size=KinshipClassifier.FACENET_OUT_SIZE * 2,
            simple_fc_layers=None, custom_fc_layers=None, final_fc_layers=None, train_ds_name=None,
            dev_ds_name=None):
    """
    Algorithm to find optimal lr, based on an algorithm from somewhere I lost the link to.
    """
    if device is None:
        device = torch.device('cpu')

    if loss_func is None:
        loss_func = torch.nn.CrossEntropyLoss()

    if simple_fc_layers is None:
        simple_fc_layers = [1024]

    if custom_fc_layers is None:
        custom_fc_layers = [1024]

    if final_fc_layers is None:
        final_fc_layers = []

    if train_ds_name is None:
        train_ds_name = 'train_dataset.pkl'

    if dev_ds_name is None:
        dev_ds_name = 'dev_dataset.pkl'

    model = model_class(combination_module, combination_size, simple_fc_layers, custom_fc_layers, final_fc_layers)

    data_path = os.path.join(project_path, 'data')
    processed_path = os.path.join(data_path, 'processed')

    dataset_path = os.path.join(data_path, train_ds_name)
    raw_path = os.path.join(processed_path, 'train')
    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    dataset = KinshipDataset.get_dataset(dataset_path, raw_path, relationships_path, data_augmentation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory)

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.AdamW(params_to_train, lr=min_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_increase)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler)

    train_engine = create_supervised_trainer(model, optimizer, loss_fn=loss_func, device=device,
                                             non_blocking=non_blocking)

    lr_list = []
    loss_list = []
    avg_loss = 0.0
    smoothed_loss_list = []
    current_lr = min_lr
    max_loss = 10.0

    @train_engine.on(Events.ITERATION_COMPLETED)
    def increment_lr_and_log(engine: Engine):
        nonlocal current_lr
        nonlocal avg_loss
        lr_list.append(current_lr)
        loss_list.append(engine.state.output)
        avg_loss = (avg_loss * beta) + (loss_list[-1] * (1 - beta))
        smoothed_loss_list.append(avg_loss / (1 - (beta ** len(lr_list))))
        lr_scheduler.step()
        passed_max_loss = loss_list[-1] >= max_loss
        passed_max_lr = lr_list[-1] >= max_lr
        stop_conditions = {'max_loss': passed_max_loss, 'max_lr': passed_max_lr}
        if any([item for _, item in stop_conditions.items()]):
            print("Stopping the computation. Stop reasons:")
            print(stop_conditions)
            plt.figure()
            plt.semilogx(lr_list, smoothed_loss_list)
            plt.xlabel('LR')
            plt.ylabel('CE')
            plt.show()
            engine.should_terminate = True
        current_lr *= lr_increase

    @train_engine.on(Events.EPOCH_COMPLETED)
    def plot_losses(engine):
        plt.figure()
        plt.semilogx(lr_list, smoothed_loss_list)
        plt.xlabel('LR')
        plt.ylabel('CE')
        plt.show()

    train_pbar = ProgressBar()
    train_pbar.attach(train_engine)

    train_engine.run(dataloader, max_epochs=4)

    return lr_list, loss_list


if __name__ == "__main__":
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    combination_module = PairCombinationModule(feature_combination_list, KinshipClassifier.FACENET_OUT_SIZE, 0.7)
    _, _ = finetune_model(KinshipClassifier, PROJECT_ROOT, 128, num_workers=8, device=device,
                          base_lr=3e-4, max_lr=9e-3, lr_gamma=0.8, lr_decay_iters=0.7,
                          n_epochs=10, weight_decay=3e-4, simple_fc_layers=[256],
                          custom_fc_layers=[2048, 256], final_fc_layers=[512], combination_module=combination_module,
                          combination_size=combination_module.output_size(), data_augmentation=True,
                          train_ds_name='train', dev_ds_name='mini',
                          pin_memory=True, non_blocking=True, logging_rate=10, loss_func=None,
                          saving_rate=500, experiment_name='smaller_model')
    # lrs, losses = find_lr(KinshipClassifier, PROJECT_ROOT, 64, num_workers=8, device=device, lr_increase=1.01,
    #                       min_lr=4e-7, max_lr=1e+1, simple_fc_layers=[512], custom_fc_layers=[2048, 512], final_fc_layers=[512],
    #                       combination_module=combination_module, combination_size=combination_module.output_size(),
    #                       data_augmentation=False, train_ds_name='mini_dataset.pkl', dev_ds_name='mini_dataset.pkl',
    #                       pin_memory=True, non_blocking=True)

