import os
import pickle

import torch
from torch import optim
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.handlers import EarlyStopping, TerminateOnNan
from torch.utils.data import DataLoader

from implementation.Models import KinshipClassifier, PairCombinationModule
from implementation.DataHandling import KinshipDataset
from implementation.Utils import *
from ax import *

PROJECT_ROOT = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle"

# class opt_parameters:
#     def __init__(self, ):


def finetune_model(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
                   device=None, lr=1e-4, max_lr=1e-3, lr_gamma=0.9, lr_decay_iters=None, weight_decay=0.0,
                   loss_func=None, n_epochs=1, patience=-1, data_augmentation=True,
                   combination_module=simple_concatenation, combination_size=KinshipClassifier.FACENET_OUT_SIZE * 2,
                   simple_fc_layers=None, custom_fc_layers=None, final_fc_layers=None, train_ds_name=None,
                   dev_ds_name=None, logging_rate=-1, saving_rate=-1, experiment_name=None, checkpoint_name=None):
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

    dataset_names = {'train': train_ds_name, 'dev': dev_ds_name}
    dataset_paths = {partition: os.path.join(data_path, dataset_names[partition]) for partition in dataset_names}
    raw_paths = {partition: os.path.join(processed_path, partition) for partition in dataset_paths}

    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    datasets = {partition: KinshipDataset.get_dataset(dataset_paths[partition], raw_paths[partition],
                                                      relationships_path, data_augmentation and (partition == 'train'))
                for partition in raw_paths}

    dataloaders = {partition: DataLoader(datasets[partition], batch_size=batch_size, shuffle=(partition == 'train'),
                                         num_workers=num_workers, pin_memory=pin_memory) for partition in datasets}

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.AdamW(params_to_train, lr=lr, weight_decay=weight_decay)
    lr_decay_iters = len(dataloaders['train']) if lr_decay_iters is None else lr_decay_iters
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=max_lr, step_size_up=lr_decay_iters//2,
                                               mode='exp_range', gamma=lr_gamma, cycle_momentum=False)

    train_engine = create_supervised_trainer(model, optimizer, loss_fn=loss_func, device=device,
                                             non_blocking=non_blocking)

    if experiment_name is not None and checkpoint_name is not None:
        experiment_dir = os.path.join(project_path, 'experiments', experiment_name)
        model, optimizer, loss_func, lr_scheduler, train_engine = load_checkpoint(model_class, experiment_dir,
                                                                                  checkpoint_name, device)

    eval_engine = create_supervised_evaluator(model, metrics=dict(accuracy=Accuracy(), cross_entropy=Loss(loss_func)),
                                              device=device, non_blocking=non_blocking)

    metrics = {}

    if logging_rate > 0:
        history_max_size = 5000
        metrics['ce_history'] = []

        @train_engine.on(Events.ITERATION_COMPLETED(every=logging_rate))
        def log_iteration_training_metrics(engine):
            nonlocal metrics
            if len(metrics['ce_history']) > history_max_size:
                metrics['ce_history'] = metrics['ce_history'][int(history_max_size)/5:]
            metrics['ce_history'].append(engine.state.output)

        beta = 0.98
        avg_loss = 0.0
        metrics['smoothed_loss_history'] = []

        @train_engine.on(Events.ITERATION_COMPLETED(every=logging_rate))
        def log_smoothed_lr(engine: Engine):
            nonlocal avg_loss
            avg_loss = (avg_loss * beta) + (engine.state.output * (1 - beta))
            metrics['smoothed_loss_history'].append(
                avg_loss / (1 - (beta ** (len(metrics['smoothed_loss_history']) + 1))))

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_metrics(engine):
            plot_metric(metrics['smoothed_loss_history'], f"Smoothed loss epoch #{engine.state.epoch}", "Cross Entropy")

    if patience >= 0:
        # Add early stopping handler
        es_handler = EarlyStopping(patience=patience,
                                   score_function=lambda engine: -engine.state.metrics['cross_entropy'],
                                   trainer=train_engine)
        eval_engine.add_event_handler(Events.COMPLETED, es_handler)

    nan_terminate = TerminateOnNan()
    train_engine.add_event_handler(Events.ITERATION_COMPLETED, nan_terminate)

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

    @train_engine.on(Events.ITERATION_COMPLETED)
    def change_lr(engine):
        lr_scheduler.step()

    if saving_rate > 0:
        if experiment_name is None:
            print("Warning: saving rate specified but experiment name is None")
            exit()

        experiment_path = os.path.join(project_path, 'experiments', experiment_name)
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

        with open(os.path.join(experiment_path, 'model.config'), 'wb+') as config_file:
            pickle.dump(model.get_configuration(), config_file)

        checkpointer = ModelCheckpoint(experiment_path, 'iter', n_saved=50,
                                       global_step_transform=lambda engine, _:
                                       f"{engine.state.epoch}-{engine.state.iteration}", require_empty=False)
        train_engine.add_event_handler(Events.ITERATION_COMPLETED(every=saving_rate), checkpointer,
                                       {'model': model, 'optimizer': optimizer, 'loss_func': loss_func,
                                        'lr_scheduler': lr_scheduler, 'train_engine': train_engine})

    train_pbar = ProgressBar()
    train_pbar.attach(train_engine)

    eval_pbar = ProgressBar(desc="Evaluation")
    eval_pbar.attach(eval_engine)

    print(model)
    print("Running on:", device)
    train_engine.run(dataloaders['train'], max_epochs=n_epochs)

    return model, metrics


def find_lr(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
            device=None, min_lr=1e-10, max_lr=1e+1, lr_increase=1.006, loss_func=None, beta=0.98,
            data_augmentation=True, combination_module=simple_concatenation,
            combination_size=KinshipClassifier.FACENET_OUT_SIZE * 2,
            simple_fc_layers=None, custom_fc_layers=None, final_fc_layers=None, train_ds_name=None,
            dev_ds_name=None):
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
                          lr=3e-4, max_lr=9e-3, lr_gamma=0.9, lr_decay_iters=95,
                          n_epochs=10, weight_decay=1e-4, simple_fc_layers=[512],
                          custom_fc_layers=[2048, 512], final_fc_layers=[512], combination_module=combination_module,
                          combination_size=combination_module.output_size(), data_augmentation=False,
                          train_ds_name='mini_dataset.pkl', dev_ds_name='mini_dataset.pkl',
                          pin_memory=True, non_blocking=True,
                          logging_rate=5, loss_func=None, saving_rate=100, experiment_name='ex3_no_aug')
    # lrs, losses = find_lr(KinshipClassifier, PROJECT_ROOT, 64, num_workers=8, device=device, lr_increase=1.01,
    #                       min_lr=4e-7, max_lr=1e+1, simple_fc_layers=[512], custom_fc_layers=[2048, 512], final_fc_layers=[512],
    #                       combination_module=combination_module, combination_size=combination_module.output_size(),
    #                       data_augmentation=False, train_ds_name='mini_dataset.pkl', dev_ds_name='mini_dataset.pkl',
    #                       pin_memory=True, non_blocking=True)

