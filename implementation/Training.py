import os
import pickle

import torch
from torch import optim
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers.checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from implementation.Models import KinshipClassifier, PairCombinationModule
from implementation.DataHandling import KinshipDataset
from implementation.Utils import *
from ax import *

PROJECT_ROOT = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle"

# class opt_parameters:
#     def __init__(self, ):


def finetune_model(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
                   device=None, lr=1e-4, lr_decay=1.0, lr_decay_iters=None, weight_decay=0.0, loss_func=None, n_epochs=1,
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
                                                      relationships_path) for partition in raw_paths}

    dataloaders = {partition: DataLoader(datasets[partition], batch_size=batch_size, shuffle=(partition == 'train'),
                                         num_workers=num_workers, pin_memory=pin_memory) for partition in datasets}

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.AdamW(params_to_train, lr=lr, weight_decay=weight_decay)
    lr_decay_iters = len(dataloaders['train']) if lr_decay_iters is None else lr_decay_iters
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=lr_decay)

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
            if not isinstance(engine.state.output, float):
                print(f"Encountered unxepected loss value: {engine.state.output} in iteration {engine.state.iteration}")
            nonlocal metrics
            if len(metrics['ce_history']) > history_max_size:
                metrics['ce_history'] = metrics['ce_history'][int(history_max_size)/5:]
            metrics['ce_history'].append(engine.state.output)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def plot_metrics(engine):
            plot_metric(metrics['ce_history'], f"CE loss after epoch #{engine.state.epoch}", "Cross Entropy")

    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_training_metrics(engine):
        print(f"Finished epoch {engine.state.epoch}")
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

    print(model)
    print("Running on:", device)
    train_engine.run(dataloaders['train'], max_epochs=n_epochs)

    return model, metrics


if __name__ == "__main__":
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    combination_module = PairCombinationModule(feature_combination_list, KinshipClassifier.FACENET_OUT_SIZE, 0.7)
    _, _ = finetune_model(KinshipClassifier, PROJECT_ROOT, 128, num_workers=8, device=device, lr=1e-4, lr_decay=1e-3,
                          n_epochs=5, weight_decay=1e-4, simple_fc_layers=[512], custom_fc_layers=[2048, 512],
                          final_fc_layers=[512], combination_module=combination_module,
                          combination_size=combination_module.output_size(),
                          train_ds_name='dev_dataset.pkl', dev_ds_name='dev_dataset.pkl',
                          pin_memory=True, non_blocking=True,
                          logging_rate=10, loss_func=None, saving_rate=10, experiment_name='ex1')


