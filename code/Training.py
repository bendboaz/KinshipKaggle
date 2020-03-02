import os

import torch
from torch import optim
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from code.Models import KinshipClassifier
from code.DataHandling import KinshipDataset
from code.Utils import simple_concatenation

PROJECT_ROOT = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle"


def finetune_model(model_class, project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True, device=None,
                   lr=1e-4, loss_func=None, n_epochs=1, combination_func=simple_concatenation,
                   combination_size=KinshipClassifier.FACENET_OUT_SIZE * 2, simple_fc_layers=None,
                   custom_fc_layers=None, final_fc_layers=None, normalize_features=True, train_ds_name=None,
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

    model = model_class(combination_func, combination_size, simple_fc_layers, custom_fc_layers, final_fc_layers,
                        normalize_features)

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
    optimizer = optim.AdamW(params_to_train, lr=lr)

    train_engine = create_supervised_trainer(model, optimizer, loss_fn=loss_func, device=device,
                                             non_blocking=non_blocking)
    eval_engine = create_supervised_evaluator(model, metrics=dict(accuracy=Accuracy(), cross_entropy=Loss(loss_func)),
                                              device=device, non_blocking=non_blocking)

    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_training_metrics(engine):
        print(f"Finished epoch {engine.state.epoch}")
        eval_engine.run(dataloaders['train'])
        print(f"Epoch {engine.state.epoch}: CE = {eval_engine.state.metrics['cross_entropy']}, "
              f"Acc = {eval_engine.state.metrics['accuracy']}")

    p_bar = ProgressBar()
    p_bar.attach(train_engine)

    print(model)
    print("Running on:", device)
    train_engine.run(dataloaders['train'], max_epochs=n_epochs)

    return model


if __name__ == "__main__":

    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
