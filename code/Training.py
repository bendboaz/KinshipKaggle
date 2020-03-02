import os

import torch
from torch import optim
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from code.Models import KinshipClassifier
from code.DataHandling import KinshipDataset

PROJECT_ROOT = "C:\\Users\\bendb\\PycharmProjects\\KinshipKaggle"


if __name__ == "__main__":
    data_path = os.path.join(PROJECT_ROOT, 'data')
    processed_path = os.path.join(data_path, 'processed')

    partitions = ['train', 'dev']
    dataset_paths = {partition: os.path.join(data_path, f"{partition}_dataset.pkl") for partition in partitions}
    raw_paths = {partition: os.path.join(processed_path, partition) for partition in dataset_paths}

    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')

    model = KinshipClassifier(lambda x1, x2: torch.cat([x1, x2], dim=1), KinshipClassifier.FACENET_OUT_SIZE * 2, [1024],
                              [512], [])

    datasets = {partition: KinshipDataset.get_dataset(dataset_paths[partition], raw_paths[partition],
                                                      relationships_path) for partition in raw_paths}

    dataloaders = {partition: DataLoader(datasets[partition], batch_size=64, shuffle=(partition == 'train'),
                                         num_workers=8, pin_memory=True) for partition in datasets}

    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = optim.AdamW(params_to_train, lr=1e-4)
    # lr_scheduler goes here (wraps the optimizer)
    loss_func = torch.nn.CrossEntropyLoss()

    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

    train_engine = create_supervised_trainer(model, optimizer, loss_fn=loss_func, device=device, non_blocking=True)
    eval_engine = create_supervised_evaluator(model, metrics=dict(accuracy=Accuracy(), cross_entropy=Loss(loss_func)),
                                              device=device, non_blocking=True)

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
    train_engine.run(dataloaders['train'], max_epochs=5)
