import os
import pickle
from implementation.Training import PROJECT_ROOT

from ax import *
import torch

from implementation.Training import finetune_model
from implementation.Models import KinshipClassifier, PairCombinationModule
from implementation.Utils import *

# Hyperparams to optimize:
#     simple_fc - num_layers + widths
#     custom_fc - num_layers + widths
#     final_fc - num_layers + widths
#     comb_module - dropout_prob
#     lr
#     weight_decay


def find_best_hypers(project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True,
                     device=None, loss_func=None, n_epochs=1, simple_fc_layers=None, custom_fc_layers=None,
                     final_fc_layers=None, train_ds_name=None, dev_ds_name=None, logging_rate=-1, n_trials=10):
    comb_dropout = dict(name='dropout_prob', bounds=[0.5, 0.95], type='range')
    initial_lr = dict(name='initial_lr', bounds=[1e-9, 1e-3], type='range', log_scale=True)
    lr_decay_mult = dict(name='lr_decay', bounds=[0.1, 0.3], type='range')
    weight_decay = dict(name='weight_decay', bounds=[1e-3, 1e-1], type='range')

    def _objective(parameters):
        dropout_prob = parameters['dropout_prob']
        initial_lr = parameters['initial_lr']
        weight_decay = parameters['weight_decay']
        lr_decay = parameters['lr_decay']

        combinator = PairCombinationModule(feature_combination_list, KinshipClassifier.FACENET_OUT_SIZE, dropout_prob)

        model, metrics = finetune_model(KinshipClassifier, project_path, batch_size,
                                        num_workers=num_workers, device=device, base_lr=initial_lr, lr_decay=lr_decay,
                                        n_epochs=n_epochs, weight_decay=weight_decay, simple_fc_layers=simple_fc_layers,
                                        custom_fc_layers=custom_fc_layers, final_fc_layers=final_fc_layers,
                                        combination_module=combinator, combination_size=combinator.output_size(),
                                        train_ds_name=train_ds_name, dev_ds_name=dev_ds_name,
                                        pin_memory=pin_memory, non_blocking=non_blocking,
                                        logging_rate=logging_rate, loss_func=loss_func)

        return metrics['final_dev_loss']

    optimization_results = optimize(parameters=[comb_dropout, initial_lr, lr_decay_mult, weight_decay],
                                    evaluation_function=_objective,
                                    experiment_name='optimizing',
                                    minimize=True,
                                    total_trials=n_trials)

    return optimization_results


if __name__ == "__main__":

    batch_size = 128
    num_workers = 8
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    train_ds = 'dev_dataset.pkl'
    dev_ds = 'dev_dataset.pkl'

    best_params, values, experiment, model = find_best_hypers(PROJECT_ROOT, batch_size, num_workers, device=device,
                                                              n_epochs=4, simple_fc_layers=[512],
                                                              custom_fc_layers=[2048, 512], final_fc_layers=[],
                                                              train_ds_name=train_ds, dev_ds_name=dev_ds,
                                                              logging_rate=30)
    results = {'best_params': best_params, 'values': values, 'experiment': experiment, 'model': model}
    with open(os.path.join(PROJECT_ROOT, 'models', 'optimize_test.pkl'), 'wb+') as f:
        pickle.dump(results, f)
