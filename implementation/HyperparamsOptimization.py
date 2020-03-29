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


def find_best_hypers(project_path, batch_size, num_workers=0, pin_memory=True, non_blocking=True, augment=True,
                     device=None, loss_func=None, n_epochs=1, train_ds_name=None, dev_ds_name=None, logging_rate=-1,
                     n_trials=10, patience=-1):
    base_lr = dict(name='base_lr', bounds=[1e-6, 1e-3], type='range', log_scale=True)
    max_lr = dict(name='max_lr', bounds=[1e-5, 1e-2], type='range', log_scale=True)
    lr_decay_iters = dict(name='lr_decay_iters', bounds=[0.3, 1.0], type='range')
    weight_decay = dict(name='weight_decay', bounds=[1e-3, 1e-1], type='range', log_scale=True)
    lr_gamma = dict(name='lr_gamma', bounds=[0.4, 0.99], type='range', log_scale=True)
    simple_1 = dict(name='simple_1', bounds=[256, 2560], type='range', log_scale=True)
    simple_2 = dict(name='simple_2', bounds=[256, 2560], type='range', log_scale=True)
    custom_1 = dict(name='custom_1', bounds=[1024, 4096], type='range', log_scale=True)
    custom_2 = dict(name='custom_2', bounds=[512, 2560], type='range', log_scale=True)
    final_1 = dict(name='final_1', bounds=[256, 1024], type='range', log_scale=True)
    final_2 = dict(name='final_2', bounds=[256, 1024], type='range', log_scale=True)

    def _objective(parameters):
        simple_fc_layers = [parameters['simple_1'], parameters['simple_2']]
        custom_fc_layers = [parameters['custom_1'], parameters['custom_2']]
        final_fc_layers = [parameters['final_1'], parameters['final_2']]
        base_lr = parameters['base_lr']
        max_lr = parameters['max_lr']
        lr_decay_iters = parameters['lr_decay_iters']
        weight_decay = parameters['weight_decay']
        lr_gamma = parameters['lr_gamma']

        combinator = PairCombinationModule(feature_combination_list, KinshipClassifier.FACENET_OUT_SIZE, 0.7)
        print("Training parameters: ")
        print(parameters)
        model, metrics = finetune_model(KinshipClassifier, project_path, batch_size,
                                        num_workers=num_workers, device=device, base_lr=base_lr, max_lr=max_lr,
                                        lr_gamma=lr_gamma, lr_decay_iters=lr_decay_iters, n_epochs=n_epochs,
                                        weight_decay=weight_decay, simple_fc_layers=simple_fc_layers,
                                        custom_fc_layers=custom_fc_layers, final_fc_layers=final_fc_layers,
                                        combination_module=combinator, combination_size=combinator.output_size(),
                                        train_ds_name=train_ds_name, dev_ds_name=dev_ds_name,
                                        pin_memory=pin_memory, non_blocking=non_blocking, data_augmentation=augment,
                                        logging_rate=logging_rate, loss_func=loss_func, patience=patience,
                                        verbose=False)

        return metrics['final_dev_score']

    optimization_results = optimize(parameters=[simple_1, simple_2, custom_1, custom_2, final_1, final_2, base_lr,
                                                max_lr, lr_decay_iters, lr_gamma, weight_decay],
                                    evaluation_function=_objective,
                                    experiment_name='optimizing',
                                    minimize=True,
                                    total_trials=n_trials)

    return optimization_results


if __name__ == "__main__":

    batch_size = 82
    num_workers = 8
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    train_ds = 'dev'
    dev_ds = 'mini'

    best_params, values, experiment, model = find_best_hypers(PROJECT_ROOT, batch_size, num_workers, device=device,
                                                              n_epochs=4, train_ds_name=train_ds, dev_ds_name=dev_ds,
                                                              augment=True)
    results = {'best_params': best_params, 'values': values, 'experiment': experiment, 'model': model}
    with open(os.path.join(PROJECT_ROOT, 'models', 'optimize_test.pkl'), 'wb+') as f:
        pickle.dump(results, f)
