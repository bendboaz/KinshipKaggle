from ax import *

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
    weight_decay = dict(name='weight_decay', bounds=[1e-3, 1e-1], type='range')

    def _objective(parameters):
        dropout_prob = parameters['dropout_prob']
        initial_lr = parameters['initial_lr']
        weight_decay = parameters['weight_decay']

        combinator = PairCombinationModule(feature_combination_list, KinshipClassifier.FACENET_OUT_SIZE, dropout_prob)

        model, metrics = finetune_model(KinshipClassifier, project_path, batch_size,
                                        num_workers=num_workers, device=device, lr=initial_lr, n_epochs=n_epochs,
                                        weight_decay=weight_decay, simple_fc_layers=simple_fc_layers,
                                        custom_fc_layers=custom_fc_layers, final_fc_layers=final_fc_layers,
                                        combination_module=combinator, combination_size=combinator.output_size(),
                                        train_ds_name=train_ds_name, dev_ds_name=dev_ds_name,
                                        compute_final_validation=True, pin_memory=pin_memory, non_blocking=non_blocking,
                                        logging_rate=logging_rate, loss_func=loss_func)

        return metrics['final_dev_loss']

    optimization_results = optimize(parameters=[comb_dropout, initial_lr, weight_decay],
                                    evaluation_function=_objective,
                                    experiment_name='optimizing',
                                    minimize=True,
                                    total_trials=n_trials)

    return optimization_results
