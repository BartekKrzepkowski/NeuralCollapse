#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import torch
import omegaconf

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from trainer.trainer_classification_phases import TrainerClassification

from src.modules.hooks import Hooks
from src.modules.metrics import RunStats, Stiffness, LinearProbing


def objective(exp, window, epochs):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_ACCUM_STEPS = 1
    NUM_CLASSES = 10
    RANDOM_SEED = 83
    
    type_names = {
        'model': 'simple_cnn',
        'criterion': 'fp',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare model ════════════════════════ #
    
    
    model_config = {'backbone_type': 'resnet18',
                    'only_features': False,
                    'batchnorm_layers': True,
                    'width_scale': 1.0,
                    'skips': True,
                    'modify_resnet': True,
                    'wheter_concate': False}
    model_params = {'model_config': model_config, 'num_classes': NUM_CLASSES, 'dataset_name': type_names['dataset']}
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    
    FP = 0.0
    criterion_params = {'model': model, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': False, 'fpw': FP}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params).to(device)
    
    criterion_params['model'] = None
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = 2e-1
    MOMENTUM = 0.0
    WD = 0.0
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * (window + epochs)
    
    
    print((T_max // (window + epochs)) // 2, len(loaders['train']))
    optim_params = {'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WD}
    scheduler_params = None
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_fp_{FP}_lr_{LR}_wd_{WD}'
    EXP_NAME = f'{GROUP_NAME}_window_{window} , original_clp'
    PROJECT_NAME = 'Critical_Periods_tunnel'
    ENTITY_NAME = 'ideas_cv'

    h_params_overall = {
        'model': model_params,
        'criterion': criterion_params,
        'dataset': dataset_params,
        'loaders': loader_params,
        'optim': optim_params,
        'scheduler': scheduler_params,
        'type_names': type_names
    }   
 
 
    # ════════════════════════ prepare held out data ════════════════════════ #
    
    
    # DODAJ - POPRAWNE DANE
    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    # x_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x.pt').to(device)
    # y_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_y.pt').to(device)
    
    # x_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x.pt').to(device)
    # y_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_y.pt').to(device) 
    
    held_out_data_plus_extra = {}
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    
    extra_modules['hooks_dead_relu'] = Hooks(model, logger=None, callback_type='dead_relu')
    extra_modules['hooks_dead_relu'].register_hooks([torch.nn.ReLU])
    extra_modules['hooks_acts'] = Hooks(model, logger=None, callback_type='gather_activations')
    extra_modules['hooks_acts'].register_hooks([torch.nn.Conv2d, torch.nn.Linear])
    
    
    reprs = extra_modules['hooks_acts'].callback.activations
    extra_modules['hooks_acts'].reset()
    
    input_dims = [rep[1].size(-1) for rep in reprs]
    output_dims = len(input_dims) * [NUM_CLASSES]
    optim_params = {'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WD}
    extra_modules['probes'] = LinearProbing(model, criterion, input_dims, output_dims, optim_type=type_names['optim'], optim_params=optim_params)
    
    
    extra_modules['run_stats'] = RunStats(model, optim)
    
    held_out_data_plus_extra['num_classes'] = NUM_CLASSES
    # extra_modules['stiffness'] = Stiffness(model, **held_out_data_plus_extra)
    
    
    # ════════════════════════ prepare trainer ════════════════════════ #
    
    
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
        #'held_out_data: '{'x_true1': x_data_proper, 'y_true1': y_data_proper, 'x_true2': x_data_blurred, 'y_true2': y_data_blurred, 'num_classes': NUM_CLASSES},
    }
    
    trainer = TrainerClassification(**params_trainer)


    # ════════════════════════ prepare run ════════════════════════ #


    CLIP_VALUE = 100.0
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    logger_config = {'logger_name': 'tensorboard',
                     'project_name': PROJECT_NAME,
                     'entity': ENTITY_NAME,
                     'hyperparameters': h_params_overall,
                     'whether_use_wandb': True,
                     'layout': ee_tensorboard_layout(params_names), # is it necessary?
                     'mode': 'online',
    }
    extra = {'window': window}
    
    config = omegaconf.OmegaConf.create()
    
    config.epoch_start_at = 0
    config.epoch_end_at = epochs
    
    config.grad_accum_steps = GRAD_ACCUM_STEPS
    config.log_multi = 1#(T_max // epochs) // 10
    config.save_multi = 0#T_max // 10
    config.stiff_multi = (T_max // (window + epochs)) // 2
    config.acts_rank_multi = (T_max // (window + epochs)) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = 'reports'
    config.exp_name = EXP_NAME
    config.extra = extra
    config.logger_config = logger_config
    
    
    # ════════════════════════ run ════════════════════════ #
    
    
    if exp == 'deficit':
        trainer.run_exp1(config)
    elif exp == 'sensitivity':
        trainer.run_exp2(config)
    elif exp == 'deficit_reverse':
        trainer.run_exp1_reverse(config)
    else:
        raise ValueError('exp should be either "deficit" or "sensitivity"')


if __name__ == "__main__":
    EPOCHS = 200
    for window in np.linspace(0, 200, 6):
        objective('deficit', int(window), EPOCHS)
