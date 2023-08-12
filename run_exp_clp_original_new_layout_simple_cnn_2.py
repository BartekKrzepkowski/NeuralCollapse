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
from src.trainer.trainer_classification_original_clp import TrainerClassification

from src.modules.hooks import Hooks
from src.modules.metrics import RunStats, Stiffness, LinearProbing
from src.modules.aux_modules import TunnelandProbing, TraceFIM


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
    
    
    N = 2
    NUM_FEATURES = 3
    DIMS = [NUM_FEATURES, 32] + [64] * N + [128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3, 3] * (N + 1), 'strides': [1, 1] * (N + 1), 'paddings': [1, 1] * (N + 1), 'whether_pooling': [False, True] * (N + 1)}
    model_params = {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS}
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)
    
    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    
    FP = 0.0#1e-2
    criterion_params = {'model': model, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': False, 'fpw': FP}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params).to(device)
    
    criterion_params['model'] = None
    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = 1e-1
    MOMENTUM = 0.0
    WD = 0.0
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * (window + epochs)
    
    
    print((T_max // (window + epochs)) // 2, len(loaders['train']))
    # print(T_max//window, T_max-3*T_max//window, 3*T_max//window)
    # h_params_overall['scheduler'] = {'eta_max':LR, 'eta_medium':1e-2, 'eta_min':1e-6, 'warmup_iters2': 3*T_max//window, 'inter_warmups_iters': T_max-3*T_max//window, 'warmup_iters1': 3*T_max//window, 'milestones':[], 'gamma':1e-1}
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
    x_held_out = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x.pt').to(device)
    y_held_out = torch.load(f'data/{type_names["dataset"]}_held_out_proper_y.pt').to(device)
    
    # x_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x.pt').to(device)
    # y_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_y.pt').to(device) 
    
    held_out_data_plus_extra = {}
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    
    # extra_modules['hooks_dead_relu'] = Hooks(model, logger=None, callback_type='dead_relu')
    # extra_modules['hooks_dead_relu'].register_hooks([torch.nn.ReLU])
    extra_modules['hooks_reprs'] = Hooks(model, logger=None, callback_type='gather_reprs')
    extra_modules['hooks_reprs'].register_hooks([torch.nn.Conv2d, torch.nn.Linear])
    
    # extra_modules['hooks_reprs'].enable()
    # _ = model(torch.randn(2, 3072).to(device))
    # reprs = extra_modules['hooks_reprs'].callback.activations
    # extra_modules['hooks_reprs'].reset()
    # extra_modules['hooks_reprs'].disable()
    
    # input_dims = [rep[1].size(-1) for rep in reprs]
    # output_dims = len(input_dims) * [NUM_CLASSES]
    optim_params = {'lr': LR, 'momentum': MOMENTUM, 'weight_decay': WD}
    # extra_modules['probes'] = LinearProbing(model, criterion, input_dims, output_dims, optim_type=type_names['optim'], optim_params=optim_params)
    
    
    # extra_modules['run_stats'] = RunStats(model, optim)
    
    held_out_data_plus_extra['num_classes'] = NUM_CLASSES
    # extra_modules['stiffness'] = Stiffness(model, **held_out_data_plus_extra)
    
    
    extra_modules['tunnel'] = TunnelandProbing(loaders, model, num_classes=NUM_CLASSES, optim_type=type_names['optim'], optim_params=optim_params,
                                               reprs_hook=extra_modules['hooks_reprs'], epochs_probing=20)
    
    extra_modules['trace_fim'] = TraceFIM(x_held_out, model, criterion, num_classes=NUM_CLASSES)
    
    
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
    # config.stiff_multi = (T_max // (window + epochs)) // 2
    # config.acts_rank_multi = (T_max // (window + epochs)) // 2
    config.tunnel_multi = (T_max // (window + epochs)) * 10
    config.fim_trace_multi = (T_max // (window + epochs)) // 2
    
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
    EPOCHS = 250
    # for window in np.linspace(0, 200, 6):
    window = 0
    objective('deficit', int(window), EPOCHS)
