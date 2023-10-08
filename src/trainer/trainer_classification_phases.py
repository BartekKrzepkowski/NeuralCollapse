from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm, trange

from src.data.loaders import Loaders
from src.data.transforms import TRANSFORMS_NAME_MAP
from src.utils.common import LOGGERS_NAME_MAP

from src.modules.aux import TimerCPU
from src.utils.utils_trainer import adjust_evaluators, adjust_evaluators_pre_log, create_paths, save_model
from src.utils.utils_optim import clip_grad_norm


def froze_model(model, is_true):
    for para in model.parameters():
        para.requires_grad = is_true


class TrainerClassification:
    def __init__(self, model, criterion, loaders, optim, lr_scheduler, extra_modules, device):
        self.model = model#torch.compile(model, mode="reduce-overhead")
        self.criterion = criterion
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = None
        self.global_step = None

        self.extra_modules = extra_modules
        self.timer = TimerCPU()
        self.device = device
        
             
    def run_exp(self, config):        
        self.manual_seed(config)
        self.at_exp_start(config)
        
        self.run_loop(config.epoch_start_at, config.epoch_end_at, config)
        
        step = f'epoch_{self.epoch + 1}'
        save_model(self.model, self.save_path(step))
        self.logger.close()
        
    def run_exp_blurred(self, config):        
        self.manual_seed(config)
        self.at_exp_start(config)
        
        self.loaders['train'].dataset.transform = TRANSFORMS_NAME_MAP['transform_blurred_train']
        
        self.run_loop(config.epoch_start_at, config.epoch_end_at, config)
        
        step = f'epoch_{self.epoch + 1}'
        save_model(self.model, self.save_path(step))
        self.logger.close()
        
    def run_exp_half(self, config):        
        self.manual_seed(config)
        self.at_exp_start(config)
        
        batch_size = config.logger_config['hyperparameters']['loaders']['batch_size']
        num_workers = config.logger_config['hyperparameters']['loaders']['num_workers']
        dataset_name = config.logger_config['hyperparameters']['type_names']['dataset']
        self.train_loader = Loaders(dataset_name=dataset_name)
        self.loaders['train'] = self.train_loader.get_half_loader(batch_size, is_train=True, num_workers=num_workers)
            
        self.run_loop(config.epoch_start_at, config.epoch_end_at, config)
        
        step = f'epoch_{self.epoch + 1}'
        save_model(self.model, self.save_path(step))
        self.logger.close()


    def run_loop(self, epoch_start_at, epoch_end_at, config):
        """
        Main method of trainer.
        Set seed, run train-val in the loop.
        Args:
            config (dict): Consists of:
                epoch_start (int): A number representing the beginning of run
                epoch_end (int): A number representing the end of run
                step_multi (int):
                base_path (str): Base path
                exp_name (str): Base name of experiment
                logger_name (str): Logger type
                random_seed (int): Seed generator
        """
        for epoch in trange(epoch_start_at, epoch_end_at, desc='run_exp',
                            leave=True, position=0, colour='green', disable=config.whether_disable_tqdm):
            self.epoch = epoch
            self.model.train()
            self.timer.start('train_epoch')
            self.run_epoch(phase='train', config=config)
            self.timer.stop('train_epoch')
            self.model.eval()
            with torch.no_grad():
                self.run_epoch(phase='test_proper', config=config)
                self.run_epoch(phase='test_blurred', config=config)
                
            self.timer.log(epoch)


    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
        config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_config["logger_name"]}'
        self.logger = LOGGERS_NAME_MAP[config.logger_config['logger_name']](config)
        
        self.timer.set_logger(self.logger)
        
        if 'stiffness' in self.extra_modules:
            self.extra_modules['stiffness'].logger = self.logger
        if 'hooks_dead_relu' in self.extra_modules:
            self.extra_modules['hooks_dead_relu'].logger = self.logger
        if 'hooks_acts' in self.extra_modules:
            self.extra_modules['hooks_acts'].logger = self.logger
        if 'tunnel' in self.extra_modules:
            self.extra_modules['tunnel'].logger = self.logger
        if 'tunnel_grads' in self.extra_modules:
            self.extra_modules['tunnel_grads'].logger = self.logger
        if 'trace_fim' in self.extra_modules:
            self.extra_modules['trace_fim'].logger = self.logger


    def run_epoch(self, phase, config):
        """
        Run single epoch
        Args:
            phase (str): phase of the trening
            config (dict):
        """
        running_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        epoch_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
        }
        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}',
                            leave=False, position=1, total=loader_size, colour='red', disable=config.whether_disable_tqdm)
        self.global_step = self.epoch * loader_size
        for i, data in enumerate(progress_bar):          
            x_true, y_true = data
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            
            self.timer.start('forward')
            y_pred = self.model(x_true)
            self.timer.stop('forward')
            
            self.timer.start('criterion')
            loss, evaluators = self.criterion(y_pred, y_true)
            self.timer.stop('criterion')
            
            step_assets = {
                'evaluators': evaluators,
                'denom': y_true.size(0),
            }
            
            if 'train' == phase:
                loss.backward(retain_graph=True)
                if config.clip_value > 0:
                    norm = clip_grad_norm(torch.nn.utils.clip_grad_norm_, self.model, config.clip_value)
                    step_assets['evaluators']['run_stats/model_gradient_norm_squared_from_pytorch'] = norm.item() ** 2
                
                self.optim.step()
                
                if self.extra_modules['run_stats'] is not None:
                    self.timer.start('run_stats')
                    step_assets['evaluators'] = self.extra_modules['run_stats'](step_assets['evaluators'], 'l2')
                    self.timer.stop('run_stats')
                    
                if self.lr_scheduler is not None and (((self.global_step + 1) % loader_size == 0) or config.logger_config['hyperparameters']['type_names']['scheduler'] != 'multiplicative'):
                    self.lr_scheduler.step()
                    step_assets['evaluators']['lr/training'] = self.optim.param_groups[0]['lr']
                    step_assets['evaluators']['steps/lr'] = self.global_step
                    
                self.optim.zero_grad(set_to_none=True)
                
                if self.extra_modules['trace_fim'] is not None and config.fim_trace_multi and self.global_step % config.fim_trace_multi == 0:
                    self.timer.start('trace_fim')
                    self.extra_modules['trace_fim'](self.global_step)
                    self.timer.stop('trace_fim')
                
                tunnel_multi = config.tunnel_multi if self.epoch > 5 else ((config.tunnel_multi // 20) if self.epoch > 0 else (config.tunnel_multi // 80))# make it more well thought
                if config.tunnel_multi and self.global_step % tunnel_multi == 0 and self.extra_modules['tunnel'] is not None:
                    froze_model(self.model, False)
                    self.extra_modules['hooks_reprs'].enable()
                    self.timer.start('tunnel')
                    self.extra_modules['tunnel'](self.global_step, scope='periodic', phase='train')
                    self.timer.stop('tunnel')
                    self.extra_modules['hooks_reprs'].disable()
                    froze_model(self.model, True)
                    
                tunnel_grads_multi = config.tunnel_grads_multi if self.epoch > 5 else ((config.tunnel_grads_multi // 20) if self.epoch > 0 else (config.tunnel_grads_multi // 80))# make it more well thought
                if config.tunnel_grads_multi and self.global_step % tunnel_grads_multi == 0 and self.extra_modules['tunnel_grads'] is not None:
                    self.timer.start('tunnel_grads')
                    self.extra_modules['tunnel_grads'](self.global_step, scope='periodic', phase='train')
                    self.timer.stop('tunnel_grads')
                    
                
                # if self.extra_modules['hooks_acts'] is not None and self.extra_modules['probes'] is not None:
                #     froze_model(self.model, False)
                #     reprs = self.extra_modules['hooks_acts'].callback.activations
                #     self.timer.start('probes')
                #     evaluators = self.extra_modules['probes'](reprs, y_true, evaluators)
                #     self.timer.stop('probes')
                #     froze_model(self.model, True)
                
                # stiffness
                # if config.stiff_multi and self.global_step % (config.grad_accum_steps * config.stiff_multi) == 0 and self.extra_modules['stiffness'] is not None:
                #     self.extra_modules['hooks_dead_relu'].disable()
                #     self.extra_modules['stiffness'].log_stiffness(self.global_step)
                #     self.extra_modules['hooks_dead_relu'].enable()
                
                # actively (batch-wise) gather reprs and its ranks
                # if config.acts_rank_multi:
                #     if self.global_step % (config.grad_accum_steps * config.acts_rank_multi) == 0 and self.extra_modules['hooks_acts'] is not None :
                #         self.timer.start('hooks_acts')
                #         self.extra_modules['hooks_acts'].write_to_tensorboard(self.global_step)
                #         self.timer.stop('hooks_acts')
                #     self.extra_modules['hooks_acts'].reset()
                # gather dead relu ratio per layer
                # if self.extra_modules['hooks_dead_relu'] is not None:
                #     self.timer.start('hooks_dead_relu')
                #     self.extra_modules['hooks_dead_relu'].write_to_tensorboard(self.global_step)
                #     self.timer.stop('hooks_dead_relu')
            
            
            # ════════════════════════ logging ════════════════════════ #
            
            self.timer.log(self.global_step)
            
            
            running_assets = self.update_assets(running_assets, step_assets, step_assets['denom'], 'running', phase)

            whether_save_model = config.save_multi and self.global_step % config.save_multi == 0
            whether_log = (i + 1) % config.log_multi == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_save_model and 'train' in phase:
                step = f'epoch_{self.epoch}_global_step_{self.global_step}'
                save_model(self.model, self.save_path(step))

            if whether_log or whether_epoch_end:
                epoch_assets = self.update_assets(epoch_assets, running_assets, 1.0, 'epoch', phase)

            if whether_log:
                self.log(running_assets, phase, 'running', progress_bar, self.global_step)
                running_assets['evaluators'] = defaultdict(float)
                running_assets['denom'] = 0.0

            if whether_epoch_end:
                self.log(epoch_assets, phase, 'epoch', progress_bar, self.epoch)

            self.global_step += 1


    def log(self, assets: Dict, phase: str, scope: str, progress_bar: tqdm, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        evaluators_log = adjust_evaluators_pre_log(assets['evaluators'], assets['denom'], round_at=4)
        evaluators_log[f'steps/{phase}_{scope}'] = step
        self.logger.log_scalars(evaluators_log, step)
        progress_bar.set_postfix(evaluators_log)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)


    def update_assets(self, assets_target: Dict, assets_source: Dict, multiplier, scope, phase: str):
        '''
        Update epoch assets
        Args:
            assets_target (Dict): Assets to which assets should be transferred
            assets_source (Dict): Assets from which assets should be transferred
            multiplier (int): Number to get rid of the average
            scope (str): Either running or epoch
            phase (str): Phase of the traning
        '''
        assets_target['evaluators'] = adjust_evaluators(assets_target['evaluators'], assets_source['evaluators'],
                                                        multiplier, scope, phase)
        assets_target['denom'] += assets_source['denom']
        return assets_target


    def manual_seed(self, config: defaultdict):
        """
        Set the environment for reproducibility purposes.
        Args:
            config (defaultdict): set of parameters
                usage of:
                    random seed (int):
                    device (torch.device):
        """
        import random
        import numpy as np
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(config.random_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
