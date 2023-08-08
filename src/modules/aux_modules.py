from copy import deepcopy

import torch

from src.utils import prepare


class TunnelandProbing(torch.nn.Module):
    def __init__(self, loaders, model, num_classes, optim_type, optim_params,
                 reprs_hook, epochs_probing):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.loaders = loaders
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.num_classes = num_classes
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.hooks_reprs = reprs_hook
        self.epochs_probing = epochs_probing
        self.logger = None
        
        
    def forward(self, step, checkpoint_step):
        self.model.eval()
        DATASET_NAME = 'test_proper'
        x_test = torch.cat([x for x, _ in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        y_test = torch.cat([y for _, y in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        _ = self.model(x_test)
        reprs_test = self.hooks_reprs.callback.activations
        self.hooks_reprs.reset()
        evaluators = {}
        
        
        input_dims = [rep[1].size(-1) for rep in reprs_test]
        output_dims = len(input_dims) * [self.num_classes]
        self.heads = self.prepare_heads(input_dims, output_dims).to(self.device)
        self.optim = prepare.prepare_optim_and_scheduler(self.heads, self.optim_type, deepcopy(self.optim_params))[0]
        
        for name, rep in reprs_test: # you can use prepare from hooks_reprs module
            rep_rank = rep.data.detach()
            name_dict = f'ranks_{DATASET_NAME}/{name}'
            name_dict_ratio = f'ranks_{DATASET_NAME}_ratio/{name}'
            name_dict_null = f'ranks_{DATASET_NAME}_null/{name}'
            if rep_rank.size(1) > 8000:
                rep_rank = rep_rank[:, self.hooks_reprs.subsampling[name]]
            cov_matrix = torch.cov(rep_rank.T)
            rank = torch.linalg.matrix_rank(cov_matrix).item()
            evaluators[name_dict] = rank
            evaluators[name_dict_ratio] = rank / cov_matrix.size(0)
            evaluators[name_dict_null] = cov_matrix.size(0) - rank
        
        accs_epoch = []
        losses_epoch = []
        self.model.train()
        for _ in range(self.epochs_probing): # ustalić warunek stopu zbieżności głowic
            self.heads.train()
            for i, (x_train, y_train) in enumerate(self.loaders['train']):
                if i > 10 and step % checkpoint_step !=0:
                    break
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                _ = self.model(x_train)
                reprs = self.hooks_reprs.callback.activations
                self.hooks_reprs.reset()
                losses_train = []
                for i, head in enumerate(self.heads):
                    rep = reprs[i][1]
                    output = head(rep)
                    loss = self.criterion(output, y_train)
                    # acc = (torch.argmax(output, dim=1) == y_train).float().mean()
                    losses_train.append(loss)
            
                loss = sum(losses_train) / len(losses_train)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            accs_test = []
            losses_head = []
            self.heads.eval()
            for i, head in enumerate(self.heads):
                rep = reprs_test[i][1]
                output = head(rep)
                loss = self.criterion(output, y_test)
                acc = (torch.argmax(output, dim=1) == y_test).float().mean()
                losses_head.append(loss)
                accs_test.append(acc)
                
            accs_epoch.append(accs_test)
            losses_epoch.append(losses_head)
            
        accs_epoch = torch.tensor(accs_epoch).max(dim=0)[0].reshape(-1)
        losses_epoch = torch.tensor(losses_epoch).max(dim=0)[0].reshape(-1)
        self.prepare_evaluators(evaluators, accs_epoch, losses_epoch) # interesują mnie jedynie wyniki ostatniej epoki
        
        evaluators['steps/tunnel'] = step
        self.logger.log_scalars(evaluators, step)
            
    def prepare_heads(self, input_dims, output_dims):
        heads = []
        for i in range(len(input_dims)):
            heads.append(torch.nn.Linear(input_dims[i], output_dims[i]))
        heads = torch.nn.ModuleList(heads)
        return heads
    
    def prepare_evaluators(self, evaluators, accs, losses):
        for i in range(len(accs)):
            evaluators[f'linear_probing/accuracy_{i}'] = accs[i].item()
            evaluators[f'linear_probing/loss_{i}'] = losses[i].item()

from torch.distributions import Categorical
from collections import defaultdict

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
class TraceFIM(torch.nn.Module):
    def __init__(self, x_held_out, model, criterion, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.labels = torch.arange(num_classes).to(self.device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.logger = None

    def forward(self, step):
        y_pred = self.model(self.x_held_out)
        y_sampled = Categorical(logits=y_pred).sample()
        loss = self.criterion(y_pred, y_sampled)
        params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        grads = torch.autograd.grad(
            loss,
            params,)
            # retain_graph=True,
            # create_graph=True)
        evaluators = defaultdict(float)
        overall_trace = 0.0
        # najlepiej rozdzielić po module
        for param_name, gr in zip(params_names, grads):
            if gr is not None:
                trace_p = (gr**2).sum()
                evaluators[f'trace_fim/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p
        
        evaluators['trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.logger.log_scalars(evaluators, step)            
