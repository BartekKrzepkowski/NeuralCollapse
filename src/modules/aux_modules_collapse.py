import numpy as np
import torch

def variance_cos(x_data, y_true, evaluators): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    denom = y_true.size(0)
    classes = np.unique(y_true.numpy())
    denom_class = classes.size(0)
    for name, internal_repr in x_data:
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr, dim=0, keepdim=True)
        general_mean /= (torch.norm(general_mean, dim=1, keepdim=True) + eps)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]
            class_mean = torch.mean(class_internal_repr, dim=0, keepdim=True)
            class_mean /= (torch.norm(class_mean, dim=1, keepdim=True) + eps)
            for sample in class_internal_repr:
                within_cos = class_mean @ sample / (torch.norm(sample, dim=1, keepdim=True) + eps)
                abs_within_angle = torch.abs(torch.acos(within_cos))
                within_class_cov += abs_within_angle  # oblicz variance zamiast sumy
            between_cos = general_mean @ general_mean
            abs_between_angle = torch.abs(torch.acos(between_cos))
            between_class_cov += abs_between_angle  # oblicz variance zamiast sumy
        
        within_class_cov /= denom  # (D, D)
        between_class_cov /= denom_class  # (D, D)
        total_class_cov = within_class_cov + between_class_cov  # (D, D)
        
        evaluators[f'within_cov_normalized_abs_angle/{name}'] = within_class_cov / total_class_cov
        evaluators[f'between_cov_normalized_abs_angle/{name}'] = between_class_cov / total_class_cov
    return evaluators


def variance_eucl(x_data, y_true, evaluators):
    eps = torch.finfo(torch.float32).eps
    denom = y_true.size(0)
    classes = np.unique(y_true.numpy())
    denom_class = classes.size(0)
    for name, internal_repr in x_data:
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr, dim=0, keepdim=True)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]
            class_mean = torch.mean(class_internal_repr, dim=0, keepdim=True)
            for sample in class_internal_repr:
                within_sample = (sample - class_mean)
                within_class_cov += within_sample @ within_sample.T
            between_sample = (class_mean - general_mean)
            between_class_cov += between_sample @ between_sample.T
        
        within_class_cov /= denom  # (D, D)
        between_class_cov /= denom_class  # (D, D)
        total_class_cov = within_class_cov + between_class_cov  # (D, D)
        trace_wcc = calculate_rank_via_svd(within_class_cov)
        trace_bcc = calculate_rank_via_svd(between_class_cov)
        trace_tcc = calculate_rank_via_svd(total_class_cov)
        
        normalized_wcc = trace_wcc / (trace_tcc + eps) 
        normalized_bcc = trace_bcc / (trace_tcc + eps)
        
        evaluators[f'within_cov_normalized_eucl/{name}'] = normalized_wcc
        evaluators[f'between_cov_normalized_eucl/{name}'] = normalized_bcc
    return evaluators
        
        
        
def calculate_rank_via_svd(cov_matrix):
    singulars = torch.linalg.svdvals(cov_matrix)
    trace = singulars.sum()
    return trace



import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
    
class TunnelGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.eps = torch.finfo(torch.float32).eps
        self.logger = None
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss, predictions

    def forward(self, x_true, y_true, scope, phase, step):
        self.model.eval()
        classes = np.unique(y_true.numpy())
        num_classes = len(classes)
        prefix = lambda a, b : f'ranks_grads_{a}_{b}'
        postfix = f'____{scope}____{phase}'
        evaluators = {}

        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names}
        buffers = {}
            
        per_sample_grads, y_pred = self.ft_criterion(params, buffers, x_true, y_true)
        y_pred_label = torch.argmax(y_pred.data.squeeze(), dim=1)
        per_sample_grads = {k1: v.detach().data for k1, v in per_sample_grads.items()}
        concatenated_grads = torch.empty((x_true.shape[0], 0), device=x_true.device)
        
        for c in classes:
            idxs_mask = y_true == c
            evaluators[f'misclassification_per_class/{c}'] = (y_pred_label[idxs_mask] != y_true[idxs_mask]).float().mean().item()
            
        
        self.prepare_variables(per_sample_grads, concatenated_grads)
            
        evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'feature'), postfix, normalize=False, batch_first=False, risky_names=('concatenated_grads'))
        evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'batch'), postfix, normalize=False, batch_first=True, risky_names=('concatenated_grads'))
        evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'feature'), postfix, normalize=True, batch_first=False, risky_names=('concatenated_grads'))
        evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'batch'), postfix, normalize=True, batch_first=True, risky_names=('concatenated_grads'))
        
        self.model.train()
        evaluators['steps/tunnel_grads'] = step
        self.logger.log_scalars(evaluators, step)
    
    def prepare_variables(self, per_sample_grads, concatenated_grads): # sampluj gdy za duże
        for weight_name in per_sample_grads:
            per_sample_grads[weight_name] = per_sample_grads[weight_name].reshape(per_sample_grads[weight_name].shape[0], -1)
            concatenated_grads = torch.cat((concatenated_grads, per_sample_grads[weight_name]), dim=1)
        per_sample_grads['concatenated_grads'] = concatenated_grads
    
    
    def prepare_and_calculate_ranks(self, matrices, evaluators, prefix, postfix, normalize=False, batch_first=True, risky_names=()):
        for name, matrix in matrices:
            if name in risky_names and not batch_first:
                continue
            if normalize:
                matrix = matrix / (torch.norm(matrix, dim=1, keepdim=True) + self.eps)
            ranks = self.calculate_rank_via_svd(matrix, batch_first)
            denom = matrix.size(0) if batch_first else matrix.T.size(0)
            
            name_dict = f'{prefix}/{name}{postfix}'
            name_dict_ratio = f'{prefix}_ratio/{name}{postfix}'
            # name_dict_null = f'{prefix}_null/{name}{postfix}'
            evaluators[name_dict] = ranks[0]
            evaluators[name_dict_ratio] = ranks[0] / denom # check if dim makes sense
            # evaluators[name_dict_null] = denom - ranks[0]
            
            name_dict_square_stable = f'{prefix}_square_stable/{name}{postfix}'
            name_dict_ratio_square_stable = f'{prefix}_ratio_square_stable/{name}{postfix}'
            name_dict_null_square_stable = f'{prefix}_null_square_stable/{name}{postfix}'
            evaluators[name_dict_square_stable] = ranks[1]
            evaluators[name_dict_ratio_square_stable] = ranks[1] / denom # check if dim makes sense
            # evaluators[name_dict_null_square_stable] = denom - ranks[1]
        return evaluators
    
    def calculate_rank_via_svd(self, matrix, batch_first):
        A = matrix if batch_first else matrix.T
        gramian_matrix = A @ A.T
        singulars = torch.linalg.svdvals(gramian_matrix)
        rank = (singulars > (singulars[0] * max(gramian_matrix.size()) * self.eps)).sum()
        square_stable_rank = singulars.sum() / (singulars[0] + self.eps)
        return rank, square_stable_rank
    
        