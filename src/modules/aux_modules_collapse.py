from collections import defaultdict

import numpy as np
import torch

def variance_cos(x_data, y_true, evaluators): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    denom = y_true.shape[0]
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data:
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr)
        general_mean /= (torch.norm(general_mean) + eps)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]
            class_mean = torch.mean(class_internal_repr)
            class_mean /= (torch.norm(class_mean) + eps)
            for sample in class_internal_repr:
                within_cos = class_mean @ sample / (torch.norm(sample) + eps)
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


def variance_angle_pairwise(x_data, y_true, evaluators): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    denom = y_true.shape[0]
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = 0.0
        within_class_angles = defaultdict(lambda: [])
        between_class_angles = defaultdict(lambda: [])
        within_class_mean = defaultdict(lambda: 0.0)
        for k in range(len(internal_repr)):
            internal_repr[k] /= (torch.norm(internal_repr[k]) + eps)
            
        cos = internal_repr @ internal_repr.T
        abs_angle_matrix = torch.abs(torch.acos(cos))
        
        for i in range(len(classes)):
            idxs_i = np.where(y_true.cpu().numpy() == i)[0]
            general_mean += abs_angle_matrix[idxs_i][:, idxs_i].sum()
        
        for i in range(abs_angle_matrix.shape[0]):
            for j in range(abs_angle_matrix.shape[1]):
                if i < j:
                    # cos = sample1 @ sample2
                    # abs_angle = torch.abs(torch.acos(cos)).item()
                    abs_angle = abs_angle_matrix[i, j].item()
                    general_mean += abs_angle
                    if y_true[i] == y_true[j]:
                        y_i = y_true[i].item()
                        within_class_mean[y_i] += abs_angle
                        within_class_angles[y_i].append(abs_angle)
                    # else:
                    #     between_class_angles[y_i].append(abs_angle)
                    
        general_mean /= (denom * (denom - 1) / 2)
        within_class_mean = {k: v / len(within_class_angles[k]) for k, v in within_class_mean.items()}
        
        for c in within_class_mean:
            within_class_cov_c = 0.0
            for angle in within_class_angles[c]:
                within_class_cov_c += (angle - within_class_mean[c]) ** 2
            within_class_cov += (within_class_cov_c / len(within_class_angles[c]))
            between_class_cov += ((general_mean - within_class_mean[c]) ** 2)
            
        within_class_cov /= len(within_class_mean)
        between_class_cov /= len(within_class_mean)
        total_class_cov = within_class_cov + between_class_cov
        
        evaluators[f'within_cov_normalized_abs_angle/{name}'] = within_class_cov / total_class_cov
        evaluators[f'between_cov_normalized_abs_angle/{name}'] = between_class_cov / total_class_cov
    return evaluators


def variance_eucl(x_data, y_true, evaluators):
    eps = torch.finfo(torch.float32).eps
    denom = y_true.size(0)
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]
            class_mean = torch.mean(class_internal_repr)
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
    def __init__(self, loaders, model, cutoff):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.loaders = loaders
        self.cutoff = cutoff
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.eps = torch.finfo(torch.float32).eps
        self.logger = None
        self.subsampling = defaultdict(lambda: None)
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss, predictions

    def forward(self, step, scope, phase):
        self.model.eval()
        DATASET_NAME = 'test_proper'
        x_test = torch.cat([x for x, _ in self.loaders[DATASET_NAME]], dim=0).to(self.device)#[:5000]
        y_test = torch.cat([y for _, y in self.loaders[DATASET_NAME]], dim=0).to(self.device)#[:5000]
        classes = np.unique(y_test.cpu().numpy())
        prefix = lambda a, b : f'ranks_grads_{a}_{b}'
        postfix = f'____{scope}____{phase}'
        evaluators = {}
        
        
        idxs = []
        for i in range(len(classes)):
            idxs_i = np.where(y_test.cpu().numpy() == i)[0]
            sampled_idxs_i = np.random.choice(idxs_i, size=750, replace=False)
            idxs.append(sampled_idxs_i)
        
        idxs = np.concatenate(idxs)
        x_test = x_test[idxs]
        y_test = y_test[idxs]

        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names and 'bias' not in k and 'downsample' not in k}
        buffers = {}
        
        per_sample_grads, y_pred = None, None
        
        for i in range(x_test.shape[0] // 250):  # accumulate grads
            per_sample_grads_, y_pred_ = self.ft_criterion(params, buffers, x_test[i*250: (i+1)*250], y_test[i*250: (i+1)*250])
            per_sample_grads_ = {k1: v.detach().data for k1, v in per_sample_grads_.items()}
            self.prepare_variables(per_sample_grads_)
            self.sample_feats(per_sample_grads_)
            per_sample_grads, y_pred = self.update(per_sample_grads, y_pred, per_sample_grads_, y_pred_)
        y_pred_label = torch.argmax(y_pred.data.squeeze(), dim=1)
        
        # concatenated_grads = torch.empty((x_test.shape[0], 0), device=x_test.device)
        
        for c in classes:
            idxs_mask = y_test == c
            evaluators[f'misclassification_per_class/{c}'] = (y_pred_label[idxs_mask] != y_test[idxs_mask]).float().mean().item()
            
        
        # self.prepare_variables(per_sample_grads, concatenated_grads)
        # self.sample_feats(per_sample_grads)
            
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'feature'), postfix, normalize=False, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'batch'), postfix, normalize=False, batch_first=True, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'feature'), postfix, normalize=True, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'batch'), postfix, normalize=True, batch_first=True, risky_names=('concatenated_grads'))
        
        variance_angle_pairwise(per_sample_grads, y_test, evaluators)
        
        self.model.train()
        evaluators['steps/tunnel_grads'] = step
        self.logger.log_scalars(evaluators, step)
        
        
    def prepare_variables(self, per_sample_grads, concatenated_grads=None): # sampluj gdy za duże
        for weight_name in per_sample_grads:
            per_sample_grads[weight_name] = per_sample_grads[weight_name].reshape(per_sample_grads[weight_name].shape[0], -1)
            # concatenated_grads = per_sample_grads[weight_name]#torch.cat((concatenated_grads, per_sample_grads[weight_name]), dim=1)
        # per_sample_grads['concatenated_grads'] = concatenated_grads
        
    def sample_feats(self, per_sample_grads):
        for name in per_sample_grads:
            if name in self.subsampling:
                per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
            elif per_sample_grads[name].size(1) > self.cutoff:
                self.subsampling[name] = torch.randperm(per_sample_grads[name].size(1))[:self.cutoff].sort()[0]
                per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
                
    def adjust_representation(self, grads, name):
        representation = torch.index_select(grads, 1, self.subsampling[name].to(self.device))
        return representation
    
    def update(self, per_sample_grads_old, y_pred_old, per_sample_grads_new, y_pred_new):
        if per_sample_grads_old is None:
            return per_sample_grads_new, y_pred_new
        for name in per_sample_grads_old:
            per_sample_grads_old[name] = torch.cat((per_sample_grads_old[name], per_sample_grads_new[name]), dim=0)
        y_pred_old = torch.cat((y_pred_old, y_pred_new), dim=0)
        return per_sample_grads_old, y_pred_old
    
    def prepare_and_calculate_ranks(self, matrices, evaluators, prefix, postfix, normalize=False, batch_first=True, risky_names=()):
        for name, matrix in matrices.items():
            if name in risky_names and not batch_first:
                continue
            if normalize:
                matrix = matrix / (torch.norm(matrix, dim=1, keepdim=True) + self.eps)
            ranks = self.calculate_rank(matrix, batch_first)
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
    
    def calculate_rank(self, matrix, batch_first): # jedyny pomysł to z paddingiem macierzy do maksymalnej
        matrix = matrix if batch_first else matrix.T
        gramian_matrix = matrix @ matrix.T
        rank = torch.linalg.matrix_rank(gramian_matrix).item()
        square_stable_rank = torch.diag(gramian_matrix).sum() / torch.lobpcg(gramian_matrix, k=1)[0][0]
        return rank, square_stable_rank
    
        