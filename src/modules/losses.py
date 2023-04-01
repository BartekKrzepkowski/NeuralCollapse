import math
from copy import deepcopy

import torch

from src.modules.metrics import acc_metric
from src.modules.regularizers import FisherPenaly, BatchGradCovariancePenalty
from src.utils import common
from src.utils.utils_regularizers import distance_between_models


class ClassificationLoss(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        acc = acc_metric(y_pred, y_true)
        evaluators = {
            'loss': loss.item(),
            'acc': acc
        }
        return loss, evaluators


class FisherPenaltyLoss(torch.nn.Module):
    def __init__(self, model, general_criterion_name, num_classes, whether_record_trace=False, fpw=0.0):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]())
        self.regularizer = FisherPenaly(model, common.LOSS_NAME_MAP[general_criterion_name](), num_classes)
        self.whether_record_trace = whether_record_trace
        self.fpw = fpw
        #przygotowanie do logowania co n kroków
        self.overall_trace_buffer = None
        self.traces = None
        self.trajectory_length_stochastic_squared = 0.0
        self.lr = 0.1

    def forward(self, y_pred, y_true):
        traces = {}
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_trace:# and self.regularizer.model.training:
            overall_trace, traces = self.regularizer(y_pred)
            evaluators['overall_trace'] = overall_trace.item()
            evaluators['distance_from_initialization_l2_squared'] = distance_between_models(self.model_zero, self.regularizer.model, 'l2')
            # evaluators['distance_from_initialization_cosine'] = distance_between_models(self.model_zero, self.regularizer.model, 'cosine')
            self.trajectory_length_stochastic_squared += self.lr ** 2 * overall_trace.item()
            evaluators['trajectory_length_stochastic_squared'] = self.trajectory_length_stochastic_squared
            if self.fpw > 0:
                loss += self.fpw * overall_trace
        return loss, evaluators, traces
    

class BatchGradCovarianceLoss(torch.nn.Module):
    def __init__(self, model, loader, general_criterion_name, bgcw=0.0, n=10):
        super().__init__()
        self.criterion = ClassificationLoss(common.LOSS_NAME_MAP[general_criterion_name]())
        self.regularizer = BatchGradCovariancePenalty(model, loader, common.LOSS_NAME_MAP[general_criterion_name])
        self.bgcw = bgcw
        self.n = n

    def forward(self, y_pred, y_true):
        loss, evaluators = self.criterion(y_pred, y_true)
        if self.whether_record_logdet:
            log_det = self.regularizer(self.n)
            evaluators['bgc_log_det'] = log_det.item()
            if self.bgcw > 0:
                loss += self.fpw * log_det
        return loss, evaluators
    

class MSESoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = torch.nn.MSELoss()
   

    def forward(self, y_pred, y_true):
        y_true = torch.nn.functional.one_hot(y_true, num_classes=10).float()
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        loss = self.criterion(y_pred, y_true)
        return loss
