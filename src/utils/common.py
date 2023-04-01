import torch

from src.data.datasets import get_mnist, get_cifar10, get_cifar100, get_tinyimagenet, get_food101
from src.modules.losses import ClassificationLoss, FisherPenaltyLoss, MSESoftmaxLoss
from src.modules.architectures.models import MLP, SimpleCNN, MLPwithNorm, SimpleCNNwithNorm
from src.modules.architectures.resnets import ResNet18, ResNet34
from src.visualization.clearml_logger import ClearMLLogger
from src.visualization.tensorboard_pytorch import TensorboardPyTorch
from src.visualization.wandb_logger import WandbLogger

ACT_NAME_MAP = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'identity': torch.nn.Identity
}

DATASET_NAME_MAP = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
    'tinyimagenet': get_tinyimagenet,
    'food101': get_food101,
}

LOGGERS_NAME_MAP = {
    'clearml': ClearMLLogger,
    'tensorboard': TensorboardPyTorch,
    'wandb': WandbLogger
}

LOSS_NAME_MAP = {
    'ce': torch.nn.CrossEntropyLoss,
    'cls': ClassificationLoss,
    'nll': torch.nn.NLLLoss,
    'mse': torch.nn.MSELoss,
    'mse_softmax': MSESoftmaxLoss,
    'fp': FisherPenaltyLoss
}

MODEL_NAME_MAP = {
    'mlp': MLP,
    'mlp_with_norm': MLPwithNorm,
    'simple_cnn': SimpleCNN,
    'simple_cnn_with_norm': SimpleCNNwithNorm,
    'resnet18': ResNet18,
    'resnet34': ResNet34
}

NORM_LAYER_NAME_MAP = {
    'bn1d': torch.nn.BatchNorm1d,
    'bn2d': torch.nn.BatchNorm2d,
    'layer_norm': torch.nn.LayerNorm,
    'group_norm': torch.nn.GroupNorm,
    'instance_norm_1d': torch.nn.InstanceNorm1d,
    'instance_norm_2d': torch.nn.InstanceNorm2d,
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}

SCHEDULER_NAME_MAP = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}
