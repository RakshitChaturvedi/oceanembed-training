import torch
import torch.nn as nn
from src.training.config import TrainingConfig

def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    # initializes RAdam to safely handle initial variance. ensures only model's params are passed to optimizer.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.RAdam(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return optimizer