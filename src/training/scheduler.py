import torch.optim as optim
from src.training.config import TrainingConfig

def build_scheduler(optimizer: optim.Optimizer, config: TrainingConfig):
    # reduces lr when metric stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    return scheduler