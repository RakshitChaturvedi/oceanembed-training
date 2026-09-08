from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # hardware vram management
    micro_batch_size: int = 4
    accumulation_steps: int = 4
    use_amp: bool = True
    num_workers: int = 2

    # optimizer (RAdam)
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    max_grad_norms: float = 1.0

    # epochs and schedulers
    max_epochs: int = 100
    scheduler_patience: float = 4
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 9

    # loss and phy
    salinity_weight: float = 0.5
    lambda_phys: float = 0.01

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps