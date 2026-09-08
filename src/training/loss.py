import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.stratification import DensityMonotonicityLoss
from src.training.config import TrainingConfig

class OceanEmbedLoss(nn.Module):
    # combines masked huber loss for temp and salinity + physics-informed density inversion penalty.
    def __init__(
            self, t_mean: float, t_std: float,  s_mean: float, 
            s_std: float, ocean_mask: torch.Tensor, lambda_phys: float = 0.01
    ):
        super().__init__()

        # register mask as non trainable buffer
        self.register_buffer("ocean_mask", ocean_mask.float())
        self.lambda_phys = lambda_phys
        self.salinity_weight = TrainingConfig.salinity_weight

        self.physics_penalty_fn = DensityMonotonicityLoss(
            t_mean=t_mean,
            t_std=t_std,
            s_mean=s_mean,
            s_std=s_std,
            ocean_mask=ocean_mask
        )

    def _masked_huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute huber loss strictly over valid ocean cells, ignoring land and nans

        # 1. sanitize ground truth (land nans = 0.0)
        target_clean = torch.nan_to_num(target, nan=0.0)

        # 2. raw unreduced loss
        raw_loss = F.huber_loss(pred, target_clean, reduction="none")

        # 3. broadcast mask [H,W] -> [1,1,H,W] and apply
        mask_broadcast = self.ocean_mask.unsqueeze(0)
        masked_loss = raw_loss * mask_broadcast

        # 4. strict valid mean
        valid_cells = self.ocean_mask.sum() * target.shape[0]
        if valid_cells == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return masked_loss.sum() / valid_cells
    
    def forward(
            self,
            t_pred: torch.Tensor,
            s_pred: torch.Tensor,
            t_true: torch.Tensor,
            s_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        # 1. supervised errors
        t_loss = self._masked_huber_loss(t_pred, t_true)
        s_loss = self._masked_huber_loss(s_pred, s_true)

        # 2. phy penalty
        phys_loss = torch.tensor(0.0, device=t_pred.device)
        if self.lambda_phys >0.0:
            phys_loss = self.physics_penalty_fn(t_pred, s_pred)

        # 3. composite objective
        total_loss = t_loss + (self.salinity_weight*s_loss) + (self.lambda_phys*phys_loss)

        # 4. observability dict
        loss_dict = {
            "temperature": t_loss.clone().detach(),
            "salinity": s_loss.clone().detach(),
            "physics": phys_loss.clone().detach(),
            "total": total_loss.clone().detach()
        }

        return total_loss, loss_dict