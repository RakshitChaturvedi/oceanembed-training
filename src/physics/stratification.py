# wraps eos, applies denormalization, calculate vector diff and apply strict masked mean

import torch
import torch.nn as nn
from src.physics.eos import linear_eos

class DensityMonotonicityLoss(nn.Module):
    # penalizes density inversions (heavy water on top of light water)
    def __init__(self, t_mean: float, t_std: float, s_mean: float, s_std: float, ocean_mask: torch.Tensor):
        super().__init__()

        # register_buffer ensures these stats move to gpu along with model, but not treated as trainable weights
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float32))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float32))
        self.register_buffer("s_mean", torch.tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.tensor(s_std, dtype=torch.float32))

        self.register_buffer("ocean_mask", ocean_mask.float())

    def forward(self, t_norm: torch.Tensor, s_norm: torch.Tensor) -> torch.Tensor:
        # 1. denormalize
        t_phys = (t_norm * self.t_std) + self.t_mean
        s_phys = (s_norm * self.s_std) + self.s_mean

        # 2. diff eos
        density = linear_eos(t_phys, s_phys)

        # 3, vectorized depth diff. delta rho = rho (z+1) - rho (z). +ve -> stable
        delta_rho = density[:, 1:, :, :] - density[:, :-1, :, :]

        # 4. inversion penalty. if delta rho -ve, -delta rho = positive. keeps +ve violations and zeros stable layers
        penalty = torch.relu(-delta_rho)

        # 5. broadcast and apply mask [H,W] -> [1,1,H,W]
        mask_3d_batch = self.ocean_mask.unsqueeze(0)
        valid_transitions = mask_3d_batch[:,1:,:,:]*mask_3d_batch[:,:-1,:,:]
        masked_penalty = penalty*valid_transitions

        # 6. strict valid mean calc. total valid ocean cells = (ocean pixels) * (14 depth) * (Batch size)
        total_valid_pairs = valid_transitions.sum()*t_norm.shape[0]

        if total_valid_pairs == 0:
            return torch.tensor(0.0, device=t_norm.device, requires_grad=True)
        physics_loss = masked_penalty.sum() / total_valid_pairs

        return physics_loss
