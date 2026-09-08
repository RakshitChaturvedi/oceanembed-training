import torch
import numpy as np

def _get_valid_mask(target: torch.Tensor, ocean_mask: torch.Tensor) -> torch.Tensor:
    # combines 2d ocean mask with a dynamic nan check on target
    mask_b = ocean_mask.unsqueeze(0).unsqueeze(0)
    valid_mask = (mask_b==1.0) & (~torch.isnan(target))
    return valid_mask

def compute_depthwise_metrics(pred: torch.Tensor, target: torch.Tensor, ocean_mask: torch.Tensor) -> dict:
    valid_mask = _get_valid_mask(target, ocean_mask)
    
    # Sanitize targets (NaNs become 0, but valid_mask zeroes them out anyway)
    target_clean = torch.nan_to_num(target, nan=0.0)
    
    abs_error = torch.abs(pred - target_clean)
    sq_error = torch.square(pred - target_clean)
    
    masked_abs_error = abs_error * valid_mask.float()
    masked_sq_error = sq_error * valid_mask.float()
    
    # Aggregate over Batch, Height, Width (dims 0, 2, 3), keeping Depth (dim 1)
    sum_abs = masked_abs_error.sum(dim=(0, 2, 3))
    sum_sq = masked_sq_error.sum(dim=(0, 2, 3))
    valid_count = valid_mask.float().sum(dim=(0, 2, 3))
    
    # Prevent divide-by-zero on pure-land/NaN depths
    valid_count = torch.clamp(valid_count, min=1.0)
    
    mae_per_depth = sum_abs / valid_count
    rmse_per_depth = torch.sqrt(sum_sq / valid_count)
    
    return {
        "mae": mae_per_depth.cpu().numpy().tolist(),
        "rmse": rmse_per_depth.cpu().numpy().tolist(),
        "mean_mae": mae_per_depth.mean().item(),
        "mean_rmse": rmse_per_depth.mean().item()
    }

def evaluate_predictions(t_pred: torch.Tensor, s_pred: torch.Tensor, t_true: torch.Tensor, s_true: torch.Tensor, ocean_mask: torch.Tensor) -> dict:
    return {
        "temperature": compute_depthwise_metrics(t_pred, t_true, ocean_mask),
        "salinity": compute_depthwise_metrics(s_pred, s_true, ocean_mask)
    }