import torch

def extract_tabular_data(
    x: torch.Tensor, 
    t_target: torch.Tensor, 
    s_target: torch.Tensor, 
    ocean_mask: torch.Tensor,
    lat_grid: torch.Tensor,
    lon_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts purely ocean pixels into a flat tabular format.
    X -> [N, 14] (12 features + Lat + Lon)
    Y -> [N, 30] (15 Temp + 15 Salinity)
    """
    B, C, H, W = x.shape
    bool_mask = (ocean_mask == 1.0)
    
    # Process X
    x_valid = x.permute(0, 2, 3, 1)[:, bool_mask, :].reshape(-1, C)
    
    # Process Lat/Lon (broadcast to match batch size)
    lat_valid = lat_grid[bool_mask].unsqueeze(0).expand(B, -1).reshape(-1, 1)
    lon_valid = lon_grid[bool_mask].unsqueeze(0).expand(B, -1).reshape(-1, 1)
    
    X_tab = torch.cat([x_valid, lat_valid, lon_valid], dim=1)
    
    # Process Targets
    t_valid = t_target.permute(0, 2, 3, 1)[:, bool_mask, :].reshape(-1, 15)
    s_valid = s_target.permute(0, 2, 3, 1)[:, bool_mask, :].reshape(-1, 15)
    Y_tab = torch.cat([t_valid, s_valid], dim=1)
    
    return X_tab, Y_tab

def reconstruct_from_tabular(
    preds: torch.Tensor, 
    ocean_mask: torch.Tensor, 
    batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Restores [N, 30] tabular predictions back to [B, 15, H, W] for evaluation.
    Land pixels are restored as 0.0.
    """
    H, W = ocean_mask.shape
    bool_mask = (ocean_mask == 1.0)
    
    t_spatial = torch.zeros(batch_size, H, W, 15, device=preds.device)
    s_spatial = torch.zeros(batch_size, H, W, 15, device=preds.device)
    
    t_spatial[:, bool_mask, :] = preds[:, :15].view(batch_size, -1, 15)
    s_spatial[:, bool_mask, :] = preds[:, 15:].view(batch_size, -1, 15)
    
    return t_spatial.permute(0, 3, 1, 2), s_spatial.permute(0, 3, 1, 2)