import torch
from torch.utils.data import Dataset

class OceanDataset(Dataset):
    """
    Dynamically slices the pre-loaded 1,419-day memory arrays based on split indices.
    """
    def __init__(self, inputs, targets_t, targets_s, indices: list[int]):
        super().__init__()
        # These are now shared memory references (NumPy arrays)
        self.inputs = inputs
        self.targets_t = targets_t
        self.targets_s = targets_s
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the DataLoader index to the actual chronological Phase 8 day index
        actual_day_idx = self.indices[idx]
        
        x = torch.from_numpy(self.inputs[actual_day_idx]).float()
        t = torch.from_numpy(self.targets_t[actual_day_idx]).float()
        s = torch.from_numpy(self.targets_s[actual_day_idx]).float()
        
        # -----------------------------------------------------------------
        # FORCE PYTORCH CONVENTION: [Channels/Depth, Height, Width]
        # -----------------------------------------------------------------
        if x.shape[-1] == 12: x = x.permute(2, 0, 1)
        if t.shape[-1] == 15: t = t.permute(2, 0, 1)
        if s.shape[-1] == 15: s = s.permute(2, 0, 1)
            
        return x, t, s