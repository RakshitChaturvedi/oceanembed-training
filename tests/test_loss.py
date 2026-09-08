import torch
import pytest
from src.training.loss import OceanEmbedLoss

@pytest.fixture
def mock_loss():
    # 2x2 mask: 3 ocean pixels, 1 land pixel
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    return OceanEmbedLoss(t_mean=20.0, t_std=2.0, s_mean=35.0, s_std=1.0, ocean_mask=mask, lambda_phys=0.01)

def test_masked_huber_behavior(mock_loss):
    """Test 2: Changing land predictions shouldn't change the supervised loss."""
    pred1 = torch.zeros(1, 15, 2, 2)
    pred2 = torch.zeros(1, 15, 2, 2)
    
    # Change prediction explicitly on the land pixel
    pred2[0, :, 1, 1] = 999.9 
    
    target = torch.ones(1, 15, 2, 2)
    
    loss1 = mock_loss._masked_huber_loss(pred1, target)
    loss2 = mock_loss._masked_huber_loss(pred2, target)
    
    assert torch.allclose(loss1, loss2), "Land prediction altered the loss!"

def test_nan_target_sanitization(mock_loss):
    """Test 3: Target NaNs over land shouldn't corrupt the loss."""
    pred = torch.zeros(1, 15, 2, 2)
    target = torch.ones(1, 15, 2, 2)
    target[0, :, 1, 1] = float('nan') # NaN on land
    
    loss = mock_loss._masked_huber_loss(pred, target)
    assert not torch.isnan(loss), "NaN escaped into the loss calculation!"

def test_salinity_weighting(mock_loss):
    """Test 4: Total supervised = T_loss + 0.5 * S_loss."""
    mock_loss.lambda_phys = 0.0 # Isolate supervised
    
    t_pred = torch.zeros(1, 1, 2, 2) # Simplify depth to 1 for test
    s_pred = torch.zeros(1, 1, 2, 2)
    t_true = torch.ones(1, 1, 2, 2)
    s_true = torch.ones(1, 1, 2, 2) * 2.0
    
    total, d = mock_loss(t_pred, s_pred, t_true, s_true)
    
    # T_loss = ~0.5, S_loss = ~1.5
    assert torch.allclose(total, d["temperature"] + 0.5 * d["salinity"])