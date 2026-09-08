import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.config import TrainingConfig
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.models.oceanembed import OceanEmbedModel
from src.training.loss import OceanEmbedLoss
from src.training.trainer import Trainer

def run_smoke_test():
    print("=" * 60)
    print("OceanEmbed — Phase 15.10 Smoke Test")
    print("=" * 60)

    # 1. Setup minimal configuration
    config = TrainingConfig(micro_batch_size=2, accumulation_steps=2, max_epochs=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OceanEmbedModel(in_channels=12)
    mask = torch.ones(101, 241)
    loss_fn = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.01)
    
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # 2. Create Dummy DataLoaders (4 batches of size 2)
    B, C, H, W = 8, 12, 101, 241
    D = 15
    dummy_x = torch.randn(B, C, H, W)
    dummy_t = torch.randn(B, D, H, W)
    dummy_s = torch.randn(B, D, H, W)
    
    dataset = TensorDataset(dummy_x, dummy_t, dummy_s)
    train_loader = DataLoader(dataset, batch_size=config.micro_batch_size)
    val_loader = DataLoader(dataset, batch_size=config.micro_batch_size) # Reuse for smoke test
    
    # 3. Instantiate Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        save_dir="checkpoints"
    )
    
    # 4. Execute Fit
    print("\n[1] Running Training Loop...")
    trainer.fit(train_loader, val_loader)
    
    # 5. Verify Checkpoint Artifact
    assert os.path.exists("checkpoints/best_model.pt"), "Checkpoint file was not created!"
    print(f"\n[2] Checkpoint created successfully at checkpoints/best_model.pt")

    print("\n" + "=" * 60)
    print("PHASE 15 COMPLETELY FINISHED: READY FOR REAL DATA")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()