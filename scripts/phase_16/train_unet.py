import os
import argparse
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.models.oceanembed import OceanEmbedModel
from src.training.config import TrainingConfig
from src.training.loss import OceanEmbedLoss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.trainer import Trainer
import numpy as np
import xarray as xr
import random

def set_seed(seed):
    """Locks all random operations for perfect reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="OceanEmbed Phase 16 U-Net Baselines")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        choices=["plain_unet", "plain_unet_physics", "cbam_unet", "oceanembed"],
        help="Which architecture to train."
    )
    parser.add_argument("--fold", type=int, default=1, help="LOYO fold to train (1-4).")
    # Fixed typo: changed lamba_phys to lambda_phys
    parser.add_argument("--lambda_phys", type=float, default=None, help="Override phy loss weight.")
    # Inside your argparse definition, add these two lines:
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_name", type=str, default=None, help="Override default save directory name")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    print("=" * 60)
    print(f"Phase 16.8: Training {args.experiment.upper()} | Fold: {args.fold}")
    print("=" * 60)

    # 1. Config & Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(
        micro_batch_size=4, 
        accumulation_steps=4  # Effective batch = 16
    )

    # 2. Configure Ablation Matrix
    if args.save_name:
        save_name = args.save_name
    else:
        save_name = args.experiment
        if args.lambda_phys is not None:
            save_name = f"{args.experiment}_L{args.lambda_phys}"
    if args.experiment == "plain_unet":
        use_cbam = False
        lambda_phys = 0.0
    elif args.experiment == "plain_unet_physics":
        use_cbam = False
        lambda_phys = config.lambda_phys
    elif args.experiment == "cbam_unet":
        use_cbam = True
        lambda_phys = 0.0
    elif args.experiment == "oceanembed":
        use_cbam = True
        lambda_phys = config.lambda_phys  # The full architecture!

    # --- NEW LOGIC: Override Lambda and Save Name ---
    if args.lambda_phys is not None:
        lambda_phys = args.lambda_phys
        if not args.save_name:
            save_name = f"{args.experiment}_L{args.lambda_phys}"
    # ------------------------------------------------

    # 3. Load Phase 5 Ocean Mask (required for Loss function)
    try:
        mask_path = "data/processed/phase5/ocean_mask_3d.nc"
        ocean_mask = torch.from_numpy(xr.open_dataarray(mask_path).values).float()
    except Exception as e:
        print(f"Warning: Could not load Phase 5 mask ({e}). Using synthetic mask.")
        ocean_mask = torch.ones(15, 101, 241)

    # 4. Initialize Model and Loss
    print(f"[1] Initializing Model (CBAM={use_cbam}) and Loss (Lambda_phys={lambda_phys})...")
    model = OceanEmbedModel(in_channels=12, use_cbam=use_cbam)
    
    loss_fn = OceanEmbedLoss(
        t_mean=0.0,
        t_std=1.0,
        s_mean=0.0,
        s_std=1.0,
        ocean_mask=ocean_mask, 
        lambda_phys=lambda_phys
    )

    # 5. Build Optimizer & Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 6. Load Phase 8 DataLoaders
    print(f"[2] Initializing Phase 8 DataLoaders for Fold {args.fold}...")
    train_loader, val_loader, _ = get_dataloaders(fold=args.fold, batch_size=config.micro_batch_size)

    # 7. Initialize Trainer (Using the dynamic save_name)
    save_dir = f"experiments/phase16/{save_name}/fold_{args.fold}"
    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        scheduler=scheduler, config=config, device=device, save_dir=save_dir
    )

    # 8. Train!
    print(f"\n[3] Launching Training Loop (Saving to {save_dir})...")
    trainer.fit(train_loader, val_loader)

    print("\n" + "=" * 60)
    print(f"PHASE 16.8 COMPLETE: {save_name.upper()} TRAINED")
    print("=" * 60)

if __name__ == "__main__":
    main()