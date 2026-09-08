import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.oceanembed import OceanEmbedModel
from src.training.loss import OceanEmbedLoss

def run_benchmark():
    print("=" * 60)
    print("OceanEmbed — Phase 15.11 Throughput Benchmark")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CRITICAL: CUDA not detected. Benchmark requires GPU.")
        return

    # 1. Setup Model, Loss, and Optimizer
    print("[1] Loading Model & Allocating VRAM...")
    model = OceanEmbedModel(in_channels=12).to(device)
    model.train()
    
    mask = torch.ones(101, 241).to(device) # Mock mask
    loss_fn = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.01).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda')

    # 2. Hardware constraints (Micro-batch 4, accumulation 4 -> Effective 16)
    micro_batch = 4
    accumulation_steps = 4
    total_steps_per_epoch = 1040 // (micro_batch * accumulation_steps)  # ~65
    
    print(f"  Micro-batch size: {micro_batch}")
    print(f"  Accumulation steps: {accumulation_steps} (Effective batch: {micro_batch * accumulation_steps})")
    print(f"  Optimizer steps per epoch: {total_steps_per_epoch}")

    # 3. Mock Data
    x = torch.randn(micro_batch, 12, 101, 241, device=device)
    t_true = torch.randn(micro_batch, 15, 101, 241, device=device)
    s_true = torch.randn(micro_batch, 15, 101, 241, device=device)

    # 4. GPU Warmup (Get past initialization spikes)
    print("\n[2] Warming up GPU (3 steps)...")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            t_pred, s_pred = model(x)
            loss, _ = loss_fn(t_pred, s_pred, t_true, s_true)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    # 5. The Timed Benchmark
    print("\n[3] Running Timed Benchmark (10 full optimizer steps)...")
    start_time = time.perf_counter()
    
    for _ in range(10):
        optimizer.zero_grad()
        # Simulate accumulation loop
        for _ in range(accumulation_steps):
            with torch.amp.autocast('cuda'):
                t_pred, s_pred = model(x)
                loss, _ = loss_fn(t_pred, s_pred, t_true, s_true)
                loss = loss / accumulation_steps # Normalize loss
            scaler.scale(loss).backward()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 6. The Math
    total_time = end_time - start_time
    time_per_opt_step = total_time / 10.0
    time_per_epoch = time_per_opt_step * total_steps_per_epoch
    
    print("\n" + "=" * 60)
    print("REALITY CHECK: THROUGHPUT PROJECTIONS")
    print("=" * 60)
    print(f"Time per effective batch (16):  {time_per_opt_step:.2f} seconds")
    print(f"Time per epoch:               ~ {time_per_epoch / 60:.2f} minutes")
    print(f"Time for 1 fold (50 epochs):  ~ {(time_per_epoch * 50) / 3600:.2f} hours")
    print(f"Time for 4 folds (200 ep):    ~ {(time_per_epoch * 200) / 3600:.2f} hours")
    print(f"Peak VRAM Reserved:             {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()