import sys
import os
import time
import numpy as np
import tracemalloc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.baselines.mlr import MLRBaseline
from src.baselines.xgboost import XGBoostBaseline

def run_benchmark():
    print("=" * 60)
    print("Phase 16.10: Tabular Memory & Scaling Benchmark")
    print("=" * 60)

    # 1. Estimate sizes for 1 Training Fold
    N_ROWS = 13_000_000  # ~1040 days * 12490 ocean pixels
    X_COLS = 14
    Y_COLS = 30

    print(f"[1] Allocating Dummy Tensors for {N_ROWS:,} rows...")
    tracemalloc.start()
    
    # Use float32 to strictly match PyTorch mixed-precision footprint
    X_dummy = np.random.rand(N_ROWS, X_COLS).astype(np.float32)
    Y_dummy = np.random.rand(N_ROWS, Y_COLS).astype(np.float32)

    current, peak = tracemalloc.get_traced_memory()
    print(f"  [✓] Base Allocation: {(current / 1024**3):.2f} GB")
    print(f"  [✓] Peak Memory Hit: {(peak / 1024**3):.2f} GB")
    
    # 2. Benchmark MLR (Full Dataset)
    print("\n[2] Benchmarking MLR (Full 13M rows)...")
    mlr = MLRBaseline()
    
    start_time = time.time()
    mlr.fit(X_dummy, Y_dummy)
    mlr_time = time.time() - start_time
    
    _, peak_mlr = tracemalloc.get_traced_memory()
    print(f"  [✓] MLR Training Time: {mlr_time:.2f} seconds")
    print(f"  [✓] Peak RAM during MLR: {(peak_mlr / 1024**3):.2f} GB")

    # 3. Benchmark XGBoost (10% Subset for projection)
    print("\n[3] Benchmarking XGBoost (Projection from 1.3M rows)...")
    xgb_baseline = XGBoostBaseline(use_gpu=True)
    
    # Override n_estimators to 10 just to get a per-tree time estimate
    xgb_baseline.model.estimator.set_params(n_estimators=10)
    
    subset_size = N_ROWS // 10
    X_sub = X_dummy[:subset_size]
    Y_sub = Y_dummy[:subset_size]

    start_time = time.time()
    xgb_baseline.fit(X_sub, Y_sub)
    xgb_time = time.time() - start_time

    _, peak_xgb = tracemalloc.get_traced_memory()
    
    # Math: (Time for 10% data) * 10 (to reach 100% data) * 10 (to reach 100 trees instead of 10)
    projected_xgb_time = xgb_time * 10 * 10 
    
    print(f"  [✓] XGBoost 10% Test Time: {xgb_time:.2f} seconds")
    print(f"  [✓] Peak RAM during XGB: {(peak_xgb / 1024**3):.2f} GB")
    print(f"  [!] Projected Full XGBoost Time: ~{(projected_xgb_time / 60):.2f} minutes")

    tracemalloc.stop()
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()