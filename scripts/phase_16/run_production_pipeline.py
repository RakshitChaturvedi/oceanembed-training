import sys
import os
import subprocess
import torch
import numpy as np
import xarray as xr
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.models.oceanembed import OceanEmbedModel
from src.physics.eos import linear_eos

TRAIN_SCRIPT = "scripts/phase_16/train_unet.py"
SEEDS = [42, 100, 2023, 777, 888]

# ---------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------
def calculate_metrics(preds, truths):
    valid_mask = ~np.isnan(truths) & (truths != 0.0)
    preds_valid = preds[valid_mask]
    truths_valid = truths[valid_mask]
    
    if len(truths_valid) == 0: return 0.0, 0.0
        
    mse = np.mean((preds_valid - truths_valid) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((truths_valid - preds_valid) ** 2)
    ss_tot = np.sum((truths_valid - np.mean(truths_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return rmse, r2

def calculate_inversion_statistics(t_preds, s_preds, valid_mask_3d, tolerance=0.0):
    if len(t_preds) == 0: return {"layer_rate": 0.0, "column_rate": 0.0, "aggregate_severity": 0.0}

    t = torch.as_tensor(t_preds, dtype=torch.float32)
    s = torch.as_tensor(s_preds, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask_3d, dtype=torch.bool)

    density = linear_eos(t, s)
    delta_rho = torch.diff(density, dim=1)
    
    valid_transitions = mask[:, 1:] & mask[:, :-1]
    inversion_mask = (delta_rho < -tolerance) & valid_transitions

    total_pairs = valid_transitions.sum().item()
    inversion_count = inversion_mask.sum().item()
    layer_rate = (inversion_count / total_pairs * 100.0 if total_pairs > 0 else 0.0)

    valid_columns_mask = valid_transitions.any(dim=1)
    total_valid_columns = valid_columns_mask.sum().item()
    inverted_columns = inversion_mask.any(dim=1) & valid_columns_mask
    column_rate = (inverted_columns.sum().item() / total_valid_columns * 100.0 if total_valid_columns > 0 else 0.0)

    inversion_magnitudes = torch.abs(delta_rho[inversion_mask])
    aggregate_severity = (inversion_magnitudes.sum().item() / total_pairs if total_pairs > 0 else 0.0)

    return {"layer_rate": layer_rate, "column_rate": column_rate, "aggregate_severity": aggregate_severity}

def apply_convective_adjustment(t_preds, s_preds, valid_mask_3d):
    t_adj = torch.tensor(t_preds, dtype=torch.float32)
    s_adj = torch.tensor(s_preds, dtype=torch.float32)
    mask = torch.tensor(valid_mask_3d, dtype=torch.bool)
    touched = torch.zeros_like(mask, dtype=torch.bool)
    passes_run = 0
    dz = torch.tensor([5, 5, 10, 10, 20, 25, 25, 25, 25, 50, 100, 200, 200, 300, 300], dtype=torch.float32)
    
    for _ in range(15):
        density = linear_eos(t_adj, s_adj)
        delta_rho = density[:, 1:] - density[:, :-1]
        inv_mask = (delta_rho < -1e-6) & mask[:, 1:] & mask[:, :-1]
        
        if not inv_mask.any(): break
        passes_run += 1
            
        for parity in [0, 1]:
            for z in range(parity, 14, 2):
                rho_z = linear_eos(t_adj[:, z], s_adj[:, z])
                rho_z1 = linear_eos(t_adj[:, z+1], s_adj[:, z+1])
                
                is_inv = (rho_z > rho_z1 + 1e-6) & mask[:, z] & mask[:, z+1]
                if not is_inv.any(): continue
                    
                w1, w2 = dz[z], dz[z+1]
                w_tot = w1 + w2
                
                t_mix = (t_adj[:, z]*w1 + t_adj[:, z+1]*w2) / w_tot
                s_mix = (s_adj[:, z]*w1 + s_adj[:, z+1]*w2) / w_tot
                
                t_adj[:, z] = torch.where(is_inv, t_mix, t_adj[:, z])
                t_adj[:, z+1] = torch.where(is_inv, t_mix, t_adj[:, z+1])
                s_adj[:, z] = torch.where(is_inv, s_mix, s_adj[:, z])
                s_adj[:, z+1] = torch.where(is_inv, s_mix, s_adj[:, z+1])
                
                touched[:, z] |= is_inv
                touched[:, z+1] |= is_inv
                
    total_valid = mask.sum().item()
    total_touched = (touched & mask).sum().item()
    frac_touched = (total_touched / total_valid * 100.0) if total_valid > 0 else 0.0
                
    return t_adj.numpy(), s_adj.numpy(), passes_run, frac_touched

def load_unet(save_name, fold, device):
    model = OceanEmbedModel(in_channels=12, use_cbam=True).to(device)
    checkpoint = torch.load(f"experiments/phase16/{save_name}/fold_{fold}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("Phase 16.12: Production Pipeline (4-Fold CV & 5-Seed Ensemble)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Geographic Masks globally
    print("[0] Loading Geographic Artifacts...")
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    ocean_mask = torch.from_numpy(mask_ds.values).float()
    mask_bool = (ocean_mask == 1).bool()

    # ---------------------------------------------------------
    # PART 1: AUTOMATED TRAINING
    # ---------------------------------------------------------
    print("\n[1] Commencing Automated Training Suite...")

    def is_trained(save_name, fold):
        # checks if valid checkpoint exists
        path=f"experiments/phase16/{save_name}/fold_{fold}/best_model.pt"
        if not os.path.exists(path):
            return False
        try:
            torch.load(path, map_location="cpu", weights_only=True)
            return True
        except Exception:
            return False
    
    for fold in range(1, 5):
        if is_trained("oe_production_cv", fold):
            print(f" -> CV Run: Fold {fold}/4 [Already completed -- skipping]")
            continue
        cmd = f"python {TRAIN_SCRIPT} --experiment oceanembed --save_name oe_production_cv --lambda_phys 10.0 --fold {fold} --seed 42"
        print(f"  -> CV Run: Fold {fold}/4")
        subprocess.run(cmd, shell=True, check=True) # Uncomment when ready to train

    for seed in SEEDS:
        if is_trained(f"oe_production_ens_seed{seed}", 1):
            print(f" -> Ensemble Run: Seed {seed} [Already completed -- skipping]")
            continue
        cmd = f"python {TRAIN_SCRIPT} --experiment oceanembed --save_name oe_production_ens_seed{seed} --lambda_phys 10.0 --fold 1 --seed {seed}"
        print(f"  -> Ensemble Run: Seed {seed}")
        subprocess.run(cmd, shell=True, check=True) # Uncomment when ready to train

    # ---------------------------------------------------------
    # PART 2: 4-FOLD CROSS VALIDATION EVALUATION
    # ---------------------------------------------------------
    print("\n[2] Evaluating 4-Fold CV (Generalization Check)...")
    cv_metrics = {"t_rmse": [], "s_rmse": []}
    
    for fold in range(1, 5):
        _, _, test_loader = get_dataloaders(fold=fold, batch_size=16)

        try:
            model = load_unet("oe_production_cv", fold, device)
        except FileNotFoundError:
            print(f"  [!] Fold {fold} missing. Did you run the training block?")
            continue
            
        T_p, S_p, T_t, S_t = [], [], [], []
        with torch.no_grad():
            for x, t_true, s_true in test_loader:
                t_pred, s_pred = model(x.to(device))
                T_p.append(t_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                S_p.append(s_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                T_t.append(t_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy())
                S_t.append(s_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy())
                
        t_p_arr, t_t_arr = np.vstack(T_p), np.vstack(T_t)
        s_p_arr, s_t_arr = np.vstack(S_p), np.vstack(S_t)
        gt_valid_mask = ~np.isnan(t_t_arr)

        t_rmse, _ = calculate_metrics(t_p_arr[gt_valid_mask], t_t_arr[gt_valid_mask])
        s_rmse, _ = calculate_metrics(s_p_arr[gt_valid_mask], s_t_arr[gt_valid_mask])
        cv_metrics["t_rmse"].append(t_rmse)
        cv_metrics["s_rmse"].append(s_rmse)
        
        print(f"     Fold {fold} - T RMSE: {t_rmse:.4f}, S RMSE: {s_rmse:.4f}")
        del model, T_p, S_p, T_t, S_t, t_p_arr, s_p_arr; gc.collect()

    if cv_metrics["t_rmse"]:
        print(f"  => Fold-Averaged T RMSE: {np.mean(cv_metrics['t_rmse']):.4f} ± {np.std(cv_metrics['t_rmse']):.4f}")
        print(f"  => Fold-Averaged S RMSE: {np.mean(cv_metrics['s_rmse']):.4f} ± {np.std(cv_metrics['s_rmse']):.4f}")

    # ---------------------------------------------------------
    # PART 3: EPISTEMIC ENSEMBLE EVALUATION & UNCERTAINTY
    # ---------------------------------------------------------
    print("\n[3] Evaluating Epistemic Ensemble (Adjust-then-Average Method)...")
    _, _, test_loader_f1 = get_dataloaders(fold=1, batch_size=16)
    
    T_t, S_t = [], []
    for _, t_true, s_true in test_loader_f1:
        T_t.append(t_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy())
        S_t.append(s_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy())
    t_t_arr, s_t_arr = np.vstack(T_t), np.vstack(S_t)
    gt_valid_mask = ~np.isnan(t_t_arr)

    ensemble_adjusted_t, ensemble_adjusted_s = [], []

    for seed in SEEDS:
        print(f"  -> Processing Ensemble Member (Seed {seed})...")
        try:
            model = load_unet(f"oe_production_ens_seed{seed}", 1, device)
        except FileNotFoundError:
            print(f"  [!] Seed {seed} missing. Skipping...")
            continue
            
        T_p, S_p = [], []
        with torch.no_grad():
            for x, _, _ in test_loader_f1:
                t_pred, s_pred = model(x.to(device))
                T_p.append(t_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                S_p.append(s_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                
        print(f"     Applying Convective Adjustment...")
        t_adj, s_adj, passes, frac = apply_convective_adjustment(np.vstack(T_p), np.vstack(S_p), gt_valid_mask)
        ensemble_adjusted_t.append(t_adj)
        ensemble_adjusted_s.append(s_adj)
        del model, T_p, S_p; gc.collect()

    if ensemble_adjusted_t:
        print("\n[4] Calculating and Saving Uncertainty & Final Metrics...")
        os.makedirs("experiments/phase16/oe_production_ens", exist_ok=True)
        
        # Save Uncertainty (Standard Deviation)
        uncertainty_t = np.std(ensemble_adjusted_t, axis=0)
        uncertainty_s = np.std(ensemble_adjusted_s, axis=0)
        np.save("experiments/phase16/oe_production_ens/uncertainty_t.npy", uncertainty_t)
        np.save("experiments/phase16/oe_production_ens/uncertainty_s.npy", uncertainty_s)
        print("  -> Saved epistemic uncertainty arrays to experiments/phase16/oe_production_ens/")
        
        # Ensemble Average
        final_t = np.mean(ensemble_adjusted_t, axis=0)
        final_s = np.mean(ensemble_adjusted_s, axis=0)
        
        t_rmse, t_r2 = calculate_metrics(final_t[gt_valid_mask], t_t_arr[gt_valid_mask])
        s_rmse, s_r2 = calculate_metrics(final_s[gt_valid_mask], s_t_arr[gt_valid_mask])
        final_inv = calculate_inversion_statistics(final_t, final_s, gt_valid_mask)

        print("\n" + "=" * 90)
        print("FINAL PRODUCTION MODEL: 5-SEED ENSEMBLE (OE_L10.0 + Post-Processing)")
        print("=" * 90)
        print(f"Temperature : RMSE = {t_rmse:.4f} | R² = {t_r2:.3f}")
        print(f"Salinity    : RMSE = {s_rmse:.4f} | R² = {s_r2:.3f}")
        print(f"Physics     : Layer Inversion = {final_inv['layer_rate']:.2f}% | Column = {final_inv['column_rate']:.2f}%")
        print("=" * 90)

if __name__ == "__main__":
    main()