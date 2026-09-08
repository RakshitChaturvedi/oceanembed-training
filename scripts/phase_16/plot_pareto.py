import matplotlib.pyplot as plt
import numpy as np

# Data from your Phase 16 Leaderboard
models = [
    "MLR", "XGB", "UNet", "UNet+Phys", "CBAM", 
    "OceanEmbed", "OE_L0.1", "OE_L1.0", "OE_L10.0", "OE_L100.0"
]

# x-axis
sal_rmse = [7.2894, 2.7697, 1.6541, 1.6594, 1.3559, 1.4902, 1.2810, 1.7806, 6.3964, 7.1565]

# y-axes
severity = [0.23820, 0.45060, 0.34883, 0.35701, 0.35766, 0.35257, 0.34780, 0.28129, 0.02500, 0.00396]
col_pct = [99.99, 97.73, 91.94, 90.24, 90.94, 87.66, 97.44, 78.81, 82.79, 43.39]

# Ground Truth references
gt_severity = 0.00012
gt_col_pct = 17.38

def get_pareto_frontier(xs, ys, maximize=False):
    """Finds the Pareto frontier. We want to minimize both X (RMSE) and Y (Severity/%)."""
    points = sorted(list(zip(xs, ys, models)), key=lambda x: x[0])
    frontier_x = []
    frontier_y = []
    min_y = float('inf')
    
    for x, y, name in points:
        if y < min_y:
            frontier_x.append(x)
            frontier_y.append(y)
            min_y = y
    return frontier_x, frontier_y

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Salinity RMSE vs Severity ---
ax1.scatter(sal_rmse, severity, color='blue', s=100, zorder=5)
for i, txt in enumerate(models):
    ax1.annotate(txt, (sal_rmse[i], severity[i]), xytext=(5, 5), textcoords='offset points')

# Draw Pareto Frontier
px, py = get_pareto_frontier(sal_rmse, severity)
ax1.plot(px, py, color='red', linestyle='--', label='Pareto Frontier', zorder=4)

# Ground Truth Reference
ax1.axhline(y=gt_severity, color='green', linestyle=':', label=f'Ground Truth ({gt_severity:.5f})')
ax1.set_xlabel("Salinity RMSE (Accuracy →)")
ax1.set_ylabel("Aggregate Inversion Severity (Physics →)")
ax1.set_title("Trade-off: Accuracy vs. Inversion Severity")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Plot 2: Salinity RMSE vs Column Inversion % ---
ax2.scatter(sal_rmse, col_pct, color='purple', s=100, zorder=5)
for i, txt in enumerate(models):
    ax2.annotate(txt, (sal_rmse[i], col_pct[i]), xytext=(5, 5), textcoords='offset points')

# Draw Pareto Frontier
px2, py2 = get_pareto_frontier(sal_rmse, col_pct)
ax2.plot(px2, py2, color='red', linestyle='--', label='Pareto Frontier', zorder=4)

# Ground Truth Reference
ax2.axhline(y=gt_col_pct, color='green', linestyle=':', label=f'Ground Truth ({gt_col_pct}%)')
ax2.set_xlabel("Salinity RMSE (Accuracy →)")
ax2.set_ylabel("Inverted Column % (Physics →)")
ax2.set_title("Trade-off: Accuracy vs. Inverted Columns")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig("pareto_analysis.png", dpi=300)
print("Saved Pareto analysis to pareto_analysis.png")