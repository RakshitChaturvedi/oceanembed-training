import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import joblib

class XGBoostBaseline:
    """
    Phase 16.7: Classical Nonlinear Tabular Baseline.
    Trains 30 independent XGBoost trees (one for each target depth) using GPU acceleration.
    """
    def __init__(self, use_gpu: bool = True):
        # Configure for RTX 4050 Mobile (6GB VRAM)
        tree_method = "hist"
        device = "cuda" if use_gpu else "cpu"
        
        # Base estimator for a single target
        base_estimator = xgb.XGBRegressor(
            n_estimators=100,       # Number of boosting rounds
            max_depth=6,            # Tree depth
            learning_rate=0.1,      
            tree_method=tree_method,
            device=device,
            n_jobs=-1               # Use all CPU cores for data staging
        )
        
        # Wrap to handle the 30 outputs automatically
        self.model = MultiOutputRegressor(base_estimator)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        print(f"Fitting XGBoost (MultiOutput: 30 targets) on {X.shape[0]:,} samples...")
        print("This will train 30 separate GPU-accelerated trees.")
        self.model.fit(X, Y)
        print("XGBoost fit complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)