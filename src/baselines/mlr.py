import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

class MLRBaseline:
    """
    Phase 16.6: Classical Multiple Linear Regression Baseline.
    Learns Y = XW + b mapping 14 features to 30 targets simultaneously.
    """
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        X: [N, 14]
        Y: [N, 30]
        """
        print(f"Fitting MLR on {X.shape[0]:,} samples. Calculating normal equations...")
        self.model.fit(X, Y)
        print("MLR fit complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)