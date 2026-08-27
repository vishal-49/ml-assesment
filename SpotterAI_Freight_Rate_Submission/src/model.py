import pandas as pd
import numpy as np
from typing import Dict, Any

from src.config import SEED
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class FreightEnsembleModel:
    """
    Weighted Ensemble Model combining CatBoost, HistGradientBoosting, and LightGBM
    trained on log-transformed posted rates.
    """
    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.cat_model = CatBoostRegressor(
            iterations=900,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            verbose=0,
            random_seed=self.seed
        )
        self.hist_model = HistGradientBoostingRegressor(
            max_iter=450,
            learning_rate=0.03,
            max_leaf_nodes=45,
            min_samples_leaf=20,
            random_state=self.seed
        )
        self.lgb_model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=45,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
            n_jobs=-1
        )
        # Blend weights: CatBoost 0.45, HistGBM 0.45, LightGBM 0.10
        self.w_cat = 0.45
        self.w_hist = 0.45
        self.w_lgb = 0.10

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits sub-models on log1p transformed target.
        """
        y_log = np.log1p(y)
        self.cat_model.fit(X, y_log)
        self.hist_model.fit(X, y_log)
        self.lgb_model.fit(X, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates ensemble predictions in dollar space ($).
        """
        pred_cat = np.expm1(self.cat_model.predict(X))
        pred_hist = np.expm1(self.hist_model.predict(X))
        pred_lgb = np.expm1(self.lgb_model.predict(X))
        
        ensemble_pred = (self.w_cat * pred_cat) + (self.w_hist * pred_hist) + (self.w_lgb * pred_lgb)
        return np.clip(ensemble_pred, a_min=1.0, a_max=None)
