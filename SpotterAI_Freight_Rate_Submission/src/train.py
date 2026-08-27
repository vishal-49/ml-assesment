import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from src.config import BASE_DIR, SEED, TARGET_COL
from src.data_preparation import load_raw_data
from src.features import FreightFeatureEngineer, get_feature_columns
from src.evaluate import calculate_metrics, evaluate_by_equipment, evaluate_by_distance_tier
from src.model import FreightEnsembleModel


MODEL_DIR = BASE_DIR / "models"


def train_and_evaluate_pipeline() -> Tuple[FreightEnsembleModel, FreightFeatureEngineer, Dict[str, float]]:
    """
    Trains FreightEnsembleModel on Jan-Aug 2025 OOT train split, evaluates on Sept-Oct 2025 OOT val split,
    and then retrains the ensemble model on the full 48,000 loads (Jan-Oct 2025).
    Saves model artifacts to models/ directory.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, val_raw, template, dec_raw = load_raw_data()
    
    print("Fitting Feature Engineering pipeline on development data...")
    fe = FreightFeatureEngineer()
    df_feats = fe.fit_transform(train_raw)
    feature_cols = get_feature_columns()
    
    df_feats['date_dt'] = pd.to_datetime(df_feats['date'])
    
    # OOT Split: Jan-Aug train (80%), Sept-Oct val (20%)
    train_mask = df_feats['date_dt'] < '2025-09-01'
    val_mask = df_feats['date_dt'] >= '2025-09-01'
    
    X_train, y_train = df_feats.loc[train_mask, feature_cols], df_feats.loc[train_mask, TARGET_COL]
    X_val, y_val = df_feats.loc[val_mask, feature_cols], df_feats.loc[val_mask, TARGET_COL]
    
    print(f"OOT Training loads: {len(X_train):,}, Validation loads: {len(X_val):,}")
    
    # Train OOT evaluation model
    oot_ensemble = FreightEnsembleModel(seed=SEED)
    oot_ensemble.fit(X_train, y_train)
    val_preds = oot_ensemble.predict(X_val)
    
    metrics = calculate_metrics(y_val, val_preds)
    print("\n==========================================")
    print("OOT VALIDATION METRICS (Sept-Oct 2025):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("==========================================")
    
    # Print group breakdowns
    val_df_analysis = df_feats.loc[val_mask].copy()
    val_df_analysis["pred_rate"] = val_preds
    
    eq_eval = evaluate_by_equipment(val_df_analysis, TARGET_COL, "pred_rate")
    print("\n--- Evaluation by Equipment Type ---")
    print(eq_eval.to_string(index=False))
    
    dist_eval = evaluate_by_distance_tier(val_df_analysis, TARGET_COL, "pred_rate")
    print("\n--- Evaluation by Distance Tier ---")
    print(dist_eval.to_string(index=False))
    
    # Train final production model on full 48,000 loads (Jan-Oct 2025)
    print("\nRetraining final FreightEnsembleModel on full development dataset (48,000 loads)...")
    fe_full = FreightFeatureEngineer()
    df_full_feats = fe_full.fit_transform(train_raw)
    
    X_full, y_full = df_full_feats[feature_cols], df_full_feats[TARGET_COL]
    
    final_ensemble = FreightEnsembleModel(seed=SEED)
    final_ensemble.fit(X_full, y_full)
    
    joblib.dump(final_ensemble, MODEL_DIR / "ensemble_model.joblib")
    joblib.dump(fe_full, MODEL_DIR / "feature_engineer.joblib")
    print(f"\nFinal model and feature engineer saved to {MODEL_DIR}")
    
    return final_ensemble, fe_full, metrics


if __name__ == "__main__":
    train_and_evaluate_pipeline()
