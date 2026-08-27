import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import BASE_DIR, VAL_PATH, DECEMBER_PATH, FINAL_VAL_PRED_PATH, TARGET_COL
from src.data_preparation import load_raw_data, enrich_december_inputs
from src.features import FreightFeatureEngineer, get_feature_columns
from src.model import FreightEnsembleModel


MODEL_DIR = BASE_DIR / "models"


def generate_predictions():
    """
    Loads saved FreightEnsembleModel pipeline, generates validation_predictions.csv
    and fills december-chart-inputs.csv.
    """
    model_path = MODEL_DIR / "ensemble_model.joblib"
    fe_path = MODEL_DIR / "feature_engineer.joblib"
    
    if not model_path.exists() or not fe_path.exists():
        raise FileNotFoundError("Trained model or feature engineer artifact not found. Run train.py first.")
        
    model: FreightEnsembleModel = joblib.load(model_path)
    fe: FreightFeatureEngineer = joblib.load(fe_path)
    feature_cols = get_feature_columns()
    
    train_raw, val_raw, template, dec_raw = load_raw_data()
    
    # 1. Generate Validation Predictions (12,000 loads)
    print("Generating validation predictions (12,000 loads)...")
    val_feats = fe.transform(val_raw)
    X_val = val_feats[feature_cols]
    
    val_preds = model.predict(X_val)
    val_preds = np.round(val_preds, 2)
    
    val_submission = pd.DataFrame({
        "load_id": val_raw["load_id"],
        "predicted_rate": val_preds
    })
    
    val_submission.to_csv(FINAL_VAL_PRED_PATH, index=False)
    print(f"Saved {len(val_submission):,} predictions to {FINAL_VAL_PRED_PATH}")
    
    # 2. Enrich and Generate December Predictions (31 days)
    print("Enriching and predicting December chart inputs (31 days)...")
    dec_enriched = enrich_december_inputs(dec_raw, train_raw, val_raw)
    dec_feats = fe.transform(dec_enriched)
    X_dec = dec_feats[feature_cols]
    
    dec_preds = model.predict(X_dec)
    dec_preds = np.round(dec_preds, 2)
    
    dec_output = dec_raw.copy()
    dec_output["predicted_rate"] = dec_preds
    
    dec_output.to_csv(DECEMBER_PATH, index=False)
    print(f"Updated December chart inputs predictions at {DECEMBER_PATH}")
    
    return val_submission, dec_output


if __name__ == "__main__":
    generate_predictions()
