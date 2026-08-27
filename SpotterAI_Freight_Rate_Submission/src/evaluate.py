import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates primary regression evaluation metrics: MAE, RMSE, MAPE, MedianAE, and R2.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 6),
        "MAPE(%)": round(mape, 4),
        "MedAE": round(med_ae, 4)
    }


def evaluate_by_group(df: pd.DataFrame, target_col: str, pred_col: str, group_col: str) -> pd.DataFrame:
    """
    Evaluates model performance metrics broken down by categorical group.
    """
    results = []
    for group_name, group_data in df.groupby(group_col):
        if len(group_data) > 0:
            metrics = calculate_metrics(group_data[target_col].values, group_data[pred_col].values)
            metrics[group_col] = group_name
            metrics["Count"] = len(group_data)
            results.append(metrics)
            
    res_df = pd.DataFrame(results)
    cols = [group_col, "Count", "MAE", "RMSE", "R2", "MAPE(%)", "MedAE"]
    return res_df[cols]


def evaluate_by_equipment(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    """
    Evaluates model performance broken down by equipment type.
    """
    return evaluate_by_group(df, target_col, pred_col, "equipment")


def evaluate_by_distance_tier(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    """
    Evaluates performance across short, medium, and long haul distance tiers.
    """
    temp_df = df.copy()
    bins = [0, 500, 1200, 10000]
    labels = ["Short Haul (<500mi)", "Medium Haul (500-1200mi)", "Long Haul (>1200mi)"]
    temp_df["distance_tier"] = pd.cut(temp_df["distance"], bins=bins, labels=labels)
    return evaluate_by_group(temp_df, target_col, pred_col, "distance_tier")
