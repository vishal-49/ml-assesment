import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"
SCORER_DIR = BASE_DIR / "scorer_results"
REPORTS_DIR = BASE_DIR / "reports"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VAL_PATH = DATA_DIR / "validation.csv"
VAL_TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

FINAL_VAL_PRED_PATH = OUTPUT_DIR / "validation_predictions.csv"
DECEMBER_PRED_PATH = DECEMBER_PATH

# Modeling settings
SEED = 42
TARGET_COL = "posted_rate"
ID_COL = "load_id"

NUMERICAL_COLS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

CATEGORICAL_COLS = [
    "pickup",
    "delivery",
    "equipment",
]

DATE_COL = "date"
