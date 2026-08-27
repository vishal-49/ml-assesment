import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Computes Great Circle Haversine distance in miles between two lat/lon points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    miles = 3956.0 * c
    return miles


class FreightFeatureEngineer:
    """
    Leakage-safe feature engineering pipeline for Freight Rate Prediction.
    Fits missing value medians and categorical statistics on training data.
    """
    def __init__(self):
        self.is_fitted = False
        self.median_weight = 31000.0
        self.median_market_index = 1.0
        self.equip_median_weight = {}
        self.equipment_categories = ['Dry Van', 'Reefer', 'Flatbed']

    def fit(self, df: pd.DataFrame):
        """
        Fits baseline imputation parameters on training DataFrame.
        """
        self.median_weight = float(df['weight'].dropna().median())
        self.median_market_index = float(df['market_index'].dropna().median())
        
        # Group medians for weight by equipment
        equip_grp = df.groupby('equipment')['weight'].median().to_dict()
        for eq in self.equipment_categories:
            self.equip_median_weight[eq] = float(equip_grp.get(eq, self.median_weight))
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw DataFrame into engineered feature space.
        """
        if not self.is_fitted:
            raise RuntimeError("FreightFeatureEngineer must be fitted on training data before calling transform.")
            
        data = df.copy()
        
        # Parse dates
        data['date_dt'] = pd.to_datetime(data['date'])
        
        # 1. Imputation (Leakage-free, using fitted statistics)
        equip_weights = data['equipment'].map(self.equip_median_weight).fillna(self.median_weight)
        data['weight_clean'] = data['weight'].fillna(equip_weights).fillna(self.median_weight)
        data['market_index_clean'] = data['market_index'].fillna(self.median_market_index)
        
        # 2. Spatial & Distance Features
        data['haversine_dist'] = haversine_distance(
            data['pickup_lat'].values,
            data['pickup_lon'].values,
            data['delivery_lat'].values,
            data['delivery_lon'].values
        )
        data['circuity'] = data['distance'] / (data['haversine_dist'] + 1.0)
        data['delta_lat'] = data['delivery_lat'] - data['pickup_lat']
        data['delta_lon'] = data['delivery_lon'] - data['pickup_lon']
        data['midpoint_lat'] = (data['pickup_lat'] + data['delivery_lat']) / 2.0
        data['midpoint_lon'] = (data['pickup_lon'] + data['delivery_lon']) / 2.0
        
        # 3. Weight & Load Density Features
        data['weight_per_mile'] = data['weight_clean'] / (data['distance'] + 1.0)
        data['weight_x_distance'] = data['weight_clean'] * data['distance']
        data['log_distance'] = np.log1p(np.maximum(0, data['distance']))
        data['log_weight'] = np.log1p(np.maximum(0, data['weight_clean']))
        
        # 4. Temporal & Calendar Features
        data['dayofweek'] = data['date_dt'].dt.dayofweek
        data['month'] = data['date_dt'].dt.month
        data['day'] = data['date_dt'].dt.day
        data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
        data['dayofyear'] = data['date_dt'].dt.dayofyear
        data['weekofyear'] = data['date_dt'].dt.isocalendar().week.astype(int)
        data['quarter'] = data['date_dt'].dt.quarter
        
        # Cyclical calendar encodings
        data['sin_dayofweek'] = np.sin(2 * np.pi * data['dayofweek'] / 7.0)
        data['cos_dayofweek'] = np.cos(2 * np.pi * data['dayofweek'] / 7.0)
        data['sin_dayofyear'] = np.sin(2 * np.pi * data['dayofyear'] / 365.25)
        data['cos_dayofyear'] = np.cos(2 * np.pi * data['dayofyear'] / 365.25)
        
        # 5. Market & Demand Interactions
        data['distance_x_market'] = data['distance'] * data['market_index_clean']
        data['market_x_quote'] = data['market_index_clean'] * data['quote_signal']
        data['distance_x_quote'] = data['distance'] * data['quote_signal']
        
        # 6. Equipment One-Hot Encoding
        for eq in self.equipment_categories:
            data[f'equip_{eq.lower().replace(" ", "_")}'] = (data['equipment'] == eq).astype(int)
            
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def get_feature_columns() -> List[str]:
    """
    Returns list of numerical feature columns used for model training.
    """
    return [
        'distance',
        'weight_clean',
        'market_index_clean',
        'quote_signal',
        'pickup_lat',
        'pickup_lon',
        'delivery_lat',
        'delivery_lon',
        'haversine_dist',
        'circuity',
        'delta_lat',
        'delta_lon',
        'midpoint_lat',
        'midpoint_lon',
        'weight_per_mile',
        'weight_x_distance',
        'log_distance',
        'log_weight',
        'dayofweek',
        'month',
        'day',
        'is_weekend',
        'dayofyear',
        'weekofyear',
        'quarter',
        'sin_dayofweek',
        'cos_dayofweek',
        'sin_dayofyear',
        'cos_dayofyear',
        'distance_x_market',
        'market_x_quote',
        'distance_x_quote',
        'equip_dry_van',
        'equip_reefer',
        'equip_flatbed'
    ]
