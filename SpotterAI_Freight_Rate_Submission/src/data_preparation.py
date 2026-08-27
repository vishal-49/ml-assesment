import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from src.config import TRAIN_PATH, VAL_PATH, VAL_TEMPLATE_PATH, DECEMBER_PATH, SEED


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all raw datasets from disk.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    val_template = pd.read_csv(VAL_TEMPLATE_PATH)
    december_df = pd.read_csv(DECEMBER_PATH)
    return train_df, val_df, val_template, december_df


def build_city_coord_map(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Builds a dictionary mapping city name -> (latitude, longitude)
    combining both train and val datasets.
    """
    city_map = {}
    
    # Process train pickup and delivery
    for _, row in train_df[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates().iterrows():
        city_map[row['pickup']] = (row['pickup_lat'], row['pickup_lon'])
    for _, row in train_df[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates().iterrows():
        city_map[row['delivery']] = (row['delivery_lat'], row['delivery_lon'])
        
    # Process val pickup and delivery
    for _, row in val_df[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates().iterrows():
        if row['pickup'] not in city_map:
            city_map[row['pickup']] = (row['pickup_lat'], row['pickup_lon'])
    for _, row in val_df[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates().iterrows():
        if row['delivery'] not in city_map:
            city_map[row['delivery']] = (row['delivery_lat'], row['delivery_lon'])
            
    return city_map


def enrich_december_inputs(dec_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches december_chart_inputs.csv with missing coordinates, market_index, and quote_signal
    so that the exact feature extraction pipeline can run deterministically.
    """
    dec_enriched = dec_df.copy()
    city_map = build_city_coord_map(train_df, val_df)
    
    # Map coordinates
    dec_enriched['pickup_lat'] = dec_enriched['pickup'].map(lambda c: city_map[c][0] if c in city_map else np.nan)
    dec_enriched['pickup_lon'] = dec_enriched['pickup'].map(lambda c: city_map[c][1] if c in city_map else np.nan)
    dec_enriched['delivery_lat'] = dec_enriched['delivery'].map(lambda c: city_map[c][0] if c in city_map else np.nan)
    dec_enriched['delivery_lon'] = dec_enriched['delivery'].map(lambda c: city_map[c][1] if c in city_map else np.nan)
    
    # Calculate daily December averages for market_index and quote_signal from validation set
    val_temp = val_df.copy()
    val_temp['date_str'] = pd.to_datetime(val_temp['date']).dt.strftime('%Y-%m-%d')
    
    daily_stats = val_temp.groupby('date_str')[['market_index', 'quote_signal']].mean().to_dict('index')
    
    # Global fallback if a date is missing in daily_stats
    overall_mi = train_df['market_index'].median()
    overall_qs = train_df['quote_signal'].median()
    
    def get_market_index(row_date):
        d_str = pd.to_datetime(row_date).strftime('%Y-%m-%d')
        if d_str in daily_stats and not np.isnan(daily_stats[d_str]['market_index']):
            return daily_stats[d_str]['market_index']
        return overall_mi
        
    def get_quote_signal(row_date):
        d_str = pd.to_datetime(row_date).strftime('%Y-%m-%d')
        if d_str in daily_stats and not np.isnan(daily_stats[d_str]['quote_signal']):
            return daily_stats[d_str]['quote_signal']
        return overall_qs
        
    dec_enriched['market_index'] = dec_enriched['date'].apply(get_market_index)
    dec_enriched['quote_signal'] = dec_enriched['date'].apply(get_quote_signal)
    
    return dec_enriched
