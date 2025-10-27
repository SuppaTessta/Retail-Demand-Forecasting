"""
Feature Engineering Module for Rossmann Store Sales
Author: Your Name
Date: October 2025
Python: 3.13.5
"""

import pandas as pd
import numpy as np
from datetime import datetime

def extract_date_features(df, date_col='Date'):
    """
    Extract comprehensive date-based features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with date column
    date_col : str
        Name of date column
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with new date features
    """
    df = df.copy()
    
    print("📅 Extracting date features...")
    
    # Basic date components
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['Quarter'] = df[date_col].dt.quarter
    
    # Binary features
    df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)
    df['IsMonthStart'] = df[date_col].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df[date_col].dt.is_month_end.astype(int)
    
    # Additional useful features
    df['DayOfMonth'] = df[date_col].dt.day
    df['DayOfYear'] = df[date_col].dt.dayofyear
    
    print(f"  ✓ Created 10 date features")
    
    return df


def create_promo2_feature(df):
    """
    Create Promo2 active indicator based on PromoInterval
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with Promo2 columns
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with Promo2Active feature
    """
    df = df.copy()
    
    print("🎯 Creating Promo2 features...")
    
    # Map PromoInterval to months
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    def is_promo2_active(row):
        """Check if Promo2 is active for given month"""
        if row['Promo2'] == 0:
            return 0
        
        if pd.isna(row['PromoInterval']) or row['PromoInterval'] == 'None':
            return 0
        
        try:
            promo_months = [month_map[m.strip()] for m in row['PromoInterval'].split(',')]
            return 1 if row['Month'] in promo_months else 0
        except:
            return 0
    
    df['Promo2Active'] = df.apply(is_promo2_active, axis=1)
    
    print(f"  ✓ Created Promo2Active feature")
    
    return df


def transform_competition_distance(df):
    """
    Transform CompetitionDistance for better model performance
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with CompetitionDistance
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with transformed features
    """
    df = df.copy()
    
    print("🏪 Transforming competition features...")
    
    # Inverse square root transformation (helps with outliers)
    df['CompDistanceTransformed'] = 1 / np.sqrt(df['CompetitionDistance'])
    
    # Calculate how long competition has been open
    df['CompetitionOpenMonths'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                   (df['Month'] - df['CompetitionOpenSinceMonth'])
    
    # Set negative values to 0 (competition not yet opened)
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].clip(lower=0)
    
    # Create binary feature: has competition or not
    df['HasCompetition'] = (df['CompetitionDistance'] < 100000).astype(int)
    
    print(f"  ✓ Created 3 competition features")
    
    return df


def create_store_features(df):
    """
    Create store-level aggregated features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with Store and Sales columns
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with store features
    """
    df = df.copy()
    
    print("🏬 Creating store-level features...")
    
    if 'Sales' in df.columns:
        # Average sales per store
        store_avg = df.groupby('Store')['Sales'].mean().to_dict()
        df['Store_AvgSales'] = df['Store'].map(store_avg)
        
        # Sales standard deviation per store
        store_std = df.groupby('Store')['Sales'].std().to_dict()
        df['Store_StdSales'] = df['Store'].map(store_std)
        
        print(f"  ✓ Created 2 store aggregate features")
    else:
        print(f"  ⚠️  Skipped (no Sales column)")
    
    return df


def create_lag_features(df, target_col='Sales', lags=[7, 14, 30]):
    """
    Create lag features (previous values)
    CRITICAL for time series forecasting!
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset sorted by Store and Date
    target_col : str
        Column to create lags for
    lags : list
        List of lag periods (days)
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with lag features
    """
    df = df.copy()
    
    if target_col not in df.columns:
        print(f"  ⚠️  Skipping lag features (no {target_col} column)")
        return df
    
    print(f"⏳ Creating lag features for {lags}...")
    
    # Ensure data is sorted
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    for lag in lags:
        df[f'{target_col}_Lag_{lag}'] = df.groupby('Store')[target_col].shift(lag)
    
    print(f"  ✓ Created {len(lags)} lag features")
    
    return df


def create_rolling_features(df, target_col='Sales', windows=[7, 14, 30]):
    """
    Create rolling window statistics
    VERY important for capturing trends!
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset sorted by Store and Date
    target_col : str
        Column to calculate rolling stats
    windows : list
        List of window sizes (days)
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with rolling features
    """
    df = df.copy()
    
    if target_col not in df.columns:
        print(f"  ⚠️  Skipping rolling features (no {target_col} column)")
        return df
    
    print(f"📊 Creating rolling features for windows {windows}...")
    
    # Ensure data is sorted
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_RollingMean_{window}'] = \
            df.groupby('Store')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Rolling standard deviation
        df[f'{target_col}_RollingStd_{window}'] = \
            df.groupby('Store')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Rolling min and max
        df[f'{target_col}_RollingMin_{window}'] = \
            df.groupby('Store')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
        
        df[f'{target_col}_RollingMax_{window}'] = \
            df.groupby('Store')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
    
    print(f"  ✓ Created {len(windows) * 4} rolling features")
    
    return df


def create_exponential_features(df, target_col='Sales', spans=[7, 14, 30]):
    """
    Create exponentially weighted moving averages
    Gives more weight to recent values
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset sorted by Store and Date
    target_col : str
        Column to calculate EWM
    spans : list
        List of span values
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with EWM features
    """
    df = df.copy()
    
    if target_col not in df.columns:
        print(f"  ⚠️  Skipping EWM features (no {target_col} column)")
        return df
    
    print(f"📈 Creating exponential weighted features for spans {spans}...")
    
    # Ensure data is sorted
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    for span in spans:
        df[f'{target_col}_EWM_{span}'] = \
            df.groupby('Store')[target_col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
    
    print(f"  ✓ Created {len(spans)} EWM features")
    
    return df


def feature_engineering_pipeline(df, is_train=True):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed dataset
    is_train : bool
        Whether this is training data (has Sales column)
        
    Returns:
    --------
    df : pandas DataFrame
        Feature-engineered dataset
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    initial_cols = len(df.columns)
    print(f"\nInitial columns: {initial_cols}")
    
    # Extract date features
    df = extract_date_features(df)
    
    # Create Promo2 feature
    df = create_promo2_feature(df)
    
    # Transform competition distance
    df = transform_competition_distance(df)
    
    # Create store-level features
    df = create_store_features(df)
    
    # Create lag and rolling features (only if Sales exists)
    if is_train and 'Sales' in df.columns:
        df = create_lag_features(df)
        df = create_rolling_features(df)
        df = create_exponential_features(df)
        
        # Drop rows with NaN (due to lag/rolling windows)
        print(f"\n🧹 Cleaning NaN values from lag/rolling features...")
        initial_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped_rows = initial_rows - len(df)
        print(f"  ✓ Dropped {dropped_rows:,} rows with NaN")
    
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final shape: {df.shape}")
    print(f"📈 New features created: {new_features}")
    print(f"📋 Total columns: {final_cols}")
    
    return df


if __name__ == "__main__":
    print("="*70)
    print("RUNNING FEATURE ENGINEERING")
    print("="*70)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    train = pd.read_csv('data/processed/train_preprocessed.csv', parse_dates=['Date'])
    test = pd.read_csv('data/processed/test_preprocessed.csv', parse_dates=['Date'])
    print(f"  ✓ Train loaded: {train.shape}")
    print(f"  ✓ Test loaded: {test.shape}")
    
    # Engineer features
    train_fe = feature_engineering_pipeline(train, is_train=True)
    test_fe = feature_engineering_pipeline(test, is_train=False)
    
    # Save feature-engineered data
    print("\n💾 Saving feature-engineered data...")
    train_fe.to_csv('data/processed/train_features.csv', index=False)
    test_fe.to_csv('data/processed/test_features.csv', index=False)
    print(f"  ✓ Saved train_features.csv: {train_fe.shape}")
    print(f"  ✓ Saved test_features.csv: {test_fe.shape}")
    
    print("\n✅ Feature engineering complete!")
    print(f"\n📋 Feature columns ({len(train_fe.columns)}):")
    for i, col in enumerate(train_fe.columns, 1):
        print(f"  {i:2d}. {col}")

