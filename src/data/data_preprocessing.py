"""
Data Preprocessing Module for Rossmann Store Sales
"""

import pandas as pd

def load_data(train_path='data/raw/train.csv', 
              test_path='data/raw/test.csv',
              store_path='data/raw/store.csv'):
    """
    Load Rossmann datasets
    
    Parameters:
    -----------
    train_path : str
        Path to train.csv
    test_path : str
        Path to test.csv  
    store_path : str
        Path to store.csv
        
    Returns:
    --------
    train, test, store : pandas DataFrames
    """
    print("📂 Loading datasets...")
    train = pd.read_csv(train_path, parse_dates=['Date'], dtype={'StateHoliday': str})
    test = pd.read_csv(test_path, parse_dates=['Date'], dtype={'StateHoliday': str})
    store = pd.read_csv(store_path)
    
    print(f"  ✓ Train: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  ✓ Test:  {test.shape[0]:,} rows × {test.shape[1]} columns")
    print(f"  ✓ Store: {store.shape[0]:,} rows × {store.shape[1]} columns")
    
    return train, test, store


def handle_missing_values(df):
    """
    Handle missing values in the store dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Store dataset with missing values
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with handled missing values
    """
    df = df.copy()
    
    print("\n🔧 Handling missing values...")
    initial_missing = df.isnull().sum().sum()
    print(f"  Initial missing values: {initial_missing}")
    
    # CompetitionDistance: If no competitor, set large distance
    if 'CompetitionDistance' in df.columns:
        missing_comp_dist = df['CompetitionDistance'].isnull().sum()
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(100000)
        print(f"  ✓ Filled {missing_comp_dist} missing CompetitionDistance values")
    
    # CompetitionOpenSince: If no competitor, set to year 1900
    if 'CompetitionOpenSinceMonth' in df.columns:
        missing_comp_month = df['CompetitionOpenSinceMonth'].isnull().sum()
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        print(f"  ✓ Filled {missing_comp_month} missing CompetitionOpenSinceMonth values")
        
    if 'CompetitionOpenSinceYear' in df.columns:
        missing_comp_year = df['CompetitionOpenSinceYear'].isnull().sum()
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(1900)
        print(f"  ✓ Filled {missing_comp_year} missing CompetitionOpenSinceYear values")
    
    # Promo2Since: If not in promo2, set to 0
    if 'Promo2SinceWeek' in df.columns:
        missing_promo_week = df['Promo2SinceWeek'].isnull().sum()
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        print(f"  ✓ Filled {missing_promo_week} missing Promo2SinceWeek values")
        
    if 'Promo2SinceYear' in df.columns:
        missing_promo_year = df['Promo2SinceYear'].isnull().sum()
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(1900)
        print(f"  ✓ Filled {missing_promo_year} missing Promo2SinceYear values")
    
    # PromoInterval: Fill with 'None'
    if 'PromoInterval' in df.columns:
        missing_promo_interval = df['PromoInterval'].isnull().sum()
        df['PromoInterval'] = df['PromoInterval'].fillna('None')
        print(f"  ✓ Filled {missing_promo_interval} missing PromoInterval values")
    
    final_missing = df.isnull().sum().sum()
    print(f"  ✓ Final missing values: {final_missing}")
    
    return df


def merge_store_data(df, store):
    """
    Merge train/test data with store information
    
    Parameters:
    -----------
    df : pandas DataFrame
        Train or test data
    store : pandas DataFrame
        Store information
        
    Returns:
    --------
    merged : pandas DataFrame
        Merged dataset
    """
    print("\n🔗 Merging with store information...")
    initial_shape = df.shape
    merged = pd.merge(df, store, on='Store', how='left')
    print(f"  ✓ Shape: {initial_shape} → {merged.shape}")
    
    return merged


def filter_closed_stores(df):
    """
    Remove records where store was closed
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with Open column
        
    Returns:
    --------
    df : pandas DataFrame
        Filtered dataset
    """
    initial_rows = len(df)
    df = df[df['Open'] == 1].copy()
    removed_rows = initial_rows - len(df)
    
    print(f"\n🚪 Filtering closed stores...")
    print(f"  ✓ Removed {removed_rows:,} closed store records")
    print(f"  ✓ Remaining: {len(df):,} rows")
    
    return df


def remove_zero_sales(df):
    """
    Remove records with zero sales (when store was open)
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with Sales column
        
    Returns:
    --------
    df : pandas DataFrame
        Filtered dataset
    """
    if 'Sales' not in df.columns:
        return df
    
    initial_rows = len(df)
    df = df[df['Sales'] > 0].copy()
    removed_rows = initial_rows - len(df)
    
    print(f"\n💰 Removing zero sales records...")
    print(f"  ✓ Removed {removed_rows:,} zero sales records")
    print(f"  ✓ Remaining: {len(df):,} rows")
    
    return df


def preprocess_pipeline(train_path='data/raw/train.csv', 
                       test_path='data/raw/test.csv',
                       store_path='data/raw/store.csv'):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    train_path, test_path, store_path : str
        Paths to data files
        
    Returns:
    --------
    train_processed, test_processed : pandas DataFrames
        Preprocessed datasets
    """
    print("="*70)
    print("ROSSMANN DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load data
    train, test, store = load_data(train_path, test_path, store_path)
    
    # Handle missing values in store data
    store = handle_missing_values(store)
    
    # Merge with store information
    train = merge_store_data(train, store)
    test = merge_store_data(test, store)
    
    # Filter closed stores (only for training)
    train = filter_closed_stores(train)
    
    # Remove zero sales (optional - only for training)
    train = remove_zero_sales(train)
    
    # Sort by date (important for time series)
    print("\n📅 Sorting by date...")
    train = train.sort_values(['Store', 'Date']).reset_index(drop=True)
    test = test.sort_values(['Store', 'Date']).reset_index(drop=True)
    print("  ✓ Data sorted chronologically")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Shapes:")
    print(f"  Train: {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Test:  {test.shape[0]:,} rows × {test.shape[1]} columns")
    
    return train, test


if __name__ == "__main__":
    # Test the preprocessing pipeline
    train, test = preprocess_pipeline()
    
    # Save processed data
    print("\n💾 Saving preprocessed data...")
    train.to_csv('data/processed/train_preprocessed.csv', index=False)
    test.to_csv('data/processed/test_preprocessed.csv', index=False)
    print("  ✓ Saved to data/processed/")
    
    print("\n✅ All done!")

