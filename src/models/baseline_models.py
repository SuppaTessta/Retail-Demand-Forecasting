"""
Baseline Models for Rossmann Store Sales
"""

import pandas as pd
import numpy as np

class SeasonalNaiveForecaster:
    """
    Seasonal Naive forecasting - uses sales from 7 days ago
    Best baseline for weekly seasonal data
    """
    
    def __init__(self, seasonality=7):
        """
        Parameters:
        -----------
        seasonality : int
            Number of periods for seasonality (7 for weekly)
        """
        self.seasonality = seasonality
        self.name = f"Seasonal_Naive_{seasonality}"
        
    def fit(self, train_data):
        """
        Store training data
        
        Parameters:
        -----------
        train_data : pandas DataFrame
            Training data with Store, Date, Sales columns
        """
        self.train_data = train_data.copy()
        self.train_data = self.train_data.sort_values(['Store', 'Date'])
        print(f"✓ Fitted {self.name} on {len(self.train_data)} records")
        return self
    
    def predict(self, valid_data):
        """
        Predict using seasonal naive approach
        
        Parameters:
        -----------
        valid_data : pandas DataFrame
            Validation data with Store, Date columns
            
        Returns:
        --------
        predictions : pandas Series
            Predicted sales
        """
        # Sales from `seasonality` days earlier (vectorised lookup);
        # fall back to the store's mean over the fitted data
        lag = valid_data[['Store', 'Date']].copy()
        lag['Date'] = lag['Date'] - pd.Timedelta(days=self.seasonality)
        hist = self.train_data[['Store', 'Date', 'Sales']]
        pred = lag.merge(hist, on=['Store', 'Date'], how='left')['Sales'].to_numpy()
        store_avg = valid_data['Store'].map(hist.groupby('Store')['Sales'].mean()).to_numpy()
        return pd.Series(np.where(np.isnan(pred), store_avg, pred), index=valid_data.index)


class MovingAverageForecaster:
    """
    Moving Average forecasting
    """
    
    def __init__(self, window=7):
        """
        Parameters:
        -----------
        window : int
            Window size for moving average
        """
        self.window = window
        self.name = f"Moving_Average_{window}"
        
    def fit(self, train_data):
        """Store training data"""
        self.train_data = train_data.copy()
        self.train_data = self.train_data.sort_values(['Store', 'Date'])
        print(f"✓ Fitted {self.name} on {len(self.train_data)} records")
        return self
    
    def predict(self, valid_data):
        """Predict using moving average"""
        predictions = []
        
        for idx, row in valid_data.iterrows():
            store = row['Store']
            date = row['Date']
            
            # Get last N days of sales
            hist = self.train_data[
                (self.train_data['Store'] == store) & 
                (self.train_data['Date'] < date)
            ].tail(self.window)
            
            if len(hist) > 0:
                pred = hist['Sales'].mean()
            else:
                # Fallback: store average
                pred = self.train_data[
                    self.train_data['Store'] == store
                ]['Sales'].mean()
            
            predictions.append(pred)
        
        return pd.Series(predictions, index=valid_data.index)


if __name__ == "__main__":
    # Test baseline models
    print("="*70)
    print("TESTING BASELINE MODELS")
    print("="*70)
    
    # Load data
    data = pd.read_csv('data/processed/train_features.csv', parse_dates=['Date'])
    
    # Split
    split_date = '2015-06-16'
    train = data[data['Date'] < split_date].reset_index(drop=True)
    valid = data[data['Date'] >= split_date].reset_index(drop=True)
    
    print(f"\nTrain: {train.shape}")
    print(f"Valid: {valid.shape}")
    
    # Test Seasonal Naive
    print("\n" + "-"*70)
    print("SEASONAL NAIVE MODEL")
    print("-"*70)
    
    model = SeasonalNaiveForecaster(seasonality=7)
    model.fit(train)
    
    # Predict on sample
    sample = valid.head(1000)
    preds = model.predict(sample)
    
    print(f"\nPredictions generated: {len(preds)}")
    print(f"Sample predictions: {preds.head()}")
