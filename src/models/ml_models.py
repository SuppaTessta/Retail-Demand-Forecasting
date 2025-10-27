"""
Machine Learning Models for Rossmann Store Sales
Author: Your Name
Date: October 2025
Python: 3.13.5
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

class LightGBMForecaster:
    """
    LightGBM model for demand forecasting
    Best for tabular time series data
    """
    
    def __init__(self, params=None):
        """
        Parameters:
        -----------
        params : dict
            LightGBM parameters
        """
        if params is None:
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'force_row_wise': True
            }
        else:
            self.params = params
        
        self.model = None
        self.feature_cols = None
        self.name = "LightGBM"
        
    def fit(self, train_data, valid_data=None, categorical_features=None, num_iterations=2000):
        """
        Train LightGBM model
        
        Parameters:
        -----------
        train_data : pandas DataFrame
            Training data
        valid_data : pandas DataFrame
            Validation data (optional)
        categorical_features : list
            List of categorical feature names
        num_iterations : int
            Maximum boosting rounds
        """
        print("="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['Sales', 'Date', 'Customers']
        self.feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        # Handle categorical features
        if categorical_features is None:
            categorical_features = ['Store', 'DayOfWeek', 'StoreType', 'Assortment', 
                                   'StateHoliday', 'SchoolHoliday', 'Promo', 'Promo2']
        
        # Ensure categorical features are in feature_cols
        categorical_features = [f for f in categorical_features if f in self.feature_cols]
        
        print(f"\n📊 Training Info:")
        print(f"  Total features: {len(self.feature_cols)}")
        print(f"  Categorical features: {len(categorical_features)}")
        print(f"  Training samples: {len(train_data):,}")
        
        # **FIX: Convert object dtypes to category codes**
        train_data_processed = train_data.copy()
        if valid_data is not None:
            valid_data_processed = valid_data.copy()
        
        # Convert categorical object columns to numeric codes
        for col in self.feature_cols:
            if train_data_processed[col].dtype == 'object':
                print(f"  Converting {col} from object to category codes")
                train_data_processed[col] = train_data_processed[col].astype('category').cat.codes
                if valid_data is not None:
                    valid_data_processed[col] = valid_data_processed[col].astype('category').cat.codes
        
        # Prepare data
        X_train = train_data_processed[self.feature_cols]
        y_train = train_data_processed['Sales']
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(
            X_train, 
            y_train,
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        
        valid_sets = [lgb_train]
        valid_names = ['train']
        
        if valid_data is not None:
            print(f"  Validation samples: {len(valid_data):,}")
            X_valid = valid_data_processed[self.feature_cols]
            y_valid = valid_data_processed['Sales']
            
            lgb_valid = lgb.Dataset(
                X_valid,
                y_valid,
                reference=lgb_train,
                categorical_feature=categorical_features,
                free_raw_data=False
            )
            valid_sets.append(lgb_valid)
            valid_names.append('valid')
        
        # Train model
        print(f"\n🚀 Training with early stopping...")
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=num_iterations,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        print(f"\n✓ Training complete!")
        print(f"  Best iteration: {self.model.best_iteration}")
        print(f"  Best score: {self.model.best_score}")
        
        return self
    
    def predict(self, data):
        """
        Generate predictions
        
        Parameters:
        -----------
        data : pandas DataFrame
            Data to predict on
            
        Returns:
        --------
        predictions : numpy array
            Predicted sales
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # **FIX: Convert object dtypes to category codes**
        data_processed = data.copy()
        for col in self.feature_cols:
            if data_processed[col].dtype == 'object':
                data_processed[col] = data_processed[col].astype('category').cat.codes
    
        X = data_processed[self.feature_cols]
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return predictions
    
    def get_feature_importance(self, importance_type='gain', top_n=20):
        """
        Get feature importance
        
        Parameters:
        -----------
        importance_type : str
            'gain' or 'split'
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        importance_df : pandas DataFrame
            Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'params': self.params
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_cols = data['feature_cols']
            self.params = data['params']
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # Test LightGBM model
    print("="*70)
    print("TESTING LIGHTGBM MODEL")
    print("="*70)
    
    # Load data
    data = pd.read_csv('data/processed/train_features.csv', parse_dates=['Date'])
    
    # Split
    split_date = '2015-06-16'
    train = data[data['Date'] < split_date].reset_index(drop=True)
    valid = data[data['Date'] >= split_date].reset_index(drop=True)
    
    # Sample for quick test (use full data in production)
    train_sample = train.sample(n=min(100000, len(train)), random_state=42)
    valid_sample = valid.sample(n=min(10000, len(valid)), random_state=42)
    
    # Train model
    model = LightGBMForecaster()
    model.fit(train_sample, valid_sample, num_iterations=100)
    
    # Make predictions
    preds = model.predict(valid_sample)
    print(f"\n✓ Predictions generated: {len(preds)}")
