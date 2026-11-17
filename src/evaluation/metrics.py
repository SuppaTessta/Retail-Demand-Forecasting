"""
Evaluation Metrics for Rossmann Store Sales
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error
    This is the main metric for Rossmann competition
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    rmspe : float
        RMSPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))


def calculate_metrics(y_true, y_pred):
    """
    Calculate multiple forecasting metrics
    
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = calculate_rmspe(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'RMSPE': rmspe,
        'MAPE': mape
    }


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([100, 200, 150, 300, 250])
    y_pred = np.array([110, 190, 160, 290, 240])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
