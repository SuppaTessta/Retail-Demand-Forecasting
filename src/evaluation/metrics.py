"""
Evaluation Metrics for Rossmann Store Sales
Author: Your Name
Date: October 2025
Python: 3.13.5
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_predictions(y_true, y_pred, title='Actual vs Predicted', sample_size=1000):
    """
    Plot actual vs predicted values
    """
    # Sample for visualization
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(y_true_sample, y_pred_sample, alpha=0.5, s=10)
    axes[0].plot([y_true_sample.min(), y_true_sample.max()], 
                 [y_true_sample.min(), y_true_sample.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Sales')
    axes[0].set_ylabel('Predicted Sales')
    axes[0].set_title(f'{title} - Scatter Plot')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residuals
    residuals = y_pred_sample - y_true_sample
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (Predicted - Actual)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(results_dict):
    """
    Compare multiple models
    
    Parameters:
    -----------
    results_dict : dict
        {model_name: {'y_true': ..., 'y_pred': ...}}
    """
    comparison = []
    
    for model_name, data in results_dict.items():
        metrics = calculate_metrics(data['y_true'], data['y_pred'])
        metrics['Model'] = model_name
        comparison.append(metrics)
    
    df = pd.DataFrame(comparison)
    df = df[['Model', 'RMSPE', 'RMSE', 'MAE', 'MAPE']]
    df = df.sort_values('RMSPE')
    
    return df


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([100, 200, 150, 300, 250])
    y_pred = np.array([110, 190, 160, 290, 240])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
