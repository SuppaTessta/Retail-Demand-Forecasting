"""
Leak-free validation run for the Rossmann LightGBM model.

Holdout: 2015-06-16 .. 2015-07-31 (last 46 days of train.csv, never used for
fitting or early stopping). Early stopping uses the 6 weeks before it.

Two LightGBM variants:
  * one_day_ahead - all features. Lag/rolling/EWM features use the actual sales
    of the preceding open days, so they assume yesterday's sales are known
    (1-day-ahead forecasting).
  * six_week_horizon - no Sales lag/rolling/EWM features at all (the shortest
    existing lag is 30 open days, below the 46-day holdout); only calendar,
    promo, store, competition and store averages from the training period.
    This matches forecasting the whole holdout from 2015-06-15.

Writes results/metrics.json, data/processed/train_features.csv,
models/lightgbm_best_model.pkl (1-day-ahead) and
models/lightgbm_horizon_model.pkl (6-week horizon, used by the app).

Run from the repo root:  python -m src.evaluate
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_preprocessing import preprocess_pipeline
from src.evaluation.metrics import calculate_metrics
from src.features.feature_engineering import feature_engineering_pipeline
from src.models.ml_models import LightGBMForecaster

ROOT = Path(__file__).resolve().parents[1]
VALID_START = pd.Timestamp('2015-06-16')
EARLY_STOP_START = VALID_START - pd.Timedelta(weeks=6)
HISTORY_PREFIXES = ('Sales_Lag_', 'Sales_Rolling', 'Sales_EWM_')


def encode_categoricals(df):
    """Fixed mappings so codes are identical across splits."""
    df = df.copy()
    maps = {
        'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3},
        'StoreType': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
        'Assortment': {'a': 0, 'b': 1, 'c': 2},
        'PromoInterval': {'None': 0, 'Jan,Apr,Jul,Oct': 1,
                          'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3},
    }
    for col, m in maps.items():
        df[col] = df[col].astype(str).map(m).fillna(0).astype(int)
    return df


def seasonal_naive(valid, history, lag_days, fallback):
    """Sales `lag_days` (per-row array or int) earlier; store fallback if missing."""
    key = pd.DataFrame({'Store': valid['Store'].values,
                        'Date': valid['Date'].values - pd.to_timedelta(lag_days, unit='D')})
    pred = key.merge(history[['Store', 'Date', 'Sales']], on=['Store', 'Date'], how='left')['Sales']
    return np.where(pred.isna(), valid['Store'].map(fallback).values, pred.values)


def rounded(metrics):
    return {k: round(float(v), 4) for k, v in metrics.items()}


def train_variant(fit, early, valid, drop_cols):
    model = LightGBMForecaster()
    t0 = time.perf_counter()
    model.fit(fit.drop(columns=drop_cols), early.drop(columns=drop_cols))
    seconds = time.perf_counter() - t0
    preds = np.maximum(model.predict(valid.drop(columns=drop_cols)), 0)
    top10 = model.get_feature_importance(top_n=10)
    return model, preds, {
        'metrics': rounded(calculate_metrics(valid['Sales'].values, preds)),
        'best_iteration': int(model.model.best_iteration),
        'n_features': len(model.feature_cols),
        'train_seconds': round(seconds, 1),
        'top10_features_gain': [
            {'feature': f, 'gain_share': round(float(g) / float(model.model.feature_importance('gain').sum()), 4)}
            for f, g in zip(top10['feature'], top10['importance'])
        ],
    }


def main():
    t_start = time.perf_counter()
    raw = ROOT / 'data' / 'raw'
    train, _ = preprocess_pipeline(raw / 'train.csv', raw / 'test.csv', raw / 'store.csv')
    data = feature_engineering_pipeline(train, is_train=True, stats_cutoff=VALID_START)
    # Leak guard: if the rolling window included the same day, max >= target on every row
    assert not (data['Sales_RollingMax_7'] >= data['Sales']).all(), 'rolling features include same-day sales'
    (ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    data.to_csv(ROOT / 'data' / 'processed' / 'train_features.csv', index=False)  # used by the app

    data = encode_categoricals(data)
    fit = data[data['Date'] < EARLY_STOP_START].reset_index(drop=True)
    early = data[(data['Date'] >= EARLY_STOP_START) & (data['Date'] < VALID_START)].reset_index(drop=True)
    valid = data[data['Date'] >= VALID_START].reset_index(drop=True)
    y = valid['Sales'].values

    history_cols = [c for c in data.columns if c.startswith(HISTORY_PREFIXES)]
    full_model, full_preds, one_day = train_variant(fit, early, valid, [])
    horizon_model, _, six_week = train_variant(fit, early, valid, history_cols)
    full_model.save_model(ROOT / 'models' / 'lightgbm_best_model.pkl')
    horizon_model.save_model(ROOT / 'models' / 'lightgbm_horizon_model.pkl')  # used by the app

    # Baselines. Fallback (store closed on the lag date) = store mean over training dates.
    train_period = data[data['Date'] < VALID_START]
    store_mean = train_period.groupby('Store')['Sales'].mean()
    lag7 = seasonal_naive(valid, data, 7, store_mean)  # actual sales 7 days earlier
    weeks_ahead = np.ceil((valid['Date'] - (VALID_START - pd.Timedelta(days=1))).dt.days / 7).astype(int)
    last_week = seasonal_naive(valid, train_period, (weeks_ahead * 7).values, store_mean)

    results = {
        'dataset': 'Kaggle Rossmann Store Sales train.csv (open days, Sales > 0)',
        'splits': {
            'fit': {'rows': len(fit), 'from': str(fit['Date'].min().date()), 'to': str(fit['Date'].max().date())},
            'early_stopping': {'rows': len(early), 'from': str(early['Date'].min().date()), 'to': str(early['Date'].max().date())},
            'validation': {'rows': len(valid), 'from': str(valid['Date'].min().date()), 'to': str(valid['Date'].max().date()),
                           'stores': int(valid['Store'].nunique())},
        },
        'metric_notes': 'RMSPE = Kaggle metric (zero-sales days excluded); RMSE/MAE in sales units; MAPE in %.',
        'lightgbm_one_day_ahead': one_day,
        'lightgbm_six_week_horizon': six_week,
        'baseline_seasonal_naive_lag7': {
            'description': 'Actual sales 7 days earlier (needs last week actuals, like the 1-day-ahead model)',
            'metrics': rounded(calculate_metrics(y, lag7)),
        },
        'baseline_seasonal_naive_last_train_week': {
            'description': 'Same weekday from the last training week, repeated for all 46 days (6-week horizon)',
            'metrics': rounded(calculate_metrics(y, last_week)),
        },
        'total_runtime_seconds': round(time.perf_counter() - t_start, 1),
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results, valid, full_preds


if __name__ == '__main__':
    main()
