This directory stores trained model artifacts. It is ignored by Git (except this file).

`python -m src.evaluate` writes:

- `lightgbm_horizon_model.pkl`: 6-week-horizon model (no sales-history features), used by the Streamlit app
- `lightgbm_best_model.pkl`: 1-day-ahead model (needs the previous days' actual sales)
