import streamlit as st

st.title("📚 About This Project")

st.markdown("""
## Project Overview

A daily sales forecasting model and dashboard for Rossmann drugstores, built on the Kaggle Rossmann Store Sales data.

### Features

- ✅ Forecast sales for any of the 1,115 stores, 7 to 90 days ahead
- ✅ 34 engineered features (calendar, promo, competition, store averages, lag/rolling sales history)
- ✅ LightGBM model, validated against seasonal-naive baselines
- ✅ Interactive web dashboard with CSV export

### Model Performance

Validation on a 46-day holdout (2015-06-16 to 2015-07-31, 44,738 store-days):

- **Dashboard model (6-week horizon)**: 13.9% RMSPE. 31 features, no sales history, so every input is known in advance.
- **1-day-ahead model**: 12.7% RMSPE. 49 features, including the previous days' actual sales, so it only applies when yesterday's sales are known. Not used by the dashboard.
- **Baseline**: 29.0% RMSPE (seasonal naive, same weekday of the last training week)
- **Training data**: 766K store-days (729K fit + 37K early stopping)
- The ±10% band on the forecast chart is illustrative, not a confidence interval

### Limitations

- Forecasts assume the store is open Monday to Saturday with no state or school holiday; Sundays are 0 for stores that never opened on Sundays.
- The model was trained on data up to 2015-05-04 and validated up to 46 days ahead. Longer horizons are not validated.

### Technology Stack

- **Frontend**: Streamlit
- **ML Model**: LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib

### Dataset

- **Source**: Kaggle Rossmann Store Sales Competition
- **Period**: 2013-01-01 to 2015-07-31
- **Stores**: 1,115
- **Records**: 1,017,209 store-days

---

**Author**: Sanchay Maity
**Date**: October 2025
**Python**: 3.12
""")
