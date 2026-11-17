"""
Rossmann Store Sales Forecasting - Streamlit Application

Forecasts use the 6-week-horizon LightGBM model saved by `python -m src.evaluate`
(models/lightgbm_horizon_model.pkl). It has no lag/rolling sales features, so it
needs nothing that is unknown on future dates.
"""

import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.data.data_preprocessing import handle_missing_values
from src.evaluate import encode_categoricals
from src.features.feature_engineering import (
    create_promo2_feature,
    extract_date_features,
    transform_competition_distance,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

MISSING_MSG = "Run `python -m src.evaluate` first (it trains the model and writes the features)."


@st.cache_resource
def load_model():
    """Load the 6-week-horizon LightGBM model"""
    try:
        with open(os.path.join(BASE_DIR, 'models', 'lightgbm_horizon_model.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found. {MISSING_MSG}")
        st.stop()


@st.cache_data
def load_store_data():
    """Store metadata with missing values filled exactly as in training"""
    try:
        return handle_missing_values(pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'store.csv')))
    except FileNotFoundError:
        st.error("❌ data/raw/store.csv not found. Download the Kaggle data first.")
        st.stop()


@st.cache_data
def load_historical_data():
    """Open-day sales history plus the per-store averages the model was trained with"""
    try:
        return pd.read_csv(
            os.path.join(BASE_DIR, 'data', 'processed', 'train_features.csv'),
            usecols=['Store', 'Date', 'DayOfWeek', 'Promo', 'Sales', 'Store_AvgSales', 'Store_StdSales'],
            parse_dates=['Date'],
        )
    except FileNotFoundError:
        st.error(f"❌ data/processed/train_features.csv not found. {MISSING_MSG}")
        st.stop()


model_data = load_model()
model = model_data['model']
feature_cols = model_data['feature_cols']
store_info = load_store_data()
historical_data = load_historical_data()
store_stats = historical_data.groupby('Store')[['Store_AvgSales', 'Store_StdSales']].first()
sunday_stores = set(historical_data.loc[historical_data['DayOfWeek'] == 7, 'Store'])

# ============================================================================
# FORECAST
# ============================================================================


def make_forecast(store_id, dates, promo=0):
    """
    Daily sales forecast for one store.

    Assumes the store is open Monday-Saturday with no state or school holiday.
    Sundays are 0 unless the store traded on Sundays in the training data.
    `promo` applies Monday-Friday only; the data has no weekend promos.
    Returns a DataFrame with Date, Open and Predicted_Sales.
    """
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Store': store_id})
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1 = Monday, as in train.csv
    df['Open'] = 1
    df['Promo'] = np.where(df['DayOfWeek'] <= 5, promo, 0)
    df['StateHoliday'] = '0'
    df['SchoolHoliday'] = 0
    df = df.merge(store_info, on='Store', how='left')
    df = transform_competition_distance(create_promo2_feature(extract_date_features(df)))
    df = df.join(store_stats, on='Store')

    X = encode_categoricals(df)[feature_cols]
    preds = np.maximum(model.predict(X, num_iteration=model.best_iteration), 0)
    is_open = (df['DayOfWeek'] != 7) | (store_id in sunday_stores)
    return pd.DataFrame({
        'Date': df['Date'],
        'Open': is_open,
        'Predicted_Sales': np.where(is_open, preds, 0.0),
    })

# ============================================================================
# TITLE AND HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🛒 Rossmann Store Sales Forecasting</h1>',
            unsafe_allow_html=True)
st.markdown("### Predict future sales for Rossmann drugstore chains")
st.caption(
    "Model: LightGBM, 6-week-horizon variant (no sales-history features), 13.9% RMSPE on the "
    "46-day holdout 2015-06-16 to 2015-07-31. Forecasts assume the store is open Monday to "
    "Saturday with no state or school holiday. Horizons beyond 46 days, and dates long after "
    "July 2015, were not validated."
)
st.markdown("---")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.header("📊 Forecast Configuration")

# Store selection
available_stores = sorted(store_info['Store'].unique())
selected_store = st.sidebar.selectbox(
    "Select Store",
    options=available_stores,
    index=0
)

# Get store information
store_data = store_info[store_info['Store'] == selected_store].iloc[0]

# Display store info
st.sidebar.markdown("### 🏪 Store Information")
st.sidebar.write(f"**Type:** {store_data['StoreType']}")
st.sidebar.write(f"**Assortment:** {store_data['Assortment']}")
comp_dist = store_data['CompetitionDistance']
if pd.notna(comp_dist) and comp_dist < 100000:
    st.sidebar.write(f"**Competition Distance:** {comp_dist:.0f}m")
else:
    st.sidebar.write(f"**Competition Distance:** No nearby competition")
promo2 = store_data['PromoInterval'] if store_data['Promo2'] == 1 else "no"
st.sidebar.write(f"**Promo2:** {promo2}")
st.sidebar.write(f"**Open on Sundays:** {'yes' if selected_store in sunday_stores else 'no'}")

st.sidebar.markdown("---")

# Date selection
st.sidebar.markdown("### 📅 Forecast Period")
forecast_start = st.sidebar.date_input(
    "Start Date",
    value=datetime(2015, 8, 1),
    min_value=datetime(2015, 8, 1),
    max_value=datetime(2016, 12, 31)
)

forecast_days = st.sidebar.slider(
    "Number of Days to Forecast",
    min_value=7,
    max_value=90,
    value=30,
    step=7
)

st.sidebar.markdown("---")

# Promotion settings
st.sidebar.markdown("### 🎯 Promotion Settings")
promo_enabled = st.sidebar.checkbox("Promotion Active", value=False,
                                    help="Applies Monday to Friday; the data has no weekend promotions.")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Generate forecast button
if st.button("🚀 Generate Forecast", type="primary", width="stretch"):
    with st.spinner("Generating forecast..."):
        forecast_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq='D')
        st.session_state['forecast'] = make_forecast(selected_store, forecast_dates, int(promo_enabled))
        st.session_state['forecast_store'] = selected_store
        st.session_state['forecast_start'] = forecast_start
    st.success("✅ Forecast generated successfully!")

# Display results if forecast exists
if 'forecast' in st.session_state:

    forecast = st.session_state['forecast']
    forecast_store = st.session_state['forecast_store']
    forecast_dates = pd.DatetimeIndex(forecast['Date'])
    predictions = forecast['Predicted_Sales'].to_numpy()
    open_preds = predictions[forecast['Open'].to_numpy()]

    # ========================================================================
    # METRICS
    # ========================================================================

    st.markdown("## 📈 Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Forecasted Sales", f"{predictions.sum():,.0f}")
    col2.metric("Average per Open Day", f"{open_preds.mean():,.0f}")
    col3.metric("Lowest Open Day", f"{open_preds.min():,.0f}")
    col4.metric("Highest Day", f"{open_preds.max():,.0f}")

    st.markdown("---")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    st.markdown("## 📊 Forecast Visualization")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(forecast_dates, predictions, marker='o', linewidth=2, markersize=4,
            color='#1f77b4', label='Predicted Sales')

    ax.fill_between(forecast_dates,
                     predictions * 0.9,
                     predictions * 1.1,
                     alpha=0.2,
                     color='#1f77b4',
                     label='±10% band (illustrative, not a confidence interval)')

    ax.set_title(f'Sales Forecast for Store {forecast_store}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()
    st.caption("Shaded band: ±10% of the forecast, illustrative only, not a confidence interval. "
               "Days shown as 0 are Sundays for a store that never opened on Sundays in 2013-2015.")

    # ========================================================================
    # HISTORICAL COMPARISON
    # ========================================================================

    st.markdown("## 📉 Historical vs Forecast")

    store_hist = historical_data[historical_data['Store'] == forecast_store].tail(90)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(store_hist['Date'], store_hist['Sales'],
            marker='o', linewidth=2, markersize=3,
            color='green', alpha=0.7, label='Historical Sales (open days)')
    ax.plot(forecast_dates, predictions,
            marker='o', linewidth=2, markersize=4,
            color='#1f77b4', label='Predicted Sales')
    ax.axvline(x=forecast_dates[0], color='red', linestyle='--',
               linewidth=2, label='Forecast Start')
    ax.set_title(f'Historical Sales vs Forecast - Store {forecast_store}',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ========================================================================
    # DATA TABLE
    # ========================================================================

    st.markdown("## 📋 Detailed Forecast Data")

    display_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Day': forecast_dates.day_name(),
        'Predicted_Sales': [f"{x:,.0f}" if is_open else "closed"
                            for x, is_open in zip(predictions, forecast['Open'])],
    })

    st.dataframe(display_df, hide_index=True)

    # ========================================================================
    # DOWNLOAD BUTTON
    # ========================================================================

    st.markdown("## 💾 Export Forecast")

    csv_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Store': forecast_store,
        'Predicted_Sales': predictions.round(2)
    })

    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_df.to_csv(index=False),
        file_name=f"forecast_store_{forecast_store}_{st.session_state['forecast_start']}.csv",
        mime="text/csv",
        width="stretch"
    )

else:
    # Show instructions when no forecast is generated yet
    st.info("👈 Configure your forecast settings in the sidebar and click 'Generate Forecast' to see predictions.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Rossmann Store Sales Forecasting System | Built with Streamlit & LightGBM</p>
        <p>© 2025 | Data from Kaggle Rossmann Competition</p>
    </div>
""", unsafe_allow_html=True)
