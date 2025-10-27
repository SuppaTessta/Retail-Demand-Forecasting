"""
Rossmann Store Sales Forecasting - Streamlit Application
Author: Your Name
Date: October 2025
Python: 3.13.5
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# ENCODING FUNCTION
# ============================================================================

def encode_categorical_features(df):
    """
    Encode categorical features to numeric values for LightGBM
    """
    df = df.copy()
    
    # StateHoliday encoding
    state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    if 'StateHoliday' in df.columns:
        df['StateHoliday'] = df['StateHoliday'].astype(str).map(state_holiday_map).fillna(0).astype(int)
    
    # StoreType encoding
    store_type_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    if 'StoreType' in df.columns:
        df['StoreType'] = df['StoreType'].astype(str).map(store_type_map).fillna(0).astype(int)
    
    # Assortment encoding
    assortment_map = {'a': 0, 'b': 1, 'c': 2}
    if 'Assortment' in df.columns:
        df['Assortment'] = df['Assortment'].astype(str).map(assortment_map).fillna(0).astype(int)
    
    # PromoInterval encoding (handle Sep vs Sept and nulls)
    if 'PromoInterval' in df.columns:
        promo_interval_map = {
            '': 0, 'none': 0, 'nan': 0, '0': 0, 'None': 0,
            'Jan,Apr,Jul,Oct': 1,
            'Feb,May,Aug,Nov': 2,
            'Mar,Jun,Sep,Dec': 3,
            'Mar,Jun,Sept,Dec': 3  # fallback if present
        }
        df['PromoInterval'] = (
            df['PromoInterval']
            .astype(str)
            .str.strip()
            .replace({'None': 'none'})
            .map(promo_interval_map)
            .fillna(0)
            .astype(int)
        )

    # Ensure no object dtypes remain
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes

    return df

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    try:
        model_path = os.path.join(BASE_DIR, 'models', 'lightgbm_best_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_data
def load_store_data():
    """Load store information"""
    try:
        store_path = os.path.join(BASE_DIR, 'data', 'raw', 'store.csv')
        store = pd.read_csv(store_path)
        return store
    except FileNotFoundError:
        st.error("❌ Store data not found!")
        st.stop()

@st.cache_data
def load_historical_data():
    """Load historical sales data for visualization"""
    try:
        hist_path = os.path.join(BASE_DIR, 'data', 'processed', 'train_features.csv')
        data = pd.read_csv(hist_path, parse_dates=['Date'])
        return data
    except FileNotFoundError:
        st.warning("⚠️ Historical data not found. Some features may be unavailable.")
        return None

# Load resources
model_data = load_model()
model = model_data['model']
feature_cols = model_data['feature_cols']
store_info = load_store_data()
historical_data = load_historical_data()

# ============================================================================
# TITLE AND HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🛒 Rossmann Store Sales Forecasting</h1>', 
            unsafe_allow_html=True)
st.markdown("### Predict future sales for Rossmann drugstore chains")
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
promo_enabled = st.sidebar.checkbox("Promotion Active", value=False)
promo2_enabled = st.sidebar.checkbox("Promo2 Active", value=False)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_forecast_features(store_id, dates, promo=0, promo2=0):
    """
    Create feature dataframe for forecasting
    
    Parameters:
    -----------
    store_id : int
        Store ID
    dates : list
        List of dates to forecast
    promo : int
        Promotion flag (0 or 1)
    promo2 : int
        Promo2 flag (0 or 1)
    """
    # Get store information
    store_row = store_info[store_info['Store'] == store_id].iloc[0]
    
    # Create base dataframe
    forecast_df = pd.DataFrame()
    forecast_df['Date'] = pd.to_datetime(dates)
    forecast_df['Store'] = store_id
    
    # Add date features
    forecast_df['Year'] = forecast_df['Date'].dt.year
    forecast_df['Month'] = forecast_df['Date'].dt.month
    forecast_df['Day'] = forecast_df['Date'].dt.day
    forecast_df['DayOfWeek'] = forecast_df['Date'].dt.dayofweek + 1  # 1-7
    forecast_df['WeekOfYear'] = forecast_df['Date'].dt.isocalendar().week
    forecast_df['Quarter'] = forecast_df['Date'].dt.quarter
    forecast_df['DayOfMonth'] = forecast_df['Date'].dt.day
    forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear
    forecast_df['IsWeekend'] = (forecast_df['DayOfWeek'] >= 6).astype(int)
    forecast_df['IsMonthStart'] = forecast_df['Date'].dt.is_month_start.astype(int)
    forecast_df['IsMonthEnd'] = forecast_df['Date'].dt.is_month_end.astype(int)
    
    # Add store features (keep as original type first)
    forecast_df['StoreType'] = store_row['StoreType']
    forecast_df['Assortment'] = store_row['Assortment']
    forecast_df['CompetitionDistance'] = store_row['CompetitionDistance']
    forecast_df['CompetitionOpenSinceMonth'] = store_row['CompetitionOpenSinceMonth']
    forecast_df['CompetitionOpenSinceYear'] = store_row['CompetitionOpenSinceYear']
    forecast_df['Promo2'] = store_row['Promo2']
    forecast_df['Promo2SinceWeek'] = store_row['Promo2SinceWeek']
    forecast_df['Promo2SinceYear'] = store_row['Promo2SinceYear']
    forecast_df['PromoInterval'] = store_row['PromoInterval']
    
    # Handle NaN values
    forecast_df['CompetitionDistance'] = forecast_df['CompetitionDistance'].fillna(100000)
    forecast_df['CompetitionOpenSinceMonth'] = forecast_df['CompetitionOpenSinceMonth'].fillna(0)
    forecast_df['CompetitionOpenSinceYear'] = forecast_df['CompetitionOpenSinceYear'].fillna(1900)
    forecast_df['Promo2SinceWeek'] = forecast_df['Promo2SinceWeek'].fillna(0)
    forecast_df['Promo2SinceYear'] = forecast_df['Promo2SinceYear'].fillna(1900)
    forecast_df['PromoInterval'] = forecast_df['PromoInterval'].fillna('None')
    
    # Add promotion flags
    forecast_df['Promo'] = promo
    forecast_df['Open'] = 1
    forecast_df['SchoolHoliday'] = 0
    forecast_df['StateHoliday'] = '0'
    
    # Competition features
    forecast_df['CompDistanceTransformed'] = 1 / np.sqrt(forecast_df['CompetitionDistance'])
    forecast_df['CompetitionOpenMonths'] = 12 * (forecast_df['Year'] - forecast_df['CompetitionOpenSinceYear']) + \
                                            (forecast_df['Month'] - forecast_df['CompetitionOpenSinceMonth'])
    forecast_df['CompetitionOpenMonths'] = forecast_df['CompetitionOpenMonths'].clip(lower=0)
    forecast_df['HasCompetition'] = (forecast_df['CompetitionDistance'] < 100000).astype(int)
    
    # Promo2 active
    forecast_df['Promo2Active'] = promo2
    
    # For missing lag/rolling features, use store averages from historical data
    if historical_data is not None:
        store_hist = historical_data[historical_data['Store'] == store_id]
        
        if len(store_hist) > 0:
            # Use recent averages
            recent_sales = store_hist['Sales'].tail(30).mean()
            
            # Fill lag features with recent average
            for lag in [7, 14, 30]:
                forecast_df[f'Sales_Lag_{lag}'] = recent_sales
            
            # Fill rolling features
            for window in [7, 14, 30]:
                forecast_df[f'Sales_RollingMean_{window}'] = recent_sales
                forecast_df[f'Sales_RollingStd_{window}'] = store_hist['Sales'].tail(30).std()
                forecast_df[f'Sales_RollingMin_{window}'] = store_hist['Sales'].tail(window).min()
                forecast_df[f'Sales_RollingMax_{window}'] = store_hist['Sales'].tail(window).max()
            
            # Fill EWM features
            for span in [7, 14, 30]:
                forecast_df[f'Sales_EWM_{span}'] = recent_sales
            
            # Store average features
            forecast_df['Store_AvgSales'] = store_hist['Sales'].mean()
            forecast_df['Store_StdSales'] = store_hist['Sales'].std()
        else:
            # Use defaults
            default_sales = 5000
            for lag in [7, 14, 30]:
                forecast_df[f'Sales_Lag_{lag}'] = default_sales
            for window in [7, 14, 30]:
                forecast_df[f'Sales_RollingMean_{window}'] = default_sales
                forecast_df[f'Sales_RollingStd_{window}'] = 500
                forecast_df[f'Sales_RollingMin_{window}'] = default_sales * 0.8
                forecast_df[f'Sales_RollingMax_{window}'] = default_sales * 1.2
            for span in [7, 14, 30]:
                forecast_df[f'Sales_EWM_{span}'] = default_sales
            forecast_df['Store_AvgSales'] = default_sales
            forecast_df['Store_StdSales'] = 500
    else:
        # Use defaults if no historical data
        default_sales = 5000
        for lag in [7, 14, 30]:
            forecast_df[f'Sales_Lag_{lag}'] = default_sales
        for window in [7, 14, 30]:
            forecast_df[f'Sales_RollingMean_{window}'] = default_sales
            forecast_df[f'Sales_RollingStd_{window}'] = 500
            forecast_df[f'Sales_RollingMin_{window}'] = default_sales * 0.8
            forecast_df[f'Sales_RollingMax_{window}'] = default_sales * 1.2
        for span in [7, 14, 30]:
            forecast_df[f'Sales_EWM_{span}'] = default_sales
        forecast_df['Store_AvgSales'] = default_sales
        forecast_df['Store_StdSales'] = 500
    
    return forecast_df

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Generate forecast button
if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
    
    with st.spinner("Generating forecast..."):
        
        try:
            # Create date range
            forecast_dates = pd.date_range(
                start=forecast_start,
                periods=forecast_days,
                freq='D'
            )
            
            # Create features
            forecast_df = create_forecast_features(
                store_id=selected_store,
                dates=forecast_dates,
                promo=1 if promo_enabled else 0,
                promo2=1 if promo2_enabled else 0
            )
            
            # CRITICAL: Encode categorical features BEFORE prediction
            forecast_df = encode_categorical_features(forecast_df)

            # Align to model's feature columns
            missing_cols = [c for c in feature_cols if c not in forecast_df.columns]
            for c in missing_cols:
                forecast_df[c] = 0
            X_forecast = forecast_df[feature_cols]

            # Safety: ensure no object dtypes
            for col in X_forecast.select_dtypes(include='object').columns:
                X_forecast[col] = X_forecast[col].astype('category').cat.codes

            # Make predictions with best_iteration fallback
            num_iter = getattr(model, 'best_iteration', None)
            if num_iter:
                predictions = model.predict(X_forecast, num_iteration=num_iter)
            else:
                predictions = model.predict(X_forecast)
            
            # Ensure no negative predictions
            predictions = np.maximum(predictions, 0)
            
            forecast_df['Predicted_Sales'] = predictions
            
            # Store in session state
            st.session_state['forecast_df'] = forecast_df
            st.session_state['predictions'] = predictions
            st.session_state['forecast_dates'] = forecast_dates
            
            st.success("✅ Forecast generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")
            st.stop()

# Display results if forecast exists
if 'forecast_df' in st.session_state:
    
    forecast_df = st.session_state['forecast_df']
    predictions = st.session_state['predictions']
    forecast_dates = st.session_state['forecast_dates']
    
    # ========================================================================
    # METRICS
    # ========================================================================
    
    st.markdown("## 📈 Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Forecasted Sales",
            value=f"${predictions.sum():,.0f}"
        )
    
    with col2:
        st.metric(
            label="Average Daily Sales",
            value=f"${predictions.mean():,.0f}"
        )
    
    with col3:
        st.metric(
            label="Minimum Sales",
            value=f"${predictions.min():,.0f}"
        )
    
    with col4:
        st.metric(
            label="Maximum Sales",
            value=f"${predictions.max():,.0f}"
        )
    
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
                     label='Confidence Interval (±10%)')
    
    ax.set_title(f'Sales Forecast for Store {selected_store}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Predicted Sales ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()
    
    # ========================================================================
    # HISTORICAL COMPARISON (if available)
    # ========================================================================
    
    if historical_data is not None:
        st.markdown("## 📉 Historical vs Forecast")
        
        store_hist = historical_data[historical_data['Store'] == selected_store].tail(90)
        
        if len(store_hist) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot historical
            ax.plot(store_hist['Date'], store_hist['Sales'], 
                    marker='o', linewidth=2, markersize=3, 
                    color='green', alpha=0.7, label='Historical Sales')
            
            # Plot forecast
            ax.plot(forecast_dates, predictions, 
                    marker='o', linewidth=2, markersize=4,
                    color='#1f77b4', label='Predicted Sales')
            
            ax.axvline(x=forecast_dates[0], color='red', linestyle='--', 
                       linewidth=2, label='Forecast Start')
            
            ax.set_title(f'Historical Sales vs Forecast - Store {selected_store}', 
                         fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales ($)', fontsize=12)
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
        'Predicted_Sales': [f"${x:,.2f}" for x in predictions]
    })
    
    st.dataframe(
        display_df, 
        use_container_width=True,
        hide_index=True
    )
    
    # ========================================================================
    # DOWNLOAD BUTTON
    # ========================================================================
    
    st.markdown("## 💾 Export Forecast")
    
    csv_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Store': selected_store,
        'Predicted_Sales': predictions
    })
    csv = csv_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"forecast_store_{selected_store}_{forecast_start}.csv",
        mime="text/csv",
        use_container_width=True
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


