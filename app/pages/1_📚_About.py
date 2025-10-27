import streamlit as st

st.title("📚 About This Project")

st.markdown("""
## Project Overview

This is a **professional retail demand forecasting system** built for Rossmann drugstore chains.

### Features

- ✅ Forecast sales for 1,115 stores
- ✅ 30+ engineered features
- ✅ LightGBM machine learning model
- ✅ Interactive web dashboard
- ✅ CSV export functionality

### Model Performance

- **RMSPE**: 0.10-0.13 (Competition-grade)
- **Features**: 52 total features
- **Training Data**: 800K+ records

### Technology Stack

- **Frontend**: Streamlit
- **ML Model**: LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

### Dataset

- **Source**: Kaggle Rossmann Store Sales Competition
- **Period**: 2013-2015
- **Stores**: 1,115 across Germany
- **Records**: 1M+ transactions

---

**Author**: Sanchay Maity  
**Date**: October 2025  
**Python**: 3.13.5
""")
