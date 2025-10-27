import pandas as pd

print("="*70)
print("VERIFYING FEATURE ENGINEERING")
print("="*70)

# Load feature-engineered data
train = pd.read_csv('data/processed/train_features.csv', parse_dates=['Date'])
test = pd.read_csv('data/processed/test_features.csv', parse_dates=['Date'])

print(f"\n✅ Files loaded successfully!")

print(f"\n📊 TRAIN DATASET")
print(f"  Shape: {train.shape}")
print(f"  Columns: {len(train.columns)}")
print(f"  Date range: {train['Date'].min()} to {train['Date'].max()}")
print(f"  Missing values: {train.isnull().sum().sum()}")

print(f"\n📊 TEST DATASET")
print(f"  Shape: {test.shape}")
print(f"  Columns: {len(test.columns)}")
print(f"  Date range: {test['Date'].min()} to {test['Date'].max()}")
print(f"  Missing values: {test.isnull().sum().sum()}")

print(f"\n📋 NEW FEATURES CREATED:")
original_cols = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                 'Promo2SinceYear', 'PromoInterval']

new_features = [col for col in train.columns if col not in original_cols]
print(f"\n  Total new features: {len(new_features)}")
print(f"\n  Feature categories:")

# Categorize features
date_features = [f for f in new_features if any(x in f for x in ['Year', 'Month', 'Day', 'Week', 'Quarter', 'Is'])]
promo_features = [f for f in new_features if 'Promo2' in f]
comp_features = [f for f in new_features if 'Comp' in f or 'Competition' in f]
store_features = [f for f in new_features if 'Store' in f]
lag_features = [f for f in new_features if 'Lag' in f]
rolling_features = [f for f in new_features if 'Rolling' in f]
ewm_features = [f for f in new_features if 'EWM' in f]

print(f"    - Date features: {len(date_features)}")
print(f"    - Promo2 features: {len(promo_features)}")
print(f"    - Competition features: {len(comp_features)}")
print(f"    - Store features: {len(store_features)}")
print(f"    - Lag features: {len(lag_features)}")
print(f"    - Rolling features: {len(rolling_features)}")
print(f"    - EWM features: {len(ewm_features)}")

print(f"\n✨ SAMPLE OF NEW FEATURES:")
sample_features = new_features[:15]
for feat in sample_features:
    print(f"  - {feat}")

print(f"\n📈 FEATURE VALUE RANGES (First 5 new features):")
for feat in new_features[:5]:
    if train[feat].dtype in ['float64', 'int64']:
        print(f"  {feat}:")
        print(f"    Min: {train[feat].min():.2f}, Max: {train[feat].max():.2f}, Mean: {train[feat].mean():.2f}")

print(f"\n✅ Feature engineering verification complete!")
print(f"\n🚀 Ready for modeling with {len(train.columns)} total features!")
