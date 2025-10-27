import pandas as pd

print("="*70)
print("VERIFYING PREPROCESSED DATA")
print("="*70)

# Load preprocessed data
train = pd.read_csv('data/processed/train_preprocessed.csv', parse_dates=['Date'])
test = pd.read_csv('data/processed/test_preprocessed.csv', parse_dates=['Date'])

print(f"\n✅ Files loaded successfully!")

print(f"\n📊 TRAIN DATASET")
print(f"  Shape: {train.shape}")
print(f"  Date range: {train['Date'].min()} to {train['Date'].max()}")
print(f"  Unique stores: {train['Store'].nunique()}")
print(f"  Missing values: {train.isnull().sum().sum()}")

print(f"\n📊 TEST DATASET")
print(f"  Shape: {test.shape}")
print(f"  Date range: {test['Date'].min()} to {test['Date'].max()}")
print(f"  Unique stores: {test['Store'].nunique()}")
print(f"  Missing values: {test.isnull().sum().sum()}")

print(f"\n📋 TRAIN COLUMNS ({len(train.columns)}):")
print(train.columns.tolist())

print(f"\n✨ KEY VERIFICATIONS:")
print(f"  ✓ All stores open (Open=1): {(train['Open'] == 1).all()}")
print(f"  ✓ All sales > 0: {(train['Sales'] > 0).all()}")
print(f"  ✓ No missing values in train: {train.isnull().sum().sum() == 0}")
print(f"  ✓ Data sorted by date: {train['Date'].is_monotonic_increasing}")

print("\n✅ Preprocessing verification complete!")

