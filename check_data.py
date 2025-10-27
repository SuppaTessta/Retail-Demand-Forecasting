import pandas as pd

print("Checking Rossmann Dataset...")
print("="*70)

# Load datasets
try:
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    store = pd.read_csv('data/raw/store.csv')
    
    print("\n✅ All files loaded successfully!")
    print("\nDataset Information:")
    print(f"  Train:  {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Test:   {test.shape[0]:,} rows × {test.shape[1]} columns")
    print(f"  Store:  {store.shape[0]:,} rows × {store.shape[1]} columns")
    
    print("\nTrain columns:")
    print(f"  {train.columns.tolist()}")
    
    print("\nDate range:")
    print(f"  Start: {train['Date'].min()}")
    print(f"  End:   {train['Date'].max()}")
    
    print("\n🎉 Dataset is ready to use!")
    
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("Make sure you extracted the files to data/raw/")
except Exception as e:
    print(f"\n❌ Error: {e}")
