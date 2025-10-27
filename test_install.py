import sys

def test_imports():
    print("Testing dependency installations...\n")
    print("="*70)
    
    dependencies = {
        "Core Data Science": ["numpy", "pandas", "scipy"],
        "Machine Learning": ["sklearn", "lightgbm", "xgboost", "catboost"],
        "Time Series": ["prophet", "statsmodels"],
        "Visualization": ["matplotlib", "seaborn", "plotly"],
        "Web Apps": ["streamlit"],
        "Utilities": ["yaml", "dotenv", "tqdm"],
        "Development": ["jupyter", "IPython"]
    }
    
    failed = []
    passed = 0
    
    for category, packages in dependencies.items():
        print(f"\n{category}:")
        for package in packages:
            try:
                # Handle special import names
                if package == "sklearn":
                    __import__("sklearn")
                elif package == "dotenv":
                    __import__("dotenv")
                elif package == "yaml":
                    __import__("yaml")
                else:
                    __import__(package)
                
                print(f"  ✓ {package}")
                passed += 1
            except ImportError as e:
                print(f"  ✗ {package} - FAILED: {str(e)}")
                failed.append((package, str(e)))
    
    print("\n" + "="*70)
    print(f"\nRESULTS:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg, error in failed:
            print(f"     - {pkg}: {error}")
        print("\nTo fix, run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✅ All dependencies installed successfully!")
        print(f"\nPython version: {sys.version}")
        print("\n🎉 You're ready to start coding!")
        sys.exit(0)

if __name__ == "__main__":
    test_imports()
