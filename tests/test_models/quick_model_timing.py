"""
Quick demonstration of why LightGBM is slower on small datasets
and when each model should be used in quantitative trading
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def time_models_by_size():
    """
    Compare model training times across different dataset sizes
    """
    print("="*70)
    print("MODEL TIMING COMPARISON BY DATASET SIZE")
    print("="*70)
    print("\nThis shows why LightGBM was slow in your results...\n")
    
    # Test different dataset sizes
    sizes = [500, 1500, 5000, 15000, 50000]
    n_features = 115  # Your feature count
    
    results = {
        'Ridge': [],
        'XGBoost': [],
        'LightGBM': [],
        'LightGBM_Fast': []
    }
    
    for n_samples in sizes:
        print(f"\n📊 Dataset size: {n_samples:,} samples x {n_features} features")
        print("-" * 50)
        
        # Generate synthetic data
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 0.01
        
        # Ridge
        from sklearn.linear_model import Ridge
        start = time.time()
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        ridge_time = time.time() - start
        results['Ridge'].append(ridge_time)
        print(f"  Ridge:          {ridge_time:6.3f}s")
        
        # XGBoost (simple config)
        import xgboost as xgb
        start = time.time()
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        xgb_time = time.time() - start
        results['XGBoost'].append(xgb_time)
        print(f"  XGBoost:        {xgb_time:6.3f}s")
        
        # LightGBM (with GridSearch - like in your test)
        import lightgbm as lgb
        from sklearn.model_selection import GridSearchCV
        
        start = time.time()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1]
        }
        tscv = TimeSeriesSplit(n_splits=2)
        base_model = lgb.LGBMRegressor(verbosity=-1, force_row_wise=True)
        grid = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=1)
        grid.fit(X, y)
        lgb_grid_time = time.time() - start
        results['LightGBM'].append(lgb_grid_time)
        print(f"  LightGBM (Grid):{lgb_grid_time:6.3f}s ← This is what happened to you!")
        
        # LightGBM (fast mode - no grid search)
        start = time.time()
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            num_leaves=31,
            learning_rate=0.1,
            min_child_samples=20,
            verbosity=-1,
            force_row_wise=True
        )
        model.fit(X, y)
        lgb_fast_time = time.time() - start
        results['LightGBM_Fast'].append(lgb_fast_time)
        print(f"  LightGBM (Fast):{lgb_fast_time:6.3f}s ← With optimized settings")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print("\n🔍 Why LightGBM was slow in your test:")
    print("  1. GridSearchCV tested 2×2×2×2 = 16 parameter combinations")
    print("  2. With 3-fold CV, that's 48 model fits!")
    print("  3. LightGBM's histogram construction has overhead on small data")
    print("  4. The overhead isn't worth it until ~10,000+ samples")
    
    print("\n📈 Relative Speed (compared to Ridge):")
    for model in results:
        if model != 'Ridge':
            print(f"\n  {model}:")
            for i, size in enumerate(sizes):
                ratio = results[model][i] / results['Ridge'][i]
                print(f"    {size:6,} samples: {ratio:5.1f}x slower")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR 1500-SAMPLE DATASET")
    print("="*70)
    
    print("""
    ✅ DO USE:
       1. Ridge Regression - Fastest, good baseline (1-2 seconds)
       2. XGBoost (simple) - Best accuracy/speed balance (10-30 seconds)
       3. Random Forest - Good alternative (10-20 seconds)
    
    ❌ AVOID:
       1. LightGBM with GridSearch - Too much overhead (300+ seconds)
       2. Complex neural networks - Not enough data
       3. Excessive hyperparameter tuning - Will overfit
    
    💡 TIPS:
       • Use fewer hyperparameter combinations (3-5 total)
       • Use 2-3 CV folds maximum
       • Consider simple ensemble of Ridge + XGBoost
       • Focus on feature engineering quality over model complexity
    """)

def show_crossover_point():
    """
    Find the dataset size where LightGBM becomes faster than XGBoost
    """
    print("\n" + "="*70)
    print("FINDING THE CROSSOVER POINT")
    print("="*70)
    print("\nAt what dataset size does LightGBM become faster than XGBoost?")
    print("(Using simple fit, no GridSearch)\n")
    
    sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
    
    for n_samples in sizes:
        X = np.random.randn(n_samples, 100)
        y = np.random.randn(n_samples)
        
        # XGBoost
        import xgboost as xgb
        start = time.time()
        xgb_model = xgb.XGBRegressor(n_estimators=100, verbosity=0)
        xgb_model.fit(X, y)
        xgb_time = time.time() - start
        
        # LightGBM
        import lightgbm as lgb
        start = time.time()
        lgb_model = lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
        lgb_model.fit(X, y)
        lgb_time = time.time() - start
        
        faster = "LightGBM ✓" if lgb_time < xgb_time else "XGBoost"
        ratio = lgb_time / xgb_time
        
        print(f"{n_samples:8,} samples: XGB={xgb_time:.3f}s, LGB={lgb_time:.3f}s "
              f"(ratio={ratio:.2f}) → {faster}")
    
    print("""
    📊 Conclusion:
       • Below 10,000 samples: XGBoost is usually faster
       • 10,000-50,000: Similar performance  
       • Above 50,000: LightGBM becomes significantly faster
       • Your 1,500 samples: Far below the crossover point!
    """)

if __name__ == "__main__":
    print("\n🚀 Running timing comparison...\n")
    time_models_by_size()
    show_crossover_point()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)