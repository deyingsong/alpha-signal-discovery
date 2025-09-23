"""
Model Comparison Script - LightGBM vs XGBoost vs Random Forest
This script compares the performance of different gradient boosting implementations
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
import yaml

# Import models
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.ridge_model import RidgeModel

def compare_models(X_train, y_train, X_test, y_test, configs):
    """
    Compare performance of different models
    """
    results = {}
    
    models_to_test = {
        'LightGBM': (LightGBMModel, configs.get('lightgbm', {})),
        'XGBoost': (XGBoostModel, configs.get('xgboost', {})),
        'RandomForest': (RandomForestModel, configs.get('random_forest', {})),
        'Ridge': (RidgeModel, configs.get('ridge', {}))
    }
    
    for model_name, (model_class, model_config) in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Record training time
        start_time = time.time()
        
        # Initialize and train model
        model = model_class(model_config)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        pred_time = time.time() - start_time
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Store results
        results[model_name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_time': train_time,
            'pred_time': pred_time,
            'best_params': model.get_params() if hasattr(model, 'get_params') else None
        }
        
        # Get feature importance for tree-based models
        if model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                results[model_name]['top_features'] = feature_importance.head(5).to_dict()
    
    return results

def print_comparison_table(results):
    """
    Print a formatted comparison table
    """
    # Create DataFrame for easy comparison
    comparison_df = pd.DataFrame(results).T
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Accuracy metrics
    print("\n Accuracy Metrics:")
    print("-" * 50)
    accuracy_cols = ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']
    for col in accuracy_cols:
        if col in comparison_df.columns:
            print(f"\n{col.upper()}:")
            for model in comparison_df.index:
                value = comparison_df.loc[model, col]
                print(f"  {model:15s}: {value:.6f}")
    
    # Speed metrics
    print("\n⚡ Speed Metrics:")
    print("-" * 50)
    speed_cols = ['train_time', 'pred_time']
    for col in speed_cols:
        if col in comparison_df.columns:
            print(f"\n{col.upper()} (seconds):")
            for model in comparison_df.index:
                value = comparison_df.loc[model, col]
                print(f"  {model:15s}: {value:.3f}")
    
    # Best model identification
    print("\n Best Model by Metric:")
    print("-" * 50)
    
    # Lower is better
    for metric in ['test_rmse', 'test_mae']:
        if metric in comparison_df.columns:
            best_model = comparison_df[metric].idxmin()
            best_value = comparison_df[metric].min()
            print(f"{metric.upper():15s}: {best_model} ({best_value:.6f})")
    
    # Higher is better
    if 'test_r2' in comparison_df.columns:
        best_model = comparison_df['test_r2'].idxmax()
        best_value = comparison_df['test_r2'].max()
        print(f"{'TEST_R2':15s}: {best_model} ({best_value:.6f})")
    
    # Fastest training
    if 'train_time' in comparison_df.columns:
        fastest_model = comparison_df['train_time'].idxmin()
        fastest_time = comparison_df['train_time'].min()
        print(f"{'FASTEST TRAIN':15s}: {fastest_model} ({fastest_time:.3f}s)")
    
    return comparison_df

def analyze_lightgbm_advantages():
    """
    Highlight specific advantages of LightGBM
    """
    print("\n" + "="*80)
    print("💡 LightGBM ADVANTAGES")
    print("="*80)
    
    advantages = {
        "Speed": [
            "✓ Histogram-based algorithm for faster training",
            "✓ Leaf-wise tree growth (more efficient than level-wise)",
            "✓ Optimized for parallel and GPU learning"
        ],
        "Accuracy": [
            "✓ Better handling of categorical features",
            "✓ Advanced regularization techniques",
            "✓ Optimized for large datasets"
        ],
        "Memory": [
            "✓ Lower memory usage through histogram binning",
            "✓ Efficient handling of sparse features",
            "✓ Optimized data loading"
        ],
        "Features": [
            "✓ Native support for categorical features",
            "✓ Built-in cross-validation",
            "✓ Feature bundling for high-dimensional data"
        ]
    }
    
    for category, points in advantages.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  {point}")
    
    print("\n" + "="*80)
    print("📌 WHEN TO USE LIGHTGBM OVER XGBOOST:")
    print("="*80)
    print("""
    1. Large datasets (>10k samples): LightGBM's histogram-based approach shines
    2. Many features (>100): Feature bundling reduces dimensionality efficiently  
    3. Need faster training: Often 10-20x faster than XGBoost on large data
    4. Limited memory: More memory-efficient through binning
    5. Categorical features: Native support without one-hot encoding
    """)

def main():
    """
    Main comparison workflow
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("QUANTITATIVE TRADING MODEL COMPARISON")
    print("Comparing LightGBM, XGBoost, Random Forest, and Ridge Regression")
    print("="*80)
    
    # Load your preprocessed data here
    # This is a placeholder - replace with your actual data loading
    print("\n📂 Loading data...")
    # X_train, X_test, y_train, y_test = load_your_data()
    
    # For demonstration, using synthetic data
    np.random.seed(42)
    n_samples_train, n_samples_test = 1000, 300
    n_features = 115  # Your feature count after engineering
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_samples_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randn(n_samples_train) * 0.01)
    y_test = pd.Series(np.random.randn(n_samples_test) * 0.01)
    
    # Load model configurations
    print("\n⚙️ Loading model configurations...")
    config_path = Path('configs/model_configs.yml')
    
    # Default configs if file doesn't exist
    configs = {
        'lightgbm': {
            'n_estimators_range': [100, 200],
            'max_depth_range': [3, 5],
            'learning_rate_range': [0.05, 0.1],
            'num_leaves_range': [31, 50],
            'cv_folds': 3
        },
        'xgboost': {
            'n_estimators_range': [100, 200],
            'max_depth_range': [3, 5],
            'learning_rate_range': [0.05, 0.1],
            'cv_folds': 3
        },
        'random_forest': {
            'n_estimators_range': [100, 200],
            'max_depth_range': [5, 10],
            'cv_folds': 3
        },
        'ridge': {
            'alpha_range': [0.1, 1.0, 10.0],
            'cv_folds': 3
        }
    }
    
    # Run comparison
    print("\n🚀 Starting model comparison...")
    results = compare_models(X_train, y_train, X_test, y_test, configs)
    
    # Print results
    comparison_df = print_comparison_table(results)
    
    # Analyze LightGBM advantages
    analyze_lightgbm_advantages()
    
    # Save results
    print("\n💾 Saving results...")
    comparison_df.to_csv('model_comparison_results.csv')
    print("Results saved to 'model_comparison_results.csv'")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()