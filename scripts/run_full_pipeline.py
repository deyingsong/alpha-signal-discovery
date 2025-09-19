import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

# Import custom modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.models.ridge_model import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.validation.time_series_cv import TimeSeriesCrossValidator
from src.backtesting.engine import BacktestEngine
from src.visualization.results_plots import ResultsVisualizer
from src.models.lightgbm_model import LightGBMModel

def setup_logging():
    """Configure logging."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )

def load_configs():
    """Load configuration files."""
    configs = {}
    config_dir = Path('configs')
    
    # Look for both .yaml and .yml files
    for pattern in ['*.yaml', '*.yml']:
        for config_file in config_dir.glob(pattern):
            with open(config_file, 'r') as f:
                configs[config_file.stem] = yaml.safe_load(f)
    
    return configs

def main():
    """Run the complete trading strategy pipeline."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting quantitative trading strategy pipeline")
    
    # Load configurations
    configs = load_configs()
    logger.info(f"Loaded config files: {list(configs.keys())}")
    
    data_config = configs['data_configs']
    model_configs = configs['model_configs']
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    loader = DataLoader(data_config)
    df, transaction_cost = loader.load_data(data_config['data']['file_path'])
    
    preprocessor = DataPreprocessor()
    
    # Analyze signal properties
    signal_stats = preprocessor.analyze_signal_properties(df)
    logger.info("Signal statistics:")
    logger.info(signal_stats)
    
    # Align features and targets
    X, y = preprocessor.align_features_targets(df)
    
    # Step 2: Feature engineering
    logger.info("Step 2: Feature engineering")
    feature_engineer = FeatureEngineer(data_config['feature_engineering'])
    X_engineered = feature_engineer.engineer_all_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_train_test(
        X_engineered, y, data_config['data']['train_test_split']
    )
    
    # Fit scaler and transform features
    feature_engineer.fit_scaler(X_train)
    X_train_scaled = feature_engineer.transform_features(X_train)
    X_test_scaled = feature_engineer.transform_features(X_test)
    
    # Step 3: Model development and cross-validation
    logger.info("Step 3: Model development and cross-validation")
    
    models = {
        'Ridge': RidgeModel,
        'RandomForest': RandomForestModel,
        'XGBoost': XGBoostModel,
        'LightGBM': LightGBMModel
    }
    
    cv_results = {}
    trained_models = {}
    
    cv = TimeSeriesCrossValidator(
        n_splits=data_config['validation']['cv_folds'],
        test_size=data_config['validation']['test_size'],
        gap=data_config['validation']['embargo_days']
    )
    
    for model_name, model_class in models.items():
        logger.info(f"Training {model_name}")
        
        # Cross-validation
        config_key = model_name.lower().replace('forest', '_forest')
        model_config = model_configs.get(config_key, {})
        
        cv_result = cv.cross_validate_model(model_class, model_config, X_train_scaled, y_train)
        cv_results[model_name] = cv_result
        
        # Train final model on full training set
        model = model_class(model_config)
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
    
    # Step 4: Select best model and make predictions
    logger.info("Step 4: Model selection and predictions")
    
    # Select model with best CV performance
    best_model_name = min(cv_results.keys(), key=lambda k: cv_results[k]['avg_mse'])
    best_model = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name}")
    
    # Make predictions on test set
    test_predictions = best_model.predict(X_test_scaled)
    
    # Step 5: Backtesting
    logger.info("Step 5: Backtesting")
    
    backtest_engine = BacktestEngine(transaction_cost)
    
    # Optimize threshold
    optimal_threshold, _ = backtest_engine.optimize_threshold(
        test_predictions, y_test.values
    )
    
    # Run final backtest
    backtest_results = backtest_engine.simulate_trading(
        test_predictions, y_test.values, optimal_threshold
    )
    
    # Calculate final performance metrics
    final_metrics = backtest_engine.calculate_performance_metrics(
        backtest_results['net_returns']
    )
    
    # Step 6: Results and visualization
    logger.info("Step 6: Results and reporting")
    
    logger.info("Final Performance Metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    results_dir = Path('data/results')
    results_dir.mkdir(exist_ok=True)
    
    backtest_results.to_csv(results_dir / 'backtest_results.csv')
    
    with open(results_dir / 'performance_metrics.yaml', 'w') as f:
        yaml.dump(final_metrics, f)
    
    # Generate visualizations
    visualizer = ResultsVisualizer()
    visualizer.create_performance_dashboard(backtest_results, final_metrics)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()