# Alpha Signal Discovery

A comprehensive Python framework for discovering alpha signal, developing trading strategy, and managing risk in financial data, adapted to ~2,000 samples.

## Project Overview

This project implements a systematic approach to quantitative trading strategy development, following industry best practices for:

- **Data Analysis**: Comprehensive exploratory data analysis and feature engineering
- **Model Development**: Implementation of multiple ML models (Ridge, Random Forest, XGBoost)
- **Validation**: Walk-forward cross-validation with purging and embargoing
- **Backtesting**: Realistic simulation including transaction costs and slippage
- **Risk Management**: Comprehensive performance and risk analytics

## Key Features

### Data Processing
- Robust data loading and validation
- Advanced feature engineering (lags, interactions, polynomials)
- Proper temporal alignment to prevent lookahead bias

### Model Implementation
- **Ridge Regression**: Regularized linear baseline model
- **Random Forest**: Ensemble model for non-linear relationships
- **XGBoost**: Gradient boosting with advanced regularization

### Validation Framework
- Walk-forward cross-validation for time series data
- Purging and embargoing to eliminate data leakage
- Risk-adjusted performance metrics (Sharpe ratio, Calmar ratio)

### Backtesting Engine
- Realistic transaction cost modeling (50 bps per trade)
- Threshold optimization for signal filtering
- Comprehensive performance analytics

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/deyingsong/quantitative-trading-strategy.git
cd quantitative-trading-strategy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running the Pipeline

```bash
python scripts/run_full_pipeline.py
```

## Project Structure

```
quantitative-trading-strategy/
├── README.md
├── configs                # Configuration files
├── data
│   ├── processed
│   ├── raw
│   └── results
├── logs                   # Experiment logs
├── models                 # Saved models
├── notebooks              # Jupyter notebooks for analysis
├── repo_structure.txt
├── reports
├── requirements.txt
├── scripts                # Execution scripts
├── setup.py
├── src                    # Source code
│   ├── backtesting        # Backtesting engine
│   ├── data               # Data processing modules
│   ├── models             # Model implementation
│   ├── utils              # Utility modules
│   ├── validation         # Cross-validation framework
│   └── visualization      # Plotting and reporting
└── tests                  # Unit tests
  ├── test_backtesting
  ├── test_data
  └── test_models
```


## Configuration

The framework uses YAML configuration files for easy parameter tuning:

Model Parameters (`configs/model_configs.yaml`)

Data Parameters (`configs/data_configs.yaml`)



## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/test_backtesting/test_engine.py
```

## Visualization

The framework generates comprehensive visualizations:

- Interactive performance dashboards
- Equity curve and drawdown analysis
- Feature importance plots
- Return distribution analysis


## References

- de Prado, M. L. (2018). Advances in Financial Machine Learning. United Kingdom: Wiley.
- Test, J., Broker, M. (2020). Machine Learning for Algorithmic Trading: Master as a PRO Applied Artificial Intelligence and Python for Predict Systematic Strategies for Options and Stocks. Learn Data-driven Finance Using Keras. Portugal: Libero Fabio Dachille.


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request



