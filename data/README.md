# Data Source

This dataset is a **synthetic simulation** designed for educational purposes in 
quantitative finance. It was generated to demonstrate systematic trading strategy 
development and contains:

- **Daily returns** from a simulated asset (resembles equity index behavior)
- **10 factor signals** representing stylized market factors (momentum, value, etc.)
- **Transaction costs** of 50 basis points to simulate realistic trading friction

**Important:** This is NOT real market data. The factor-return relationships are 
artificially constructed to enable hands-on learning of quantitative research 
techniques including:
- Feature engineering and selection
- Time-series cross-validation
- Transaction cost-aware backtesting
- Risk-adjusted performance evaluation

## Factor Definitions

These are **stylized synthetic factors** representing common categories 
in quantitative equity research:

- **momentum_factor**: Trend-following signal
- **value_factor**: Price-to-fundamental ratio
- **volatility_factor**: Risk metric
- **quality_factor**: Profitability and earnings stability measure
- **size_factor**: Market capitalization exposure
- **liquidity_factor**: Trading volume and market impact indicator
- **sentiment_factor**: Market positioning and investor sentiment
- **technical_factor**: Price pattern and technical indicator
- **macro_factor**: Macroeconomic sensitivity measure
- **carry_factor**: Yield-based return signal

**Note:** Factor construction details are intentionally abstract as this 
is educational data. Real implementations would require specific formulas.