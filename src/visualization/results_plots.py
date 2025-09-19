import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResultsVisualizer:
    """Visualization utilities for trading strategy results."""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def plot_equity_curve(self, results: pd.DataFrame, save_path: str = None):
        """Plot cumulative equity curve."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Equity curves
        ax1.plot(results.index, results['cumulative_gross'], 
                label='Gross Returns', linewidth=2)
        ax1.plot(results.index, results['cumulative_net'], 
                label='Net Returns', linewidth=2)
        ax1.set_title('Cumulative Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        net_cumulative = (1 + results['net_returns']).cumprod()
        running_max = net_cumulative.expanding().max()
        drawdown = (net_cumulative - running_max) / running_max
        
        ax2.fill_between(results.index, drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_returns_distribution(self, results: pd.DataFrame, save_path: str = None):
        """Plot distribution of returns."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Return distribution
        axes[0].hist(results['net_returns'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(results['net_returns'], dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot vs Normal Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_dashboard(self, results: pd.DataFrame, 
                                   metrics: Dict[str, float]):
        """Create comprehensive performance dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Equity Curve', 'Monthly Returns', 
                          'Rolling Sharpe Ratio', 'Position Distribution'],
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 1. Equity curve with drawdown
        fig.add_trace(
            go.Scatter(y=results['cumulative_net'], name='Cumulative Returns',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # 2. Monthly returns (simplified - daily returns)
        fig.add_trace(
            go.Bar(y=results['net_returns'], name='Daily Returns',
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe ratio
        rolling_sharpe = results['net_returns'].rolling(60).mean() / \
                        results['net_returns'].rolling(60).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(y=rolling_sharpe, name='60-Day Rolling Sharpe',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. Position distribution
        position_counts = results['positions'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Short', 'Flat', 'Long'], 
                  values=[position_counts.get(-1, 0), position_counts.get(0, 0), 
                         position_counts.get(1, 0)],
                  name="Positions"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Trading Strategy Performance Dashboard",
            showlegend=False,
            height=800,
            template="plotly_white"
        )
        
        # Add performance metrics as annotations
        metrics_text = f"""
        <b>Performance Metrics:</b><br>
        Annual Return: {metrics['annualized_return']:.2%}<br>
        Volatility: {metrics['annualized_volatility']:.2%}<br>
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}<br>
        Max Drawdown: {metrics['max_drawdown']:.2%}<br>
        Calmar Ratio: {metrics['calmar_ratio']:.3f}
        """
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.show()
        
        # Save as HTML
        fig.write_html("reports/figures/performance_dashboard.html")
