import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_advanced_health(monthly_df, burnout_score):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Historical Net Cash Flow
    ax.plot(monthly_df.index, monthly_df['net_cash_flow'], label='Historical Net Profit', 
            color='#2ecc71', marker='o', linewidth=2, alpha=0.8)
    
    # 2. Simple Forecast Overlay (for visuals)
    last_val = monthly_df['net_cash_flow'].iloc[-1]
    future_dates = pd.date_range(start=monthly_df.index[-1], periods=4, freq='MS')[1:]
    # Mocking visual trend for the dashboard
    ax.plot(future_dates, [last_val*0.9, last_val*0.85, last_val*0.8], 
            label='AI Trend Projection', color='#3498db', linestyle='--', marker='s')

    ax.fill_between(monthly_df.index, monthly_df['net_cash_flow'], color='#2ecc71', alpha=0.1)
    
    if burnout_score > 60:
        ax.text(0.02, 0.95, f"⚠️ BURNOUT ALERT: {burnout_score}%", 
                transform=ax.transAxes, color='#e74c3c', fontweight='bold', fontsize=12)

    ax.set_title('AI Business Intelligence: Net Cash Flow Forecast', fontsize=14, pad=20)
    ax.set_ylabel('Profit / Loss ($)')
    ax.legend()
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.show()