"""
Backtesting System - Test ML models on historical data
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import pandas as pd
import numpy as np
from pathlib import Path
import json

try:
    import lightgbm as lgb
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install lightgbm matplotlib")
    sys.exit(1)

ART = Path("ml/artifacts")
MODEL_DIR = Path("ml/models")

print("="*70)
print("BACKTESTING ML TRADING STRATEGY")
print("="*70)

# Load data
print("\nLoading dataset...")
df = pd.read_parquet(ART / "dataset.parquet")
with open(ART / "feature_cols.json") as f:
    FEATURES = json.load(f)

# Load models
print("Loading models...")
clf_dir = lgb.Booster(model_file=str(MODEL_DIR / "dir_model.txt"))
reg_vol = lgb.Booster(model_file=str(MODEL_DIR / "vol_model.txt"))
clf_regime = lgb.Booster(model_file=str(MODEL_DIR / "regime_model.txt"))

print(f"Dataset: {len(df)} samples")
print(f"Features: {len(FEATURES)}")

# Select one ticker for detailed backtest
ticker = "RELIANCE.NS"
print(f"\nBacktesting on: {ticker}")

df_t = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)

# Use only test period (last 20%)
start_idx = int(len(df_t) * 0.8)
test = df_t.iloc[start_idx:].reset_index(drop=True)

print(f"Test period: {test['date'].min()} to {test['date'].max()}")
print(f"Test days: {len(test)}")

# Trading simulation
initial_cash = 100000  # 1 lakh INR
cash = initial_cash
position = 0.0  # shares
transaction_cost = 0.002  # 0.2%

portfolio_values = []
trades = []
signals = []

for i, row in test.iterrows():
    X = row[FEATURES].values.reshape(1, -1)

    # Get predictions
    dir_probs = clf_dir.predict(X)[0]
    pred_dir = dir_probs.argmax()
    vol_pred = reg_vol.predict(X)[0]
    regime_probs = clf_regime.predict(X)[0]
    pred_regime = regime_probs.argmax()

    price = row['close']

    # Store signal
    signals.append({
        'date': row['date'],
        'price': price,
        'pred_dir': pred_dir,
        'prob_up': dir_probs[2],
        'prob_down': dir_probs[0],
        'vol_pred': vol_pred,
        'regime': pred_regime
    })

    # Trading strategy
    # Buy signal: high probability of UP and not in CRASH regime
    buy_signal = (dir_probs[2] > 0.6) and (pred_regime != 2)

    # Sell signal: high probability of DOWN or in CRASH regime
    sell_signal = (dir_probs[0] > 0.6) or (pred_regime == 2)

    # Execute trades
    if buy_signal and cash > 0:
        # Buy with 50% of cash
        investment = cash * 0.5
        shares_bought = investment / (price * (1 + transaction_cost))
        position += shares_bought
        cash -= investment
        trades.append({
            'date': row['date'],
            'action': 'BUY',
            'shares': shares_bought,
            'price': price,
            'amount': investment
        })

    elif sell_signal and position > 0:
        # Sell all position
        proceeds = position * price * (1 - transaction_cost)
        cash += proceeds
        trades.append({
            'date': row['date'],
            'action': 'SELL',
            'shares': position,
            'price': price,
            'amount': proceeds
        })
        position = 0

    # Calculate portfolio value
    portfolio_value = cash + position * price
    portfolio_values.append(portfolio_value)

# Final stats
final_value = portfolio_values[-1]
total_return = (final_value / initial_cash - 1) * 100

# Buy and hold comparison
buy_hold_shares = initial_cash / test.iloc[0]['close']
buy_hold_value = buy_hold_shares * test.iloc[-1]['close']
buy_hold_return = (buy_hold_value / initial_cash - 1) * 100

print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)

print(f"\nInitial Capital:     ₹{initial_cash:,.2f}")
print(f"Final Portfolio:     ₹{final_value:,.2f}")
print(f"Total Return:        {total_return:+.2f}%")
print(f"\nBuy & Hold Return:   {buy_hold_return:+.2f}%")
print(f"Outperformance:      {total_return - buy_hold_return:+.2f}%")

print(f"\nNumber of Trades:    {len(trades)}")
print(f"Final Cash:          ₹{cash:,.2f}")
print(f"Final Position:      {position:.2f} shares")

# Calculate metrics
returns = pd.Series(portfolio_values).pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

max_value = pd.Series(portfolio_values).cummax()
drawdown = (pd.Series(portfolio_values) - max_value) / max_value
max_drawdown = drawdown.min() * 100

print(f"\nSharpe Ratio:        {sharpe:.2f}")
print(f"Max Drawdown:        {max_drawdown:.2f}%")

# Show recent trades
if trades:
    print(f"\nRecent Trades (last 5):")
    for trade in trades[-5:]:
        print(f"  {trade['date']}: {trade['action']:4s} {trade['shares']:.2f} @ ₹{trade['price']:.2f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Portfolio value
ax1.plot(portfolio_values, label='ML Strategy', linewidth=2)
buy_hold_values = [buy_hold_shares * test.iloc[i]['close'] for i in range(len(test))]
ax1.plot(buy_hold_values, label='Buy & Hold', linewidth=2, alpha=0.7)
ax1.set_title(f'Portfolio Value - {ticker}', fontsize=14, fontweight='bold')
ax1.set_ylabel('Value (INR)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative returns
strategy_returns = [(v / initial_cash - 1) * 100 for v in portfolio_values]
buy_hold_rets = [(v / initial_cash - 1) * 100 for v in buy_hold_values]
ax2.plot(strategy_returns, label='ML Strategy', linewidth=2)
ax2.plot(buy_hold_rets, label='Buy & Hold', linewidth=2, alpha=0.7)
ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Return (%)')
ax2.set_xlabel('Trading Days')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(ART / "backtest.png", dpi=150, bbox_inches='tight')
print(f"\nBacktest plot saved to: {ART / 'backtest.png'}")

# Save backtest results
backtest_results = {
    'ticker': ticker,
    'test_period': {
        'start': str(test['date'].min()),
        'end': str(test['date'].max()),
        'days': len(test)
    },
    'performance': {
        'initial_capital': initial_cash,
        'final_value': float(final_value),
        'total_return_pct': float(total_return),
        'buy_hold_return_pct': float(buy_hold_return),
        'outperformance_pct': float(total_return - buy_hold_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown_pct': float(max_drawdown),
        'num_trades': len(trades)
    },
    'trades': trades
}

with open(ART / "backtest_results.json", "w") as f:
    json.dump(backtest_results, f, indent=2, default=str)

print(f"Backtest results saved to: {ART / 'backtest_results.json'}")

print("\n" + "="*70)
print("BACKTEST COMPLETE!")
print("="*70)
print("\nNext step: Start API server")
print("  uvicorn ml.api:app --reload --port 8001")
