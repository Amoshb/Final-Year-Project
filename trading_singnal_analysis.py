import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
def main_trading(analysis_df):
    # Parameters
    window = 15
    min_distance = 5
    signal_shift = 5
    stop_loss_percentage = 0.5
    atr_multiplier = 8
    risk_free_rate = 0.1

    # Use the predicted and actual close prices (from ensemble_forex_trading.py)
    pred_close = np.array(analysis_df['Pred_Close'])
    actual_close = np.array(analysis_df['Actual_Close'])

    # Detect swing points
    potential_highs = argrelextrema(pred_close, np.greater, order=window)[0]
    potential_lows = argrelextrema(pred_close, np.less, order=window)[0]

    def filter_swings(swings, min_distance):
        filtered = []
        last = -min_distance
        for idx in swings:
            if idx - last >= min_distance:
                filtered.append(idx)
                last = idx
        return filtered

    swing_highs = filter_swings(potential_highs, min_distance)
    swing_lows = filter_swings(potential_lows, min_distance)

    analysis_df['Swing_High'] = np.nan
    analysis_df['Swing_Low'] = np.nan
    analysis_df.loc[swing_highs, 'Swing_High'] = pred_close[swing_highs]
    analysis_df.loc[swing_lows, 'Swing_Low'] = pred_close[swing_lows]

    analysis_df['Signal'] = 0
    analysis_df['Take_Profit'] = np.nan
    analysis_df['Stop_Loss'] = np.nan
    analysis_df['Trade_Result'] = np.nan

    # Signal generation
    for i in range(window, len(analysis_df) - window):
        if i in swing_lows and pred_close[i + 1] > pred_close[i]:
            idx = max(i - signal_shift, 0)
            analysis_df.loc[idx, 'Signal'] = 1
            future = [j for j in swing_highs if j > i]
            if future:
                tp = pred_close[future[0]]
                ac = analysis_df.loc[idx, 'Actual_Close']
                sl = ac - stop_loss_percentage * abs(tp - ac)
                analysis_df.loc[idx, ['Take_Profit', 'Stop_Loss']] = [tp, sl]

        elif i in swing_highs and pred_close[i + 1] < pred_close[i]:
            idx = max(i - signal_shift, 0)
            analysis_df.loc[idx, 'Signal'] = -1
            future = [j for j in swing_lows if j > i]
            if future:
                tp = pred_close[future[0]]
                ac = analysis_df.loc[idx, 'Actual_Close']
                sl = ac + stop_loss_percentage * abs(tp - ac)
                analysis_df.loc[idx, ['Take_Profit', 'Stop_Loss']] = [tp, sl]

    # ATR-based fallback for final signal
    atr = analysis_df['Actual_Close'].diff().abs().rolling(window=14).mean()
    last_trade_index = analysis_df[analysis_df['Signal'] != 0].index[-1]

    if pd.isna(analysis_df.loc[last_trade_index, 'Take_Profit']):
        ac = analysis_df.loc[last_trade_index, 'Actual_Close']
        if analysis_df.loc[last_trade_index, 'Signal'] == 1:
            tp = ac + (atr_multiplier * atr.iloc[-1])
            sl = ac - stop_loss_percentage * abs(tp - ac)
        else:
            tp = ac - (atr_multiplier * atr.iloc[-1])
            sl = ac + stop_loss_percentage * abs(tp - ac)
        analysis_df.loc[last_trade_index, ['Take_Profit', 'Stop_Loss']] = [tp, sl]

    # Determine trade result

    def check_trade_outcome(row):
        if row['Signal'] == 0 or pd.isna(row['Take_Profit']):
            return np.nan
        for price in actual_close[row.name:]:
            if row['Signal'] == 1 and price >= row['Take_Profit']:
                return 'Win'
            if row['Signal'] == 1 and price <= row['Stop_Loss']:
                return 'Loss'
            if row['Signal'] == -1 and price <= row['Take_Profit']:
                return 'Win'
            if row['Signal'] == -1 and price >= row['Stop_Loss']:
                return 'Loss'
        return np.nan

    analysis_df['Trade_Result'] = analysis_df.apply(check_trade_outcome, axis=1)

    # PnL calculation
    def calculate_pnl(row):
        if row['Trade_Result'] == 'Win':
            return abs(row['Take_Profit'] - row['Actual_Close'])
        elif row['Trade_Result'] == 'Loss':
            return -abs(row['Stop_Loss'] - row['Actual_Close'])
        return 0

    analysis_df['PnL'] = analysis_df.apply(lambda row: calculate_pnl(row) if row['Signal'] != 0 else 0, axis=1)

    # Metrics
    trades = analysis_df[analysis_df['Signal'] != 0]
    wins = analysis_df[analysis_df['Trade_Result'] == 'Win']
    losses = analysis_df[analysis_df['Trade_Result'] == 'Loss']
    returns = analysis_df['PnL'][analysis_df['PnL'] != 0]

    win_rate = (len(wins) / len(trades) * 100) if len(trades) else 0
    avg_pnl = trades['PnL'].mean()
    total_pnl = returns.sum()
    sharpe = (returns.mean() - risk_free_rate) / returns.std() if returns.std() != 0 else 0
    total_profit = returns[returns > 0].sum()
    total_loss = abs(returns[returns < 0].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else np.inf
    cumulative_pnl = returns.cumsum()
    max_drawdown = np.max(np.maximum.accumulate(cumulative_pnl) - cumulative_pnl)
    expectancy = (win_rate / 100) * returns[returns > 0].mean() - (1 - win_rate / 100) * abs(returns[returns < 0].mean())

    # Display
    print(f"Total Trades: {len(trades)}")
    print(f"Profitable Trades: {len(wins)}")
    print(f"Losing Trades: {len(losses)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average PnL: {avg_pnl:.4f}")
    print(f"Total PnL: {total_pnl:.4f}")
    print("------------------------------")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print(f"Expectancy per trade: {expectancy:.4f}")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(analysis_df.index, analysis_df['Actual_Close'], label='Actual Close', color='black', linewidth=2)
    plt.plot(analysis_df.index, analysis_df['Pred_Close'], label='Predicted Close', color='blue', linestyle='--', linewidth=1.5)
    plt.scatter(analysis_df.index[analysis_df['Signal'] == 1], analysis_df['Actual_Close'][analysis_df['Signal'] == 1], marker='^', color='green', label='Buy Signal', edgecolors='black')
    plt.scatter(analysis_df.index[analysis_df['Signal'] == -1], analysis_df['Actual_Close'][analysis_df['Signal'] == -1], marker='v', color='red', label='Sell Signal', edgecolors='black')

    for index, row in analysis_df.iterrows():
        if row['Trade_Result'] == 'Loss':
            plt.scatter(index, row['Actual_Close'], s=200, facecolors='none', edgecolors='red', linewidth=2, label='Losing Trade' if 'Losing Trade' not in plt.gca().get_legend_handles_labels()[1] else "")

    for _, row in analysis_df.iterrows():
        if not np.isnan(row['Take_Profit']):
            color = 'green' if row['Signal'] == 1 else 'red'
            plt.plot([row.name, row.name + signal_shift], [row['Actual_Close'], row['Take_Profit']], color=color, linestyle='dotted', alpha=0.6, label='Take Profit' if 'Take Profit' not in plt.gca().get_legend_handles_labels()[1] else "")
        if not np.isnan(row['Stop_Loss']):
            plt.plot([row.name, row.name + signal_shift], [row['Actual_Close'], row['Stop_Loss']], color='orange', linestyle='dotted', alpha=0.6, label='Stop Loss' if 'Stop Loss' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title("Swing High/Low Trading Signals")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    plt.show()