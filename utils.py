# utils.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 1.1 Profit/Loss
def calc_pl(entry, exit_p, size, direction, spread=0.0, commission=0.0):
    # Apply spread to entry for buys (entry becomes entry+spread/2) and sells
    # For simplicity we treat spread as absolute subtracted from exit for buy, added for sell
    if direction.lower().startswith("buy"):
        effective_entry = entry + spread/2
        effective_exit = exit_p - spread/2
        pl = (effective_exit - effective_entry) * size - commission
    else:
        # short
        effective_entry = entry - spread/2
        effective_exit = exit_p + spread/2
        pl = (effective_entry - effective_exit) * size - commission
    pl_pct = (pl / (effective_entry * size)) * 100 if effective_entry * size != 0 else 0
    return pl, pl_pct

# 1.3 Target price for desired profit
def calc_target_price(entry, desired_pct, direction):
    if direction.lower().startswith("buy"):
        return entry * (1 + desired_pct / 100)
    else:
        return entry * (1 - desired_pct / 100)

# 2.1 Risk/Reward
def calc_risk_reward(entry, stop, take):
    risk = abs(entry - stop)
    reward = abs(take - entry)
    rr = reward / risk if risk != 0 else float('inf')
    return rr, risk, reward

# 4.2 Stock Return Calculator
def calc_stock_return(buy_price, sell_price, dividends=0.0, buy_date=None, sell_date=None):
    total_return = ((sell_price + dividends) / buy_price - 1) * 100
    cagr = None
    try:
        if buy_date and sell_date:
            bd = pd.to_datetime(buy_date)
            sd = pd.to_datetime(sell_date)
            years = (sd - bd).days / 365.25
            if years > 0:
                cagr = ((sell_price + dividends) / buy_price) ** (1 / years) - 1
                cagr = cagr * 100
    except Exception:
        cagr = None
    return total_return, cagr

# 4.3 Cost Averaging
def calc_cost_avg(df_purchases, current_price=None):
    df = df_purchases.copy()
    df['cost'] = df['price'] * df['quantity']
    total_cost = df['cost'].sum()
    total_qty = df['quantity'].sum()
    avg_cost = total_cost / total_qty if total_qty != 0 else 0
    pnl = 0
    if current_price is not None:
        pnl = (current_price - avg_cost) * total_qty
    return avg_cost, total_qty, pnl

# 4.4 Stop levels calculator (simple)
def calc_stop_levels(entry, account_balance, risk_pct, rr_desired):
    # Risk per trade absolute
    risk_amount = account_balance * (risk_pct / 100)
    # For simplicity assume risk per unit is some pct of entry price (user later adjusts). We'll compute stop distance assuming position size of 1
    # We'll set stop distance such that position size 1 would risk `entry - stop = risk_abs` -> unrealistic; instead compute position size
    # Position size = risk_amount / (entry - stop). But we don't have stop. Approach: assume stop distance is entry * 0.02 (2%) default
    stop_distance = entry * 0.02
    stop_price = entry - stop_distance
    take_price = entry + stop_distance * rr_desired
    position_size = risk_amount / stop_distance if stop_distance != 0 else 0
    return stop_price, take_price, position_size

# Shared helpers for data fetching and analytics

def fetch_close_prices(tickers, period='1y'):
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna()

def correlation_matrix(df_close):
    returns = df_close.pct_change().dropna()
    return returns.corr()

# 5.4 Event impact: compute avg move % after event and volatility before/after

def event_impact_summary(ticker, dates, window=3):
    # Build results list
    results = []
    # Determine start and end for fetching enough history
    start = pd.to_datetime(min(dates)) - pd.Timedelta(days=window + 30)
    end = pd.to_datetime(max(dates)) + pd.Timedelta(days=window + 10)
    hist = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)['Close']
    for d in dates:
        try:
            dt = pd.to_datetime(d)
            if dt not in hist.index:
                # find nearest trading day
                dt = hist.index[hist.index.get_indexer([dt], method='nearest')[0]]
            before = hist.loc[dt - pd.Timedelta(days=window): dt - pd.Timedelta(days=1)]
            after = hist.loc[dt + pd.Timedelta(days=1): dt + pd.Timedelta(days=window)]
            if len(before) == 0 or len(after) == 0:
                continue
            move_after_pct = (after.iloc[-1] / hist.loc[dt] - 1) * 100
            std_before = before.pct_change().std() * 100
            std_after = after.pct_change().std() * 100
            results.append({
                'date': dt.strftime('%Y-%m-%d'),
                'move_after_pct': move_after_pct,
                'std_before_pct': std_before,
                'std_after_pct': std_after,
            })
        except Exception:
            continue
    return pd.DataFrame(results)
