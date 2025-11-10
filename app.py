## `app.py`

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    calc_pl,
    calc_target_price,
    calc_risk_reward,
    calc_stock_return,
    calc_cost_avg,
    calc_stop_levels,
    fetch_close_prices,
    correlation_matrix,
    event_impact_summary,
)

st.set_page_config(page_title="Trader Calculators", layout="wide")

# Load custom CSS
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    # fallback small CSS
    st.markdown(
        """
        <style>
        .card {background:#0f1724; padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6);}
        .card-title {font-size:18px; font-weight:600; margin-bottom:8px;}
        .muted {color: #9CA3AF}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("Trader Calculators — Single Page Dashboard")
st.write("Professional calculators for CFD & Stock trading. Click each calculator's `Calculate` button to run.")

# Layout: 2 columns grid for cards
col1, col2 = st.columns(2)

# ----------------------- Calculator: Profit/Loss (1.1) -----------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Profit / Loss Calculator (CFD / Stock)</div>', unsafe_allow_html=True)
    with st.form("pl_form"):
        direction = st.selectbox("Position Type", ["Buy/Long", "Sell/Short"], index=0)
        entry = st.number_input("Entry price", value=100.0, format="%.6f")
        exit_p = st.number_input("Exit price", value=110.0, format="%.6f")
        size = st.number_input("Position size (units/shares)", value=10.0, format="%.6f")
        spread = st.number_input("Spread (absolute)", value=0.0, format="%.6f")
        commission = st.number_input("Commission (total)", value=0.0, format="%.6f")
        submit = st.form_submit_button("Calculate P/L")
    if submit:
        pl, pl_pct = calc_pl(entry, exit_p, size, direction, spread, commission)
        col_a, col_b = st.columns([1,1])
        with col_a:
            st.metric("Profit / Loss", f"{pl:.2f}")
        with col_b:
            st.metric("Return %", f"{pl_pct:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Target Price for Desired Profit (1.3) -----------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Target Price for Desired Profit</div>', unsafe_allow_html=True)
    with st.form("target_form"):
        entry_t = st.number_input("Entry price", value=100.0, key="entry_t")
        desired_pct = st.number_input("Desired profit (%)", value=5.0, key="desired_pct")
        direction_t = st.selectbox("Position Type", ["Buy/Long", "Sell/Short"], key="dir_t")
        submit_t = st.form_submit_button("Calculate Target Price")
    if submit_t:
        target = calc_target_price(entry_t, desired_pct, direction_t)
        st.metric("Target Price", f"{target:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Risk/Reward Ratio (2.1) -----------------------
with col1:
    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Risk / Reward Ratio</div>', unsafe_allow_html=True)
    with st.form("rr_form"):
        entry_rr = st.number_input("Entry price", value=100.0, key="entry_rr")
        stop = st.number_input("Stop loss price", value=95.0, key="stop_rr")
        take = st.number_input("Take profit price", value=110.0, key="take_rr")
        submit_rr = st.form_submit_button("Calculate R/R")
    if submit_rr:
        rr, risk_amt, reward_amt = calc_risk_reward(entry_rr, stop, take)
        if rr >= 2.0:
            st.success(f"Risk/Reward = {rr:.2f} (Good)")
        elif rr >= 1.0:
            st.warning(f"Risk/Reward = {rr:.2f}")
        else:
            st.error(f"Risk/Reward = {rr:.2f} (Poor)")
        st.write(f"Risk amount (per unit): {risk_amt:.6f}, Reward (per unit): {reward_amt:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Stock Return (4.2) -----------------------
with col2:
    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Stock Return Calculator</div>', unsafe_allow_html=True)
    with st.form("stock_return_form"):
        ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL")
        buy_price = st.number_input("Buy price", value=100.0, key="buy_price")
        sell_price = st.number_input("Sell price", value=120.0, key="sell_price")
        dividends = st.number_input("Dividends received (total)", value=0.0)
        buy_date = st.date_input("Buy date (optional)")
        sell_date = st.date_input("Sell date (optional)")
        submit_stock = st.form_submit_button("Calculate Return")
    if submit_stock:
        total_return, cagr = calc_stock_return(buy_price, sell_price, dividends, buy_date, sell_date)
        st.metric("Total Return %", f"{total_return:.2f}%")
        if cagr is not None:
            st.metric("Annualized (CAGR)", f"{cagr:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Cost Averaging (4.3) -----------------------
with col1:
    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Cost Averaging Calculator</div>', unsafe_allow_html=True)
    st.write("Enter purchases as CSV (price,quantity) — one per line — or upload a CSV file with columns price,quantity.")
    with st.form("cost_avg_form"):
        text = st.text_area("Purchases (CSV lines)", value="100,10\n110,5")
        upload = st.file_uploader("Or upload CSV", type=["csv"])
        current_price = st.number_input("Current market price (optional)", value=115.0)
        submit_ca = st.form_submit_button("Calculate Average Cost")
    if submit_ca:
        df = None
        if upload is not None:
            df = pd.read_csv(upload)
        else:
            try:
                df = pd.read_csv(pd.compat.StringIO(text), header=None, names=["price", "quantity"])
            except Exception:
                st.error("Failed to parse purchases input. Use lines like: 100,10")
        if df is not None:
            avg_cost, total_qty, pnl = calc_cost_avg(df, current_price)
            st.write(f"Average cost: {avg_cost:.6f}")
            st.write(f"Total quantity: {total_qty}")
            st.write(f"Unrealised P/L at current price: {pnl:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Stop-Loss / Take-Profit Level Finder (4.4) -----------------------
with col2:
    st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Stop-Loss / Take-Profit Level Finder</div>', unsafe_allow_html=True)
    with st.form("stop_form"):
        entry_s = st.number_input("Entry price", value=100.0, key="entry_s")
        account = st.number_input("Account balance", value=10000.0)
        risk_pct = st.number_input("Risk per trade (%)", value=1.0)
        rr_desired = st.number_input("Desired Risk/Reward", value=2.0)
        submit_s = st.form_submit_button("Calculate Levels")
    if submit_s:
        stop_price, take_price, position_size = calc_stop_levels(entry_s, account, risk_pct, rr_desired)
        st.write(f"Suggested stop price: {stop_price:.6f}")
        st.write(f"Suggested take-profit price: {take_price:.6f}")
        st.write(f"Suggested position size (units): {position_size:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Correlation Heatmap (5.2) -----------------------
st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Correlation Heatmap</div>', unsafe_allow_html=True)
with st.form("corr_form"):
    tickers = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,GOOG,AMZN,TSLA")
    period = st.selectbox("Period", ["1y", "2y", "5y", "6mo"], index=0)
    submit_corr = st.form_submit_button("Fetch & Plot")
if submit_corr:
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(tickers_list) < 2:
        st.error("Enter at least two tickers")
    else:
        df_close = fetch_close_prices(tickers_list, period=period)
        corr = correlation_matrix(df_close)
        st.write("Correlation matrix")
        st.dataframe(corr)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Calculator: Economic Event Impact Tracker (5.4) -----------------------
st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Economic Event Impact Tracker (Stocks)</div>', unsafe_allow_html=True)
with st.form("event_form"):
    tick = st.text_input("Ticker to analyze", value="AAPL", key="evt_tick")
    dates_txt = st.text_area("Event dates (one YYYY-MM-DD per line)", value="2024-01-01\n2024-04-01")
    window = st.number_input("Days before/after event", value=3, min_value=1)
    submit_evt = st.form_submit_button("Analyze Events")
if submit_evt:
    dates = [d.strip() for d in dates_txt.splitlines() if d.strip()]
    if not dates:
        st.error("Please provide event dates")
    else:
        summary = event_impact_summary(tick, dates, window)
        st.write("Summary table (avg move %, std before, std after)")
        st.dataframe(summary)
        st.bar_chart(summary.set_index('date')['move_after_pct'])

# Footer
st.markdown("---")
st.caption("Built with Streamlit — calculators designed for quick trade planning and analysis")
```

---

## `utils.py`

```python
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
    df = yf.download(tickers, period=period, auto_adjust=True)['Close']
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
    hist = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True)['Close']
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
```

---

## `requirements.txt`

```
streamlit>=1.20
pandas
numpy
yfinance
matplotlib
seaborn
```

---

## `assets/style.css`

```css
/* assets/style.css - lightweight professional look */
body {
  color: #e6eef8;
  background: linear-gradient(180deg,#071124 0%, #071a2b 100%);
}
.card {background:#07172a; padding:18px; border-radius:12px; margin-bottom:12px;}
.card-title {font-size:18px; font-weight:700; color:#e6eef8;}
```

---

## `README.md`

````md
# Trader Calculators — Streamlit

Single-page Streamlit app with multiple trading calculators.

## Run locally

1. Create virtualenv and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

2. Run streamlit:

```bash
streamlit run app.py
```

## Notes

* The Correlation and Event Impact calculators use `yfinance` to fetch historical data. Internet connection required at runtime.
* Event Impact tracker expects event dates entered as `YYYY-MM-DD` lines.
* The stop-level calculator uses a simple 2% default stop distance; this is conservative placeholder logic — you can change to ATR-based stop in a future iteration.

```
```
