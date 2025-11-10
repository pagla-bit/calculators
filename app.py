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
                from io import StringIO
                df = pd.read_csv(StringIO(text), header=None, names=["price", "quantity"])
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
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with Streamlit — calculators designed for quick trade planning and analysis")
