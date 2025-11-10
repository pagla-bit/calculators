# app.py
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
