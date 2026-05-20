import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor")
st.caption("Linear Regression · Moving Average Features · Backtesting")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="SPLV").upper().strip()
    period = st.selectbox("History", ["1y", "2y", "5y", "10y", "max"], index=2)
    test_pct = st.slider("Test set size (%)", 10, 40, 20) / 100
    st.markdown("---")
    run = st.button("Run", use_container_width=True)

if not run:
    st.info("Enter a ticker in the sidebar and click **Run**.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────
with st.spinner(f"Downloading {ticker} data…"):
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

if raw.empty:
    st.error(f"No data found for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

df = raw[["Close"]].copy()
df.columns = ["Close"]

# ── Feature engineering ───────────────────────────────────────────────────────
df["MA5"]  = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["Lag1"] = df["Close"].shift(1)   # yesterday's close
df["Lag2"] = df["Close"].shift(2)
df.dropna(inplace=True)

df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

FEATURES = ["Close", "MA5", "MA10", "MA20", "Lag1", "Lag2"]
X = df[FEATURES]
y = df["Target"]

# ── Train / test split (time-ordered) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_pct, shuffle=False
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# ── Retrain on full data for next-day prediction ───────────────────────────────
model.fit(X, y)
last = df.iloc[-1]
next_features = np.array([[
    float(last["Close"].iloc[0]) if hasattr(last["Close"], "iloc") else float(last["Close"]),
    float(last["MA5"].iloc[0])   if hasattr(last["MA5"],   "iloc") else float(last["MA5"]),
    float(last["MA10"].iloc[0])  if hasattr(last["MA10"],  "iloc") else float(last["MA10"]),
    float(last["MA20"].iloc[0])  if hasattr(last["MA20"],  "iloc") else float(last["MA20"]),
    float(last["Lag1"].iloc[0])  if hasattr(last["Lag1"],  "iloc") else float(last["Lag1"]),
    float(last["Lag2"].iloc[0])  if hasattr(last["Lag2"],  "iloc") else float(last["Lag2"]),
]])
next_pred = float(model.predict(next_features)[0])
last_close = float(df["Close"].iloc[-1])

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Last close", f"${last_close:.2f}")
c3.metric("Next-day prediction", f"${next_pred:.2f}", f"{next_pred - last_close:+.2f}")
c4.metric("Test RMSE", f"${rmse:.3f}")

st.markdown("---")

# ── Actual vs Predicted chart ─────────────────────────────────────────────────
st.subheader("Actual vs Predicted (test set)")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(y_test.values, label="Actual", linewidth=1.5)
ax1.plot(predictions,  label="Predicted", linewidth=1.5, linestyle="--")
ax1.set_xlabel("Test-set day index")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ── Backtesting ───────────────────────────────────────────────────────────────
st.subheader("Backtesting: model strategy vs buy-and-hold")

# Re-fit on train set only so backtest uses true out-of-sample predictions
model_bt = LinearRegression()
model_bt.fit(X_train, y_train)
bt_preds = model_bt.predict(X_test)

bt = pd.DataFrame({
    "actual":    y_test.values,
    "predicted": bt_preds,
    "price":     X_test["Close"].values,
}, index=y_test.index)

# Signal: buy (1) when model predicts price will rise, else cash (0)
bt["signal"]  = (bt["predicted"] > bt["price"]).astype(int)
bt["mkt_ret"] = bt["actual"].pct_change().fillna(0)

# Strategy return = market return on days we hold, 0 otherwise
# shift signal by 1: we act on today's prediction for tomorrow
bt["strat_ret"] = bt["mkt_ret"] * bt["signal"].shift(1).fillna(0)

bt["cum_market"]   = (1 + bt["mkt_ret"]).cumprod()
bt["cum_strategy"] = (1 + bt["strat_ret"]).cumprod()

final_market   = bt["cum_market"].iloc[-1]
final_strategy = bt["cum_strategy"].iloc[-1]

b1, b2, b3 = st.columns(3)
b1.metric("Buy-and-hold return", f"{(final_market   - 1)*100:.1f}%")
b2.metric("Strategy return",     f"{(final_strategy - 1)*100:.1f}%")
b3.metric("Days in market",      f"{int(bt['signal'].sum())} / {len(bt)}")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(bt.index, bt["cum_market"],   label="Buy & hold", linewidth=1.5)
ax2.plot(bt.index, bt["cum_strategy"], label="Model strategy", linewidth=1.5, linestyle="--")
ax2.set_ylabel("Growth of $1")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(plt.MaxNLocator(8))
fig2.autofmt_xdate()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ── Feature importance (coefficients) ────────────────────────────────────────
st.subheader("Model coefficients")
coef_df = pd.DataFrame({
    "Feature":     FEATURES,
    "Coefficient": model.coef_,
}).sort_values("Coefficient", key=abs, ascending=False)

fig3, ax3 = plt.subplots(figsize=(7, 3))
colors = ["steelblue" if c > 0 else "tomato" for c in coef_df["Coefficient"]]
ax3.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
ax3.set_xlabel("Coefficient value")
ax3.axvline(0, color="black", linewidth=0.8)
ax3.grid(True, alpha=0.3, axis="x")
st.pyplot(fig3)
plt.close(fig3)
