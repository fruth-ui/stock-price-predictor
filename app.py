import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2D3144;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { color: #8B8FA8; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-value { color: #FAFAFA; font-size: 28px; font-weight: 700; }
    .metric-delta-up   { color: #00D4A0; font-size: 14px; font-weight: 600; }
    .metric-delta-down { color: #FF4B6E; font-size: 14px; font-weight: 600; }
    .section-header { color: #00D4FF; font-size: 18px; font-weight: 600; margin: 8px 0 4px 0; border-left: 3px solid #00D4FF; padding-left: 10px; }
    div[data-testid="stSidebar"] { background: #1A1D27; }
    .stButton > button { background: #00D4FF; color: #0E1117; font-weight: 700; border-radius: 8px; border: none; width: 100%; padding: 10px; }
    .stButton > button:hover { background: #00B8D9; color: #0E1117; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font_color="#FAFAFA",
    xaxis=dict(gridcolor="#2D3144", showgrid=True),
    yaxis=dict(gridcolor="#2D3144", showgrid=True),
    legend=dict(bgcolor="#1A1D27", bordercolor="#2D3144", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Close"]].copy()
    df.columns = ["Close"]
    df["MA5"]  = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df.dropna(inplace=True)
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df

FEATURES = ["Close", "MA5", "MA10", "MA20", "Lag1", "Lag2"]

def train_and_predict(df: pd.DataFrame, test_pct: float):
    X, y = df[FEATURES], df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, shuffle=False)
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    # Retrain on full data for next-day prediction
    model.fit(X, y)
    last = df.iloc[-1]
    nf = np.array([[float(last[f].iloc[0]) if hasattr(last[f], "iloc") else float(last[f]) for f in FEATURES]])
    next_pred = float(model.predict(nf)[0])
    return model, X_train, X_test, y_train, y_test, preds, rmse, next_pred

def backtest(X_test, y_test, preds):
    bt = pd.DataFrame({"actual": y_test.values, "predicted": preds, "price": X_test["Close"].values}, index=y_test.index)
    bt["signal"]    = (bt["predicted"] > bt["price"]).astype(int)
    bt["mkt_ret"]   = bt["actual"].pct_change().fillna(0)
    bt["strat_ret"] = bt["mkt_ret"] * bt["signal"].shift(1).fillna(0)
    bt["cum_market"]   = (1 + bt["mkt_ret"]).cumprod()
    bt["cum_strategy"] = (1 + bt["strat_ret"]).cumprod()
    return bt

def metric_card(label, value, delta=None):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-up" if delta >= 0 else "metric-delta-down"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{arrow} ${abs(delta):.2f}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Settings")
    st.markdown("---")

    ticker1 = st.text_input("Primary ticker", value="SPLV").upper().strip()
    ticker2 = st.text_input("Compare ticker (optional)", value="").upper().strip()

    st.markdown("**Date range**")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=5*365), max_value=date.today() - timedelta(days=60))
    with col_b:
        end_date = st.date_input("End", value=date.today(), max_value=date.today())

    test_pct = st.slider("Test set size (%)", 10, 40, 20) / 100
    st.markdown("---")
    run = st.button("Run Analysis")

st.markdown("# 📈 Stock Price Predictor")
st.markdown("Linear Regression · Moving Averages · Backtesting")
st.markdown("---")

if not run:
    st.markdown("""
    <div style="text-align:center; padding: 60px 0; color: #8B8FA8;">
        <div style="font-size: 64px;">📊</div>
        <div style="font-size: 20px; margin-top: 16px;">Enter a ticker in the sidebar and click <b style="color:#00D4FF">Run Analysis</b></div>
        <div style="font-size: 14px; margin-top: 8px;">Try: AAPL · TSLA · SPY · SPLV · NVDA</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & process data ───────────────────────────────────────────────────────
tickers = [t for t in [ticker1, ticker2] if t]
results = {}

for tk in tickers:
    with st.spinner(f"Downloading {tk}…"):
        raw = yf.download(tk, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        st.error(f"No data for **{tk}**. Check the symbol.")
        continue
    df = build_features(raw)
    model, X_train, X_test, y_train, y_test, preds, rmse, next_pred = train_and_predict(df, test_pct)
    bt = backtest(X_test, y_test, preds)
    results[tk] = dict(df=df, X_test=X_test, y_test=y_test, preds=preds, rmse=rmse, next_pred=next_pred, bt=bt)

if not results:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Overview", "Predictions", "Backtesting", "Download"])

COLORS = {"primary": "#00D4FF", "secondary": "#FF6B9D", "green": "#00D4A0", "red": "#FF4B6E"}
ticker_colors = {tickers[0]: COLORS["primary"]}
if len(tickers) > 1:
    ticker_colors[tickers[1]] = COLORS["secondary"]

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    for tk, res in results.items():
        df      = res["df"]
        last_close = float(df["Close"].iloc[-1])
        delta      = res["next_pred"] - last_close
        days_in    = int((res["bt"]["signal"] == 1).sum())
        total_days = len(res["bt"])

        st.markdown(f'<div class="section-header">{tk}</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Last Close", f"${last_close:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Next-Day Prediction", f"${res['next_pred']:.2f}", delta), unsafe_allow_html=True)
        c3.markdown(metric_card("Test RMSE", f"${res['rmse']:.3f}"), unsafe_allow_html=True)
        c4.markdown(metric_card("Days in Market", f"{days_in} / {total_days}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Price history chart
        color = ticker_colors[tk]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=color, width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="#FFB347", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA5"],  name="MA5",  line=dict(color="#A78BFA", width=1, dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, title=f"{tk} – Price History", height=350)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Predictions ────────────────────────────────────────────────────────
with tabs[1]:
    if len(results) == 1:
        tk, res = next(iter(results.items()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res["y_test"].values, name="Actual",    line=dict(color=COLORS["green"],   width=2)))
        fig.add_trace(go.Scatter(y=res["preds"],         name="Predicted", line=dict(color=COLORS["primary"], width=2, dash="dash")))
        fig.update_layout(**PLOT_LAYOUT, title=f"{tk} – Actual vs Predicted (test set)", height=400, xaxis_title="Test-set day index", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Side-by-side comparison
        fig = make_subplots(rows=1, cols=2, subplot_titles=list(results.keys()))
        for i, (tk, res) in enumerate(results.items(), 1):
            color = ticker_colors[tk]
            fig.add_trace(go.Scatter(y=res["y_test"].values, name=f"{tk} Actual",    line=dict(color=COLORS["green"], width=2)),    row=1, col=i)
            fig.add_trace(go.Scatter(y=res["preds"],         name=f"{tk} Predicted", line=dict(color=color,           width=2, dash="dash")), row=1, col=i)
        fig.update_layout(**PLOT_LAYOUT, height=420, title="Actual vs Predicted – Comparison")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">Model Coefficients</div>', unsafe_allow_html=True)
    coef_cols = st.columns(len(results))
    for col, (tk, res) in zip(coef_cols, results.items()):
        with col:
            coefs   = LinearRegression().fit(res["df"][FEATURES], res["df"]["Target"]).coef_
            df_coef = pd.DataFrame({"Feature": FEATURES, "Coefficient": coefs})
            df_coef = df_coef.sort_values("Coefficient", key=abs, ascending=True)
            colors  = [COLORS["green"] if c > 0 else COLORS["red"] for c in df_coef["Coefficient"]]
            fig_c = go.Figure(go.Bar(x=df_coef["Coefficient"], y=df_coef["Feature"], orientation="h", marker_color=colors))
            fig_c.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=60, r=20, t=40, b=20)}, title=f"{tk} Coefficients", height=280)
            st.plotly_chart(fig_c, use_container_width=True)

# ── Tab 3: Backtesting ────────────────────────────────────────────────────────
with tabs[2]:
    fig = go.Figure()
    for tk, res in results.items():
        bt    = res["bt"]
        color = ticker_colors[tk]
        fig.add_trace(go.Scatter(x=bt.index, y=bt["cum_market"],   name=f"{tk} Buy & Hold", line=dict(color="#8B8FA8", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["cum_strategy"], name=f"{tk} Strategy",   line=dict(color=color,    width=2)))
    fig.update_layout(**PLOT_LAYOUT, title="Strategy vs Buy & Hold – Growth of $1", height=420, yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Backtest Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for tk, res in results.items():
        bt = res["bt"]
        summary_rows.append({
            "Ticker":              tk,
            "Buy & Hold Return":   f"{(bt['cum_market'].iloc[-1]   - 1)*100:.1f}%",
            "Strategy Return":     f"{(bt['cum_strategy'].iloc[-1] - 1)*100:.1f}%",
            "Days in Market":      f"{int(bt['signal'].sum())} / {len(bt)}",
            "Win Rate":            f"{(bt['strat_ret'] > 0).mean()*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows).set_index("Ticker"), use_container_width=True)

# ── Tab 4: Download ───────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
    st.markdown("Export predictions and backtest data as CSV.")

    for tk, res in results.items():
        bt = res["bt"]
        export = pd.DataFrame({
            "Date":            bt.index,
            "Actual_Close":    bt["actual"].values,
            "Predicted_Close": bt["predicted"].values,
            "Signal":          bt["signal"].values,
            "Market_Return":   bt["mkt_ret"].values,
            "Strategy_Return": bt["strat_ret"].values,
            "Cum_Market":      bt["cum_market"].values,
            "Cum_Strategy":    bt["cum_strategy"].values,
        })
        csv = export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download {tk} results (.csv)",
            data=csv,
            file_name=f"{tk}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
