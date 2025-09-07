import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from utils.data_fetch import fetch_price_data, compute_rsi, compute_zscore

st.title("📊 Stock Screener & Advanced Analyzer")

tickers = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA").split(",")
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

df_summary = []

# === Helper Functions ===
def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper.iloc[-1], lower.iloc[-1]

def anomaly_score(series):
    """Detect extreme moves relative to rolling volatility (Anomaly Trading)."""
    returns = series.pct_change().dropna()
    rolling_vol = returns.rolling(20).std()
    score = (returns.iloc[-1] / rolling_vol.iloc[-1])
    return score

def deja_vu_similarity(series, window=10):
    """Pattern matching: compare last N days with rolling history."""
    if len(series) < window * 2:
        return None
    recent = series[-window:].pct_change().dropna().values
    best_corr = -1
    for i in range(len(series) - 2 * window):
        past = series[i:i+window].pct_change().dropna().values
        corr = np.corrcoef(recent, past)[0,1]
        if corr > best_corr:
            best_corr = corr
    return best_corr

def trending_reversion_signals(series):
    ma20, ma50 = series.rolling(20).mean(), series.rolling(50).mean()
    rsi = compute_rsi(series)
    if ma20.iloc[-1] > ma50.iloc[-1] and rsi < 70:
        return "Trending Up"
    elif ma20.iloc[-1] < ma50.iloc[-1] and rsi > 30:
        return "Trending Down"
    elif rsi > 70:
        return "Overbought (Reversion risk)"
    elif rsi < 30:
        return "Oversold (Reversion chance)"
    else:
        return "Neutral"

# === Loop through tickers ===
for ticker in [t.strip().upper() for t in tickers]:
    try:
        data = fetch_price_data(ticker, period)
        if data.empty:
            continue

        close = data["Close"]
        current_price = close.iloc[-1]

        # Technicals
        rsi = compute_rsi(close)
        zscore = compute_zscore(close)
        macd, macd_signal = compute_macd(close)
        upper_bb, lower_bb = compute_bollinger(close)

        # Quant signals
        anomaly = anomaly_score(close)
        deja_corr = deja_vu_similarity(close)
        trend_signal = trending_reversion_signals(close)

        # Fundamentals
        info = yf.Ticker(ticker).info
        pe_ratio = info.get("trailingPE", None)
        pb_ratio = info.get("priceToBook", None)
        ptb_ratio = None
        if pb_ratio:  # Price / Tangible Book = P/B adjusted
            tangible_assets = info.get("bookValue", None)  # Simplified approximation
            if tangible_assets:
                ptb_ratio = current_price / tangible_assets
        dividend = info.get("dividendYield", None)
        market_cap = info.get("marketCap", None)

        df_summary.append({
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "RSI": round(rsi, 2),
            "Z-Score": round(zscore, 2),
            "MACD": round(macd, 2),
            "MACD Signal": round(macd_signal, 2),
            "Anomaly Score": round(anomaly, 2),
            "Déjà Vu Corr": round(deja_corr, 2) if deja_corr else None,
            "Trend/Reversion": trend_signal,
            "P/E": pe_ratio,
            "P/B": pb_ratio,
            "Price/Tangible Book": ptb_ratio,
            "Dividend Yield": dividend,
            "Market Cap": market_cap
        })

        # === Chart with Bollinger Bands ===
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price"
        ))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(20).mean(), line=dict(color='blue'), name="20MA"))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(50).mean(), line=dict(color='orange'), name="50MA"))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(200).mean(), line=dict(color='green'), name="200MA"))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(20).mean()+2*close.rolling(20).std(), line=dict(color='red', dash='dot'), name="Upper BB"))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(20).mean()-2*close.rolling(20).std(), line=dict(color='red', dash='dot'), name="Lower BB"))

        fig.update_layout(title=f"{ticker} Technical Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")

# === Show results ===
if df_summary:
    st.subheader("📋 Stock Analysis Summary")
    df = pd.DataFrame(df_summary)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "stock_analysis.csv", "text/csv")
