import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from utils.data_fetch import fetch_price_data, compute_rsi, compute_zscore

st.title("📊 Stock Screener & Fundamental Analyzer")

tickers = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA").split(",")
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

df_summary = []

for ticker in [t.strip().upper() for t in tickers]:
    try:
        data = fetch_price_data(ticker, period)
        if data.empty:
            continue

        rsi = compute_rsi(data["Close"])
        zscore = compute_zscore(data["Close"])
        current_price = data["Close"].iloc[-1]

        info = yf.Ticker(ticker).fast_info
        pe_ratio = getattr(info, "trailing_pe", None)
        dividend = getattr(info, "dividend_yield", None)
        market_cap = getattr(info, "market_cap", None)

        df_summary.append({
            "Ticker": ticker,
            "Price": current_price,
            "Z-Score": zscore,
            "RSI": rsi,
            "PE": pe_ratio,
            "Dividend": dividend,
            "Market Cap": market_cap
        })

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price"
        ))
        fig.update_layout(title=f"{ticker} Price Chart")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")

if df_summary:
    st.dataframe(pd.DataFrame(df_summary))
