import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.title("🧮 Strategy Decision Matrix")

ticker = st.text_input("Enter ticker", "AAPL").upper()
period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=2)

try:
    data = yf.download(ticker, period=period, interval="1d")
    ma20 = data["Close"].rolling(20).mean()
    ma50 = data["Close"].rolling(50).mean()
    trend = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"
    st.write(f"**Detected Trend:** {trend}")
except Exception:
    st.error("Could not fetch data.")
    st.stop()

strategies = {
    "Bullish": ["Long Call", "Bull Call Spread", "Cash-Secured Put"],
    "Bearish": ["Long Put", "Bear Put Spread", "Call Credit Spread"]
}
st.write("Recommended Strategies:")
st.write(strategies[trend])
