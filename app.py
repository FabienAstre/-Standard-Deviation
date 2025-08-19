import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock & Crypto Screener", layout="wide")

st.title("📉 Undervalued Stock & Crypto Screener")
st.write("Analyze assets based on Z-score and standard deviation.")

tickers = st.text_input("Enter comma-separated tickers (e.g. AAPL, MSFT, BTC-USD)", value="AAPL, MSFT, BTC-USD")
period = st.selectbox("Select time period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y'])

tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

def analyze_asset(ticker, period='6mo'):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return None

    close_prices = data['Close']
    mean_price = close_prices.mean()
    std_dev = close_prices.std()
    current_price = close_prices[-1]
    z_score = (current_price - mean_price) / std_dev

    signal = ""
    if z_score < -2:
        signal = "🔻 Strongly Undervalued"
    elif z_score < -1:
        signal = "⚠️ Possibly Undervalued"
    elif z_score > 2:
        signal = "📈 Strongly Overvalued"
    elif z_score > 1:
        signal = "⚠️ Possibly Overvalued"
    else:
        signal = "✅ Normal Range"

    return {
        'ticker': ticker,
        'current_price': current_price,
        'mean_price': mean_price,
        'std_dev': std_dev,
        'z_score': z_score,
        'signal': signal,
        'prices': close_prices
    }

# Analyze and display
results = []

for ticker in tickers:
    result = analyze_asset(ticker, period)
    if result:
        results.append(result)
    else:
        st.warning(f"No data for {ticker}")

# Display table
if results:
    df = pd.DataFrame([{
        'Ticker': r['ticker'],
        'Current Price': round(r['current_price'], 2),
        'Mean Price': round(r['mean_price'], 2),
        'Std Dev': round(r['std_dev'], 2),
        'Z-Score': round(r['z_score'], 2),
        'Signal': r['signal']
    } for r in results])
    
    st.dataframe(df, use_container_width=True)

    # Plot charts
    st.subheader("📊 Price Charts")
    for r in results:
        with st.expander(f"{r['ticker']} Chart"):
            fig, ax = plt.subplots()
            ax.plot(r['prices'], label='Close Price')
            ax.axhline(r['mean_price'], color='green', linestyle='--', label='Mean')
            ax.axhline(r['mean_price'] + r['std_dev'], color='orange', linestyle='--', label='+1 STD')
            ax.axhline(r['mean_price'] - r['std_dev'], color='orange', linestyle='--', label='-1 STD')
            ax.axhline(r['mean_price'] + 2*r['std_dev'], color='red', linestyle='--', label='+2 STD')
            ax.axhline(r['mean_price'] - 2*r['std_dev'], color='red', linestyle='--', label='-2 STD')
            ax.set_title(f"{r['ticker']} Price Chart")
            ax.legend()
            st.pyplot(fig)
