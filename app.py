import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock & Crypto Screener", layout="wide")

st.title("📉 Undervalued Stock & Crypto Screener")
st.write("Analyze assets based on Z-score, RSI, moving averages, and fundamentals.")

# =========================
# Sidebar / UX
# =========================
tickers = st.text_input("Enter comma-separated tickers (e.g. AAPL, MSFT, BTC-USD)", value="AAPL, MSFT, BTC-USD")
period = st.selectbox("Select time period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y'])
show_only_undervalued = st.checkbox("Show only Undervalued Assets (Z-score < -1)", value=False)

tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

# =========================
# Helper Functions
# =========================
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_asset(ticker, period='6mo'):
    try:
        data = yf.Ticker(ticker).history(period=period)
    except Exception:
        return None
    
    if data.empty:
        return None

    close_prices = data['Close']
    mean_price = close_prices.mean()
    std_dev = close_prices.std()
    current_price = close_prices[-1]
    z_score = (current_price - mean_price) / std_dev
    rsi = calculate_rsi(close_prices).iloc[-1]

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

    # Fundamentals
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    pe = info.get("trailingPE", None)
    pb = info.get("priceToBook", None)
    div_yield = info.get("dividendYield", None)
    market_cap = info.get("marketCap", None)

    return {
        'ticker': ticker,
        'current_price': current_price,
        'mean_price': mean_price,
        'std_dev': std_dev,
        'z_score': z_score,
        'rsi': rsi,
        'signal': signal,
        'prices': close_prices,
        'info': {
            'P/E': pe,
            'P/B': pb,
            'Div Yield': div_yield,
            'Market Cap': market_cap
        }
    }

# =========================
# Analyze
# =========================
results = []
for ticker in tickers:
    result = analyze_asset(ticker, period)
    if result:
        if show_only_undervalued and result["z_score"] > -1:
            continue
        results.append(result)
    else:
        st.warning(f"No data for {ticker}")

# =========================
# Display Table
# =========================
if results:
    df = pd.DataFrame([{
        'Ticker': r['ticker'],
        'Current Price': round(r['current_price'], 2),
        'Mean Price': round(r['mean_price'], 2),
        'Std Dev': round(r['std_dev'], 2),
        'Z-Score': round(r['z_score'], 2),
        'RSI': round(r['rsi'], 2),
        'Signal': r['signal'],
        'P/E': r['info']['P/E'],
        'P/B': r['info']['P/B'],
        'Div Yield': r['info']['Div Yield'],
        'Market Cap': r['info']['Market Cap']
    } for r in results])
    
    st.subheader("📑 Analysis Table")
    st.dataframe(df, use_container_width=True)

    # Download button
    st.download_button("Download Results (CSV)", data=df.to_csv(index=False), file_name="screener_results.csv")

    # =========================
    # Charts
    # =========================
    st.subheader("📊 Interactive Charts")
    for r in results:
        with st.expander(f"{r['ticker']} Chart"):
            prices = r['prices']
            ma50 = prices.rolling(50).mean()
            ma200 = prices.rolling(200).mean()

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=prices.index,
                open=yf.Ticker(r['ticker']).history(period=period)['Open'],
                high=yf.Ticker(r['ticker']).history(period=period)['High'],
                low=yf.Ticker(r['ticker']).history(period=period)['Low'],
                close=prices,
                name="Candlestick"
            ))

            # Moving Averages
            fig.add_trace(go.Scatter(x=prices.index, y=ma50, mode="lines", name="MA 50", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=prices.index, y=ma200, mode="lines", name="MA 200", line=dict(color="orange")))

            # Mean & Std Dev bands
            fig.add_hline(y=r['mean_price'], line_dash="dash", line_color="green", annotation_text="Mean")
            fig.add_hline(y=r['mean_price'] + r['std_dev'], line_dash="dash", line_color="orange", annotation_text="+1 STD")
            fig.add_hline(y=r['mean_price'] - r['std_dev'], line_dash="dash", line_color="orange", annotation_text="-1 STD")
            fig.add_hline(y=r['mean_price'] + 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="+2 STD")
            fig.add_hline(y=r['mean_price'] - 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="-2 STD")

            fig.update_layout(title=f"{r['ticker']} Price Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
