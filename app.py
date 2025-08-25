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

# =========================
# Fibonacci Levels Calculator with Trend Guidance
# =========================
st.subheader("📏 Fibonacci Levels Calculator & Trend Guidance")

# Allow any ticker
fib_ticker = st.text_input("Enter Ticker (Stock or Crypto, e.g. AAPL, BTC-USD)", value="AAPL")
fib_period = st.selectbox(
    "Select Period for Fibonacci Calculation",
    options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
    index=2
)

if fib_ticker:
    try:
        # Fetch historical data
        fib_data = yf.Ticker(fib_ticker).history(period=fib_period)
        if not fib_data.empty:
            high_price = fib_data['High'].max()
            low_price = fib_data['Low'].min()
            current_price = fib_data['Close'][-1]

            # Calculate Fibonacci levels
            levels = {
                "0.0% (Low)": low_price,
                "23.6%": low_price + 0.236 * (high_price - low_price),
                "38.2%": low_price + 0.382 * (high_price - low_price),
                "50.0%": low_price + 0.5 * (high_price - low_price),
                "61.8%": low_price + 0.618 * (high_price - low_price),
                "100% (High)": high_price
            }

            # Display levels in a table
            fib_df = pd.DataFrame(levels.items(), columns=["Level", "Price"])
            st.write(f"**{fib_ticker.upper()}** — High: {round(high_price,2)}, Low: {round(low_price,2)}, Current: {round(current_price,2)}")
            st.table(fib_df)

            # Plot Fibonacci levels on price chart
            fig_fib = go.Figure()
            fig_fib.add_trace(go.Scatter(
                x=fib_data.index,
                y=fib_data['Close'],
                mode="lines",
                name="Close Price"
            ))

            for level_name, price in levels.items():
                fig_fib.add_hline(
                    y=price,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=level_name,
                    annotation_position="top left"
                )

            fig_fib.update_layout(
                title=f"{fib_ticker.upper()} Fibonacci Levels",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_fib, use_container_width=True)

            # =========================
            # Trend & Trading Guidance
            # =========================
            st.markdown("### 📌 Trend Analysis & Guidance")
            
            trend = ""
            guidance = ""

            # Simple trend logic based on price relative to 50% Fibonacci level
            fib_50 = levels["50.0%"]
            fib_618 = levels["61.8%"]
            fib_382 = levels["38.2%"]

            if current_price > fib_618:
                trend = "🔥 Strong Uptrend"
                guidance = "Price is above 61.8% retracement. Consider bullish continuation or scaling in on dips."
            elif current_price > fib_50:
                trend = "📈 Moderate Uptrend"
                guidance = "Price is between 50% and 61.8%. Look for support near 50% retracement for long entries."
            elif current_price > fib_382:
                trend = "⚖️ Neutral / Consolidation"
                guidance = "Price is between 38.2% and 50%. Market may be consolidating. Watch for breakout or breakdown."
            else:
                trend = "🔻 Downtrend"
                guidance = "Price is below 38.2% retracement. Be cautious with longs; potential support near low."

            st.markdown(f"**Trend:** {trend}")
            st.markdown(f"**Guidance:** {guidance}")

        else:
            st.warning("No historical data found for this ticker.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
