# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock & Crypto Screener", layout="wide")
st.title("📉 Stock & Crypto Screener + Fibonacci & Options Analysis")
st.write("""
Analyze assets based on **Z-score**, **RSI**, **Moving Averages**, **Fibonacci retracement**, 
and **Options sentiment** with actionable guidance.
""")

# =========================
# Sidebar / User Input
# =========================
tickers_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL, MSFT, BTC-USD)", value="AAPL, MSFT, BTC-USD")
period = st.selectbox("Select price history period for analysis", ['1mo', '3mo', '6mo', '1y', '2y', '5y'])
show_only_undervalued = st.checkbox("Show only Undervalued Assets (Z-score < -1)", value=False)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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
    """
    Fetch historical data and compute metrics:
    - Current price, mean, std dev, Z-score
    - RSI
    - Fundamental metrics (P/E, P/B, Div Yield, Market Cap)
    """
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return None
    except:
        return None

    close_prices = data['Close']
    mean_price = close_prices.mean()
    std_dev = close_prices.std()
    current_price = close_prices[-1]
    z_score = (current_price - mean_price) / std_dev
    rsi = calculate_rsi(close_prices).iloc[-1]

    # Signal interpretation
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
    info = yf.Ticker(ticker).info
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
        'data': data,
        'info': {
            'P/E': pe,
            'P/B': pb,
            'Div Yield': div_yield,
            'Market Cap': market_cap
        }
    }

# =========================
# Analyze all tickers
# =========================
results = []
for ticker in tickers:
    r = analyze_asset(ticker, period)
    if r:
        if show_only_undervalued and r['z_score'] > -1:
            continue
        results.append(r)
    else:
        st.warning(f"No data available for {ticker}")

# =========================
# Display Analysis Table
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
    
    st.subheader("📑 Screener Table")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="screener_results.csv")

# =========================
# Charts + Moving Averages + Std Dev Bands
# =========================
st.subheader("📊 Interactive Price Charts")
for r in results:
    with st.expander(f"{r['ticker']} Chart"):
        prices = r['prices']
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        
        fig = go.Figure()
        hist_data = r['data']

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="Candlestick"
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, mode="lines", name="MA 50", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma200, mode="lines", name="MA 200", line=dict(color="orange")))

        # Std dev bands
        fig.add_hline(y=r['mean_price'], line_dash="dash", line_color="green", annotation_text="Mean")
        fig.add_hline(y=r['mean_price'] + r['std_dev'], line_dash="dash", line_color="orange", annotation_text="+1 STD")
        fig.add_hline(y=r['mean_price'] - r['std_dev'], line_dash="dash", line_color="orange", annotation_text="-1 STD")
        fig.add_hline(y=r['mean_price'] + 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="+2 STD")
        fig.add_hline(y=r['mean_price'] - 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="-2 STD")

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Explanation:**  
        - **MA50 / MA200**: Trend indicator (bullish if MA50 > MA200)  
        - **Std Dev Bands**: Shows volatility and potential support/resistance zones  
        - **Candlestick Patterns**: Help detect short-term reversal signals
        """)

# =========================
# Fibonacci Levels + Trend Guidance
# =========================
st.subheader("📏 Fibonacci Levels & Trend Guidance")
fib_ticker = st.text_input("Enter ticker for Fibonacci analysis (stock/crypto)", value="AAPL")
fib_period = st.selectbox("Select period for Fibonacci", ['1mo','3mo','6mo','1y','2y','5y'], index=2)

if fib_ticker:
    try:
        fib_data = yf.Ticker(fib_ticker).history(period=fib_period)
        if not fib_data.empty:
            high_price = fib_data['High'].max()
            low_price = fib_data['Low'].min()
            current_price = fib_data['Close'][-1]

            # Fibonacci levels
            levels = {
                "0.0% (Low)": low_price,
                "23.6%": low_price + 0.236*(high_price - low_price),
                "38.2%": low_price + 0.382*(high_price - low_price),
                "50.0%": low_price + 0.5*(high_price - low_price),
                "61.8%": low_price + 0.618*(high_price - low_price),
                "100% (High)": high_price
            }

            fib_df = pd.DataFrame(levels.items(), columns=["Level", "Price"])
            st.write(f"**{fib_ticker.upper()}** — High: {round(high_price,2)}, Low: {round(low_price,2)}, Current: {round(current_price,2)}")
            st.table(fib_df)

            # Plot Fibonacci
            fig_fib = go.Figure()
            fig_fib.add_trace(go.Scatter(x=fib_data.index, y=fib_data['Close'], mode="lines", name="Close Price"))
            for lvl_name, price in levels.items():
                fig_fib.add_hline(y=price, line_dash="dash", line_color="orange", annotation_text=lvl_name, annotation_position="top left")
            fig_fib.update_layout(title=f"{fib_ticker.upper()} Fibonacci Levels", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_fib, use_container_width=True)

            # Trend guidance
            fib_50 = levels["50.0%"]
            fib_618 = levels["61.8%"]
            fib_382 = levels["38.2%"]

            if current_price > fib_618:
                trend = "🔥 Strong Uptrend"
                guidance = "Price above 61.8%. Consider bullish continuation or scaling in on dips."
            elif current_price > fib_50:
                trend = "📈 Moderate Uptrend"
                guidance = "Price between 50%-61.8%. Support may hold near 50% retracement."
            elif current_price > fib_382:
                trend = "⚖️ Neutral / Consolidation"
                guidance = "Price between 38.2%-50%. Market consolidating; watch for breakout/breakdown."
            else:
                trend = "🔻 Downtrend"
                guidance = "Price below 38.2%. Be cautious with longs; support near low."

            st.markdown(f"**Trend:** {trend}")
            st.markdown(f"**Guidance:** {guidance}")

        else:
            st.warning("No historical data for Fibonacci calculation.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# =========================
# Options Indicators
# =========================
st.subheader("📊 Options Indicators (Long/Short/Covered Call Guidance)")
opt_ticker = st.text_input("Enter ticker for options analysis", value="AAPL")

if opt_ticker:
    try:
        opt_chain = yf.Ticker(opt_ticker)
        if not opt_chain.options:
            st.info("No options available for this ticker.")
        else:
            exp_date = st.selectbox("Select expiration date", options=opt_chain.options)
            chain = opt_chain.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            # Top options by volume
            st.write("Top 5 Call Options by Volume")
            st.dataframe(calls.sort_values("volume", ascending=False).head(5))
            st.write("Top 5 Put Options by Volume")
            st.dataframe(puts.sort_values("volume", ascending=False).head(5))

            # Simple sentiment
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            if call_oi > put_oi * 1.2:
                st.success("Market Bias: Bullish (Calls dominate Open Interest)")
            elif put_oi > call_oi * 1.2:
                st.error("Market Bias: Bearish (Puts dominate Open Interest)")
            else:
                st.info("Market Bias: Neutral")

            # Covered call guidance
            avg_call_premium = calls['lastPrice'].mean()
            st.markdown(f"💡 Covered Call Tip: Avg Call Premium = ${round(avg_call_premium,2)}")
            st.markdown("If you hold the stock, selling calls near strike price above current price generates income.")

    except Exception as e:
        st.error(f"Error fetching options data: {e}")

# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock & Crypto Screener", layout="wide")
st.title("📉 Stock & Crypto Screener + Fibonacci & Options Analysis")
st.write("""
Analyze assets based on **Z-score**, **RSI**, **Moving Averages**, **Fibonacci retracement**, 
and **Options sentiment** with actionable guidance.
""")

# =========================
# Sidebar / User Input
# =========================
tickers_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL, MSFT, BTC-USD)", value="AAPL, MSFT, BTC-USD")
period = st.selectbox("Select price history period for analysis", ['1mo', '3mo', '6mo', '1y', '2y', '5y'])
show_only_undervalued = st.checkbox("Show only Undervalued Assets (Z-score < -1)", value=False)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

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
    """
    Fetch historical data and compute metrics:
    - Current price, mean, std dev, Z-score
    - RSI
    - Fundamental metrics (P/E, P/B, Div Yield, Market Cap)
    """
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return None
    except:
        return None

    close_prices = data['Close']
    mean_price = close_prices.mean()
    std_dev = close_prices.std()
    current_price = close_prices[-1]
    z_score = (current_price - mean_price) / std_dev
    rsi = calculate_rsi(close_prices).iloc[-1]

    # Signal interpretation
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
    info = yf.Ticker(ticker).info
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
        'data': data,
        'info': {
            'P/E': pe,
            'P/B': pb,
            'Div Yield': div_yield,
            'Market Cap': market_cap
        }
    }

# =========================
# Analyze all tickers
# =========================
results = []
for ticker in tickers:
    r = analyze_asset(ticker, period)
    if r:
        if show_only_undervalued and r['z_score'] > -1:
            continue
        results.append(r)
    else:
        st.warning(f"No data available for {ticker}")

# =========================
# Display Analysis Table
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
    
    st.subheader("📑 Screener Table")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="screener_results.csv")

# =========================
# Charts + Moving Averages + Std Dev Bands
# =========================
st.subheader("📊 Interactive Price Charts")
for r in results:
    with st.expander(f"{r['ticker']} Chart"):
        prices = r['prices']
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        
        fig = go.Figure()
        hist_data = r['data']

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="Candlestick"
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, mode="lines", name="MA 50", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma200, mode="lines", name="MA 200", line=dict(color="orange")))

        # Std dev bands
        fig.add_hline(y=r['mean_price'], line_dash="dash", line_color="green", annotation_text="Mean")
        fig.add_hline(y=r['mean_price'] + r['std_dev'], line_dash="dash", line_color="orange", annotation_text="+1 STD")
        fig.add_hline(y=r['mean_price'] - r['std_dev'], line_dash="dash", line_color="orange", annotation_text="-1 STD")
        fig.add_hline(y=r['mean_price'] + 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="+2 STD")
        fig.add_hline(y=r['mean_price'] - 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="-2 STD")

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Explanation:**  
        - **MA50 / MA200**: Trend indicator (bullish if MA50 > MA200)  
        - **Std Dev Bands**: Shows volatility and potential support/resistance zones  
        - **Candlestick Patterns**: Help detect short-term reversal signals
        """)

# =========================
# Fibonacci Levels + Trend Guidance
# =========================
st.subheader("📏 Fibonacci Levels & Trend Guidance")
fib_ticker = st.text_input("Enter ticker for Fibonacci analysis (stock/crypto)", value="AAPL")
fib_period = st.selectbox("Select period for Fibonacci", ['1mo','3mo','6mo','1y','2y','5y'], index=2)

if fib_ticker:
    try:
        fib_data = yf.Ticker(fib_ticker).history(period=fib_period)
        if not fib_data.empty:
            high_price = fib_data['High'].max()
            low_price = fib_data['Low'].min()
            current_price = fib_data['Close'][-1]

            # Fibonacci levels
            levels = {
                "0.0% (Low)": low_price,
                "23.6%": low_price + 0.236*(high_price - low_price),
                "38.2%": low_price + 0.382*(high_price - low_price),
                "50.0%": low_price + 0.5*(high_price - low_price),
                "61.8%": low_price + 0.618*(high_price - low_price),
                "100% (High)": high_price
            }

            fib_df = pd.DataFrame(levels.items(), columns=["Level", "Price"])
            st.write(f"**{fib_ticker.upper()}** — High: {round(high_price,2)}, Low: {round(low_price,2)}, Current: {round(current_price,2)}")
            st.table(fib_df)

            # Plot Fibonacci
            fig_fib = go.Figure()
            fig_fib.add_trace(go.Scatter(x=fib_data.index, y=fib_data['Close'], mode="lines", name="Close Price"))
            for lvl_name, price in levels.items():
                fig_fib.add_hline(y=price, line_dash="dash", line_color="orange", annotation_text=lvl_name, annotation_position="top left")
            fig_fib.update_layout(title=f"{fib_ticker.upper()} Fibonacci Levels", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_fib, use_container_width=True)

            # Trend guidance
            fib_50 = levels["50.0%"]
            fib_618 = levels["61.8%"]
            fib_382 = levels["38.2%"]

            if current_price > fib_618:
                trend = "🔥 Strong Uptrend"
                guidance = "Price above 61.8%. Consider bullish continuation or scaling in on dips."
            elif current_price > fib_50:
                trend = "📈 Moderate Uptrend"
                guidance = "Price between 50%-61.8%. Support may hold near 50% retracement."
            elif current_price > fib_382:
                trend = "⚖️ Neutral / Consolidation"
                guidance = "Price between 38.2%-50%. Market consolidating; watch for breakout/breakdown."
            else:
                trend = "🔻 Downtrend"
                guidance = "Price below 38.2%. Be cautious with longs; support near low."

            st.markdown(f"**Trend:** {trend}")
            st.markdown(f"**Guidance:** {guidance}")

        else:
            st.warning("No historical data for Fibonacci calculation.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# =========================
# Options Indicators
# =========================
st.subheader("📊 Options Indicators (Long/Short/Covered Call Guidance)")
opt_ticker = st.text_input("Enter ticker for options analysis", value="AAPL")

if opt_ticker:
    try:
        opt_chain = yf.Ticker(opt_ticker)
        if not opt_chain.options:
            st.info("No options available for this ticker.")
        else:
            exp_date = st.selectbox("Select expiration date", options=opt_chain.options)
            chain = opt_chain.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            # Top options by volume
            st.write("Top 5 Call Options by Volume")
            st.dataframe(calls.sort_values("volume", ascending=False).head(5))
            st.write("Top 5 Put Options by Volume")
            st.dataframe(puts.sort_values("volume", ascending=False).head(5))

            # Simple sentiment
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            if call_oi > put_oi * 1.2:
                st.success("Market Bias: Bullish (Calls dominate Open Interest)")
            elif put_oi > call_oi * 1.2:
                st.error("Market Bias: Bearish (Puts dominate Open Interest)")
            else:
                st.info("Market Bias: Neutral")

            # Covered call guidance
            avg_call_premium = calls['lastPrice'].mean()
            st.markdown(f"💡 Covered Call Tip: Avg Call Premium = ${round(avg_call_premium,2)}")
            st.markdown("If you hold the stock, selling calls near strike price above current price generates income.")

    except Exception as e:
        st.error(f"Error fetching options data: {e}")
