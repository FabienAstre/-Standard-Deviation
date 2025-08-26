# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Stock & Crypto Screener", layout="wide")
st.title("📉 Stock & Crypto Screener + Fibonacci & Options Analysis")
st.write("""
Analyze assets based on **Z-score**, **RSI**, **Moving Averages**, **Fibonacci retracement**, 
and **Options sentiment** with actionable guidance.
""")

# =========================
# Sidebar / Inputs
# =========================
tickers_input = st.text_input(
    "Enter tickers (comma-separated, e.g., AAPL, MSFT, BTC-USD)", 
    value="AAPL, MSFT, BTC-USD"
)
period = st.selectbox(
    "Select price history period for analysis", 
    ['1mo', '3mo', '6mo', '1y', '2y', '5y']
)
show_only_undervalued = st.checkbox(
    "Show only Undervalued Assets (Z-score < -1)", 
    value=False
)
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
        'info': {'P/E': pe, 'P/B': pb, 'Div Yield': div_yield, 'Market Cap': market_cap}
    }

# =========================
# Analyze All Tickers
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
# Screener Table
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
# Candlestick Charts + Indicators
# =========================
st.subheader("📊 Price Charts & Indicators")
for r in results:
    with st.expander(f"{r['ticker']} Chart"):
        prices = r['prices']
        ma50 = prices.rolling(50).mean()
        ma200 = prices.rolling(200).mean()
        hist_data = r['data']
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="Candlestick"
        ))
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma50, mode="lines", name="MA 50", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma200, mode="lines", name="MA 200", line=dict(color="orange")))
        
        fig.add_hline(y=r['mean_price'], line_dash="dash", line_color="green", annotation_text="Mean")
        fig.add_hline(y=r['mean_price'] + r['std_dev'], line_dash="dash", line_color="orange", annotation_text="+1 STD")
        fig.add_hline(y=r['mean_price'] - r['std_dev'], line_dash="dash", line_color="orange", annotation_text="-1 STD")
        fig.add_hline(y=r['mean_price'] + 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="+2 STD")
        fig.add_hline(y=r['mean_price'] - 2*r['std_dev'], line_dash="dash", line_color="red", annotation_text="-2 STD")

        fig.update_layout(title=f"{r['ticker']} Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Explanation:**  
        - **MA50 / MA200**: Trend direction (bullish if MA50 > MA200)  
        - **Std Dev Bands**: Show volatility and potential support/resistance  
        - **Candlestick**: Short-term price patterns
        """)
# =========================
# Combined Recommendation Dashboard
# =========================
st.subheader("📊 Combined Trend & Recommendation Dashboard")
recommendations = []

for r in results:
    ticker = r['ticker']
    current_price = r['current_price']
    z_score = r['z_score']
    rsi = r['rsi']

    # Fibonacci trend
    try:
        fib_data = yf.Ticker(ticker).history(period=period)
        high_price = fib_data['High'].max()
        low_price = fib_data['Low'].min()
        fib_50 = low_price + 0.5*(high_price-low_price)
        fib_618 = low_price + 0.618*(high_price-low_price)
        fib_382 = low_price + 0.382*(high_price-low_price)
        if current_price > fib_618:
            fib_trend = 2
        elif current_price > fib_50:
            fib_trend = 1
        elif current_price > fib_382:
            fib_trend = 0
        else:
            fib_trend = -1
    except:
        fib_trend = 0

    # Z-score weight
    if z_score < -2: z_weight = 2
    elif z_score < -1: z_weight = 1
    elif z_score > 2: z_weight = -2
    elif z_score > 1: z_weight = -1
    else: z_weight = 0

    # RSI weight
    if rsi < 30: rsi_weight = 1
    elif rsi > 70: rsi_weight = -1
    else: rsi_weight = 0

    # Options weight
    try:
        opt_chain = yf.Ticker(ticker)
        if opt_chain.options:
            exp_date = opt_chain.options[0]
            chain = opt_chain.option_chain(exp_date)
            call_oi = chain.calls['openInterest'].sum()
            put_oi = chain.puts['openInterest'].sum()
            if call_oi > put_oi * 1.2: opt_weight = 1
            elif put_oi > call_oi * 1.2: opt_weight = -1
            else: opt_weight = 0
        else:
            opt_weight = 0
    except:
        opt_weight = 0

    total_score = z_weight + rsi_weight + fib_trend + opt_weight
    if total_score >= 3: recommendation = "🟢 Strong Buy"
    elif total_score == 2: recommendation = "🟢 Buy"
    elif total_score in [0,1]: recommendation = "🟡 Hold"
    else: recommendation = "🔴 Sell"

    recommendations.append({
        'Ticker': ticker,
        'Z-score': round(z_score,2),
        'RSI': round(rsi,2),
        'Fib Trend': fib_trend,
        'Options Sentiment': opt_weight,
        'Combined Score': total_score,
        'Recommendation': recommendation
    })

rec_df = pd.DataFrame(recommendations)
st.dataframe(rec_df, use_container_width=True)
# =========================
# Fibonacci Levels + Trend Guidance
# =========================
st.subheader("📏 Fibonacci Levels & Trend Guidance")
fib_ticker = st.text_input("Ticker for Fibonacci analysis", value="AAPL", key="fib_ticker")
fib_period = st.selectbox("Period for Fibonacci", ['1mo','3mo','6mo','1y','2y','5y'], index=2, key="fib_period")

if fib_ticker:
    try:
        fib_data = yf.Ticker(fib_ticker).history(period=fib_period)
        if not fib_data.empty:
            high_price = fib_data['High'].max()
            low_price = fib_data['Low'].min()
            current_price = fib_data['Close'][-1]

            levels = {
                "0.0% (Low)": low_price,
                "23.6%": low_price + 0.236*(high_price-low_price),
                "38.2%": low_price + 0.382*(high_price-low_price),
                "50.0%": low_price + 0.5*(high_price-low_price),
                "61.8%": low_price + 0.618*(high_price-low_price),
                "100% (High)": high_price
            }

            fib_df = pd.DataFrame(levels.items(), columns=["Level", "Price"])
            st.write(f"**{fib_ticker.upper()}** — High: {round(high_price,2)}, Low: {round(low_price,2)}, Current: {round(current_price,2)}")
            st.table(fib_df)

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
                guidance = "Price above 61.8%. Consider bullish continuation."
            elif current_price > fib_50:
                trend = "📈 Moderate Uptrend"
                guidance = "Price between 50%-61.8%. Support may hold near 50% retracement."
            elif current_price > fib_382:
                trend = "⚖️ Neutral / Consolidation"
                guidance = "Price between 38.2%-50%. Watch for breakout."
            else:
                trend = "🔻 Downtrend"
                guidance = "Price below 38.2%. Be cautious with longs."

            st.markdown(f"**Trend:** {trend}")
            st.markdown(f"**Guidance:** {guidance}")
        else:
            st.warning("No data for Fibonacci.")
    except Exception as e:
        st.error(f"Error: {e}")
# =========================
# Full Options Dashboard
# =========================
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Options Dashboard & Simulator", layout="wide")
st.title("📊 Options Dashboard & Payoff Simulator")

# =========================
# Inputs
# =========================
col1, col2 = st.columns([1,1])
with col1:
    ticker = st.text_input("Ticker for analysis", value="AAPL", key="ticker_input")
with col2:
    use_live_price = st.checkbox("Auto-fetch current price", value=True, key="use_live_price")

# =========================
# Fetch current price
# =========================
def get_current_price(ticker:str, fallback=100.0):
    try:
        h = yf.Ticker(ticker).history(period="5d")
        if not h.empty:
            return float(h["Close"][-1])
    except:
        pass
    return fallback

S0 = get_current_price(ticker) if use_live_price else 100.0
st.markdown(f"**Seed Price (S₀): `{round(S0,2)}` — baseline for payoff chart**")


# =========================
# Options Analysis with Greeks
# =========================
st.subheader("📊 Options Indicators & Greeks")
opt_ticker = st.text_input("Ticker for options analysis", value="AAPL", key="opt_ticker")

if opt_ticker:
    try:
        opt_chain = yf.Ticker(opt_ticker)
        if not opt_chain.options:
            st.info("No options available for this ticker.")
        else:
            # Select expiration
            exp_date = st.selectbox("Select expiration date", options=opt_chain.options, key="opt_exp_date")
            chain = opt_chain.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            # Display top 5 by volume
            st.markdown("### 🔹 Top 5 Call Options by Volume")
            st.dataframe(calls.sort_values("volume", ascending=False).head(5))
            st.markdown("### 🔹 Top 5 Put Options by Volume")
            st.dataframe(puts.sort_values("volume", ascending=False).head(5))

            # Calculate Greeks (approximate using Black-Scholes if needed)
            # Using yfinance, we usually have: impliedVolatility, delta, gamma, theta, vega, rho
            # If missing, we'll set as None
            def extract_greeks(df):
                greeks_list = []
                for _, row in df.head(5).iterrows():
                    greeks_list.append({
                        'Contract': row['contractSymbol'],
                        'Strike': row['strike'],
                        'Last Price': row['lastPrice'],
                        'Bid': row['bid'],
                        'Ask': row['ask'],
                        'Volume': row['volume'],
                        'Open Interest': row['openInterest'],
                        'Implied Volatility': row.get('impliedVolatility', None),
                        'Delta': row.get('delta', None),
                        'Gamma': row.get('gamma', None),
                        'Theta': row.get('theta', None),
                        'Vega': row.get('vega', None),
                        'Rho': row.get('rho', None),
                    })
                return pd.DataFrame(greeks_list)

            st.markdown("### 📈 Top Calls with Greeks")
            calls_greeks = extract_greeks(calls)
            st.dataframe(calls_greeks)

            st.markdown("### 📉 Top Puts with Greeks")
            puts_greeks = extract_greeks(puts)
            st.dataframe(puts_greeks)

            # Explain Greeks
            st.markdown("""
### 💡 Greek Explanation
- **Delta (Δ):** How much the option price moves for $1 move in stock.  
- **Gamma (Γ):** How much Delta changes if stock moves $1.  
- **Theta (Θ):** Time decay of the option per day. Negative for buyers.  
- **Vega (ν):** Sensitivity to implied volatility. Higher IV → option price rises.  
- **Rho (ρ):** Sensitivity to interest rate changes.
""")

            # Market bias
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            if call_oi > put_oi * 1.2:
                st.success("Market Bias: Bullish (Calls dominate Open Interest)")
            elif put_oi > call_oi * 1.2:
                st.error("Market Bias: Bearish (Puts dominate Open Interest)")
            else:
                st.info("Market Bias: Neutral")

    except Exception as e:
        st.error(f"Error fetching options: {e}")


# =========================
# Options Education
# =========================
st.subheader("📘 Options Basics & Explanation")
st.markdown("""
**Options come in two main types:**

- **Call Option** → Right (not obligation) to **BUY** a stock at strike price before expiration.
- **Put Option** → Right (not obligation) to **SELL** a stock at strike price before expiration.

**Quick Tips:**
- Calls = Bullish
- Puts = Bearish
- Covered Call: Collect premium, risk losing shares
- Cash-Secured Put: Collect premium, potential to buy stock at discount
""")

# =========================
# Payoff Simulator
# =========================
st.subheader("🎓 Options Payoff Simulator")
colA, colB = st.columns([1,1])
with colA:
    strategy = st.selectbox(
        "Choose a strategy",
        ["Long Call", "Long Put", "Covered Call", "Cash-Secured Put", "Bull Call Spread"],
        key="strategy"
    )
with colB:
    rng = st.slider(
        "Underlying price range at expiration (Sₜ)",
        min_value=max(1,int(S0*0.2)),
        max_value=int(S0*2.0),
        value=(int(S0*0.6), int(S0*1.4)),
        step=1
    )
S_grid = np.linspace(rng[0], rng[1], 300)

K = st.number_input("Strike Price (K)", value=int(S0), step=1)
premium = st.number_input("Option Premium ($)", value=5.0, step=0.1)
K2 = None
if strategy == "Bull Call Spread":
    K2 = st.number_input("Second Strike Price (K2)", value=int(S0+10), step=1)

# Payoff calculation
if strategy == "Long Call":
    payoff = np.maximum(S_grid - K,0) - premium
    breakeven = K + premium
    max_loss, max_profit = -premium, "Unlimited"
elif strategy == "Long Put":
    payoff = np.maximum(K - S_grid,0) - premium
    breakeven = K - premium
    max_loss, max_profit = -premium, K - premium
elif strategy == "Covered Call":
    payoff = (S_grid - S0) + premium - np.maximum(S_grid - K,0)
    breakeven = S0 - premium
    max_loss, max_profit = -(S0 - premium), (K - S0 + premium)
elif strategy == "Cash-Secured Put":
    payoff = premium - np.maximum(K - S_grid,0)
    breakeven = K - premium
    max_loss, max_profit = -(K - premium), premium
elif strategy == "Bull Call Spread":
    payoff = np.maximum(S_grid - K,0) - np.maximum(S_grid - K2,0) - premium
    breakeven = K + premium
    max_loss = -premium
    max_profit = (K2 - K - premium)
else:
    payoff = np.zeros_like(S_grid)
    breakeven, max_loss, max_profit = None, None, None

# Plot payoff
fig = go.Figure()
fig.add_trace(go.Scatter(x=S_grid, y=payoff, mode='lines', name="Payoff", line=dict(color="royalblue")))
if breakeven:
    fig.add_vline(x=breakeven, line=dict(dash="dot", color="green"))
    fig.add_annotation(x=breakeven, y=0, text="Breakeven", showarrow=True, arrowhead=2)
fig.update_layout(title=f"{strategy} Payoff at Expiration", xaxis_title="Underlying Price Sₜ", yaxis_title="Profit / Loss")
st.plotly_chart(fig, use_container_width=True)

# =========================
# Recommendation
# =========================
st.subheader("💡 Recommendation & Insights")
st.write(f"Market Bias: {market_bias}")
recommendation = ""
advice = ""

if strategy == "Long Call":
    if S0 < K:
        recommendation = "🟢 Buy Call"
        advice = "Expect stock to rise above strike; favorable if bullish bias."
    else:
        recommendation = "⚠️ Call is In-The-Money"
        advice = "Premium may be high; check risk/reward."
elif strategy == "Long Put":
    if S0 > K:
        recommendation = "🟢 Buy Put"
        advice = "Expect stock to drop below strike; favorable if bearish bias."
    else:
        recommendation = "⚠️ Put is In-The-Money"
        advice = "Premium may be high; check risk/reward."
elif strategy == "Covered Call":
    recommendation = "🟡 Sell Covered Call"
    advice = f"Collect premium; stock may be called away. Market bias: {market_bias}."
elif strategy == "Cash-Secured Put":
    recommendation = "🟢 Sell Cash-Secured Put"
    advice = f"Collect premium; potential to buy stock at discount. Market bias: {market_bias}."
elif strategy == "Bull Call Spread":
    recommendation = "🟢 Bull Call Spread"
    advice = f"Moderate bullish view; limits upside and risk. Market bias: {market_bias}."

st.markdown(f"### Recommendation: {recommendation}")
st.markdown(f"**Advice:** {advice}")
st.write(f"**Breakeven:** {breakeven}, **Max Profit:** {max_profit}, **Max Loss:** {max_loss}")

# =========================
# High-Probability Option Picks
# =========================
st.subheader("📈 High-Probability Option Picks")

if calls is not None and puts is not None:
    # Approximate probability using Delta (calls) as a proxy
    calls['prob_ITM'] = calls.get('delta', 0.5)  # if delta missing, assume 50%
    puts['prob_ITM'] = abs(puts.get('delta', -0.5))  # absolute delta for puts

    # Filter for high probability (e.g., delta > 0.7)
    high_prob_calls = calls[calls['prob_ITM'] >= 0.7].sort_values('prob_ITM', ascending=False)
    high_prob_puts = puts[puts['prob_ITM'] >= 0.7].sort_values('prob_ITM', ascending=False)

    st.markdown("### 🔹 High-Probability Calls (Delta > 0.7)")
    if not high_prob_calls.empty:
        st.dataframe(high_prob_calls[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest','prob_ITM']])
    else:
        st.write("No high-probability calls found.")

    st.markdown("### 🔹 High-Probability Puts (Delta > 0.7)")
    if not high_prob_puts.empty:
        st.dataframe(high_prob_puts[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest','prob_ITM']])
    else:
        st.write("No high-probability puts found.")

    # Suggestion based on market bias
    suggestion = ""
    if market_bias == "Bullish" and not high_prob_calls.empty:
        top_call = high_prob_calls.iloc[0]
        suggestion = f"🟢 Consider buying {top_call['contractSymbol']} (Call) with Δ≈{round(top_call['prob_ITM'],2)}"
    elif market_bias == "Bearish" and not high_prob_puts.empty:
        top_put = high_prob_puts.iloc[0]
        suggestion = f"🔴 Consider buying {top_put['contractSymbol']} (Put) with Δ≈{round(top_put['prob_ITM'],2)}"
    elif market_bias == "Neutral":
        suggestion = "🟡 Market is neutral: consider selling Covered Calls or Cash-Secured Puts for income."

    st.markdown(f"### 💡 Suggested High-Probability Trade: {suggestion}")
else:
    st.info("Options data not available to suggest high-probability trades.")

