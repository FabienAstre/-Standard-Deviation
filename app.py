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
# Full Options Dashboard & Payoff Simulator
# =========================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Options Dashboard & Simulator", layout="wide")
st.title("📊 Options Dashboard & Payoff Simulator")

# =========================
# 1️⃣ Inputs
# =========================
col1, col2 = st.columns([1,1])
with col1:
    ticker = st.text_input("Ticker for analysis", value="AAPL", key="ticker_input")
with col2:
    use_live_price = st.checkbox("Auto-fetch current price", value=True, key="use_live_price")

# =========================
# 2️⃣ Fetch Current Price
# =========================
def get_current_price(ticker:str, fallback=100.0):
    """Fetch last close price of the stock, fallback if unavailable"""
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
# 3️⃣ Fetch Options Chain & Greeks
# =========================
st.subheader("📊 Options Chain & Greeks")
calls, puts = None, None
market_bias = "Neutral"

try:
    opt_chain = yf.Ticker(ticker)
    if opt_chain.options:
        # Select expiration
        exp_date = st.selectbox("Select expiration date", opt_chain.options)
        chain = opt_chain.option_chain(exp_date)
        calls = chain.calls
        puts = chain.puts

        # Display top options by volume
        st.markdown("### 🔹 Top 5 Call Options by Volume")
        st.dataframe(calls.sort_values("volume", ascending=False).head(5))
        st.markdown("### 🔹 Top 5 Put Options by Volume")
        st.dataframe(puts.sort_values("volume", ascending=False).head(5))

        # Market bias based on open interest
        call_oi = calls['openInterest'].sum()
        put_oi = puts['openInterest'].sum()
        if call_oi > put_oi * 1.2:
            market_bias = "Bullish"
        elif put_oi > call_oi * 1.2:
            market_bias = "Bearish"

        # Extract Greeks for top 5 options
        def extract_greeks(df):
            """Returns a DataFrame with Greeks and key data"""
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
        st.dataframe(extract_greeks(calls))

        st.markdown("### 📉 Top Puts with Greeks")
        st.dataframe(extract_greeks(puts))

        st.markdown("""
### 💡 Greek Explanation
- **Delta (Δ):** Expected price change in option per $1 move in stock. Also ≈ probability option finishes ITM.
- **Gamma (Γ):** Rate of change of Delta if stock moves $1.
- **Theta (Θ):** Daily time decay of option; negative for buyers.
- **Vega (ν):** Sensitivity to implied volatility.
- **Rho (ρ):** Sensitivity to interest rates.
""")

    else:
        st.info("No options available for this ticker.")

except Exception as e:
    st.warning(f"Error fetching options: {e}")

st.write(f"**Market Bias:** {market_bias}")

# =========================
# 4️⃣ Options Education
# =========================
st.subheader("📘 Options Basics")
st.markdown("""
**Two Main Types of Options:**
- **Call Option:** Right to BUY at strike price → bullish strategy.
- **Put Option:** Right to SELL at strike price → bearish strategy.

**Quick Tips:**
- Calls = Bet stock will rise.
- Puts = Bet stock will fall.
- Covered Call: Sell calls to collect premium; risk losing stock if it rises.
- Cash-Secured Put: Sell puts to collect premium; may have to buy stock at strike.
""")

# =========================
# 5️⃣ Payoff Simulator
# =========================
st.subheader("🎓 Payoff Simulator")

colA, colB = st.columns([1,1])
with colA:
    strategy = st.selectbox(
        "Choose a strategy",
        ["Long Call", "Long Put", "Covered Call", "Cash-Secured Put", "Bull Call Spread"]
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

# Calculate payoff
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
# 6️⃣ Recommendation & Insights
# =========================
st.subheader("💡 Recommendation & Insights")
recommendation, advice = "", ""

if strategy == "Long Call":
    if S0 < K:
        recommendation = "🟢 Buy Call"
        advice = "Expect stock to rise above strike; favorable if bullish."
    else:
        recommendation = "⚠️ Call is In-The-Money"
        advice = "Premium may be high; check risk/reward."
elif strategy == "Long Put":
    if S0 > K:
        recommendation = "🟢 Buy Put"
        advice = "Expect stock to drop below strike; favorable if bearish."
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
    advice = f"Moderate bullish view; limits upside/risk. Market bias: {market_bias}."

st.markdown(f"### Recommendation: {recommendation}")
st.markdown(f"**Advice:** {advice}")
st.write(f"**Breakeven:** {breakeven}, **Max Profit:** {max_profit}, **Max Loss:** {max_loss}")

# =========================
# 7️⃣ High-Probability Option Picks
# =========================
st.subheader("📈 High-Probability Option Picks")
if calls is not None and puts is not None:
    # Approximate probability using Delta
    calls['prob_ITM'] = calls.get('delta', 0.5)
    puts['prob_ITM'] = abs(puts.get('delta', 0.5))

    high_prob_calls = calls[calls['prob_ITM'] >= 0.7].sort_values('prob_ITM', ascending=False)
    high_prob_puts = puts[puts['prob_ITM'] >= 0.7].sort_values('prob_ITM', ascending=False)

    st.markdown("### 🔹 High-Probability Calls (Delta ≥ 0.7)")
    st.dataframe(high_prob_calls[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest','prob_ITM']])
    st.markdown("### 🔹 High-Probability Puts (Delta ≥ 0.7)")
    st.dataframe(high_prob_puts[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest','prob_ITM']])

    # Suggest top actionable trade
    suggestion = ""
    if market_bias == "Bullish" and not high_prob_calls.empty:
        top_call = high_prob_calls.iloc[0]
        suggestion = f"🟢 Buy {top_call['contractSymbol']} (Call) Δ≈{round(top_call['prob_ITM'],2)}"
    elif market_bias == "Bearish" and not high_prob_puts.empty:
        top_put = high_prob_puts.iloc[0]
        suggestion = f"🔴 Buy {top_put['contractSymbol']} (Put) Δ≈{round(top_put['prob_ITM'],2)}"
    else:
        suggestion = "🟡 Market neutral: consider selling Covered Calls or Cash-Secured Puts for income."

    st.markdown(f"### 💡 Suggested High-Probability Trade: {suggestion}")
else:
    st.info("Options data not available for high-probability suggestions.")




# app.py — Options Decision Matrix & Strategy Recommender
# Built for Streamlit

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Options Decision Matrix", layout="wide")
st.title("🧭 Options Decision Matrix — Direction × Volatility")
st.caption("Type a ticker, we infer trend & vol regime, then recommend an options strategy with a payoff preview.")

# =========================
# Helpers
# =========================
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out

def get_hist(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_options_snapshot(ticker: str):
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        # choose nearest expiry >= today
        today = datetime.utcnow().date()
        exp = None
        for e in expirations:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
                if d >= today:
                    exp = e
                    break
            except Exception:
                continue
        if not exp:
            exp = expirations[0]
        chain = t.option_chain(exp)
        return {"exp": exp, "calls": chain.calls.copy(), "puts": chain.puts.copy()}
    except Exception:
        return None

def realized_vol(close: pd.Series, lookback: int = 21) -> float:
    # simple 21d annualized realized vol
    rets = close.pct_change().dropna()
    if len(rets) < 2:
        return float("nan")
    window = rets.tail(lookback)
    return float(window.std() * np.sqrt(252))

def approx_atm_iv(snapshot, spot: float) -> float:
    """Estimate ATM IV from closest strike call/put if available; else NaN."""
    if not snapshot:
        return float("nan")
    calls = snapshot["calls"]
    puts = snapshot["puts"]
    if calls is None or puts is None or calls.empty or puts.empty:
        return float("nan")
    try:
        atm_strike = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]].strike.values[0]
        c_iv = calls.loc[calls['strike'] == atm_strike, 'impliedVolatility']
        p_iv = puts.loc[puts['strike'] == atm_strike, 'impliedVolatility']
        ivs = []
        if not c_iv.empty and pd.notna(c_iv.iloc[0]):
            ivs.append(float(c_iv.iloc[0]))
        if not p_iv.empty and pd.notna(p_iv.iloc[0]):
            ivs.append(float(p_iv.iloc[0]))
        if ivs:
            return float(np.mean(ivs))
    except Exception:
        pass
    return float("nan")

def trend_signal(close: pd.Series):
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    latest = close.iloc[-1]
    rsi14 = rsi(close).iloc[-1]
    # Simple scoring
    score = 0
    # MA alignment
    if len(ma200.dropna()) > 0:
        if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
            score += 2
        elif ma20.iloc[-1] > ma50.iloc[-1]:
            score += 1
        elif ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
            score -= 2
        elif ma20.iloc[-1] < ma50.iloc[-1]:
            score -= 1
    # RSI tilt
    if rsi14 >= 70: score += 1
    elif rsi14 <= 30: score -= 1

    # Map score to direction bucket
    if score >= 2:
        direction = "Bullish"
    elif score <= -2:
        direction = "Bearish"
    else:
        direction = "Neutral"
    return direction, float(rsi14), latest, ma20.iloc[-1] if not np.isnan(ma20.iloc[-1]) else np.nan, ma50.iloc[-1] if not np.isnan(ma50.iloc[-1]) else np.nan, ma200.iloc[-1] if not np.isnan(ma200.iloc[-1]) else np.nan

# =========================
# Sidebar Inputs
# =========================
with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    period = st.selectbox("Price history window", ["3mo","6mo","1y","2y"], index=1)
    lookback = st.slider("Realized vol lookback (days)", 10, 60, 21)
    risk_mode = st.selectbox("Risk preference", ["Conservative","Balanced","Aggressive"], index=1)
    own_shares = st.checkbox("I own 100+ shares (for covered calls)", value=False)

# =========================
# Data Fetch
# =========================
prices = get_hist(ticker, period)
if prices.empty:
    st.error("No price data. Try another ticker.")
    st.stop()

close = prices['Close']
S0 = float(close.iloc[-1])
rv = realized_vol(close, lookback)
opt_snap = get_options_snapshot(ticker)
iv_atm = approx_atm_iv(opt_snap, S0)

# Volatility regime classification
vol_note = ""
if not np.isnan(iv_atm):
    # compare IV to RV
    ratio = iv_atm / rv if rv and rv > 0 else np.nan
    if not np.isnan(ratio):
        if ratio >= 1.25:
            vol_regime = "High (IV >> RV)"
        elif ratio <= 0.85:
            vol_regime = "Low (IV << RV)"
        else:
            vol_regime = "Normal"
    else:
        vol_regime = "Normal"
else:
    # fallback using RV z-score vs its own history
    rv_hist = close.pct_change().rolling(252).std() * np.sqrt(252)
    z = (rv - rv_hist.mean()) / rv_hist.std() if rv_hist.std() not in [0, np.nan] else 0
    if z > 1:
        vol_regime = "High (RV)"
    elif z < -1:
        vol_regime = "Low (RV)"
    else:
        vol_regime = "Normal"
    vol_note = "(IV unavailable, used realized vol proxy)"

# Direction signal
direction, rsi14, ma20, ma20v, ma50v, ma200v = trend_signal(close)

# =========================
# Header Metrics
# =========================
colA, colB, colC, colD = st.columns(4)
colA.metric("Price", f"{S0:,.2f}")
colB.metric("Direction", direction)
colC.metric("Vol Regime", vol_regime)
colD.metric("ATM IV", f"{iv_atm:.2%}" if not np.isnan(iv_atm) else "N/A")
st.caption(f"RV(annualized): {rv:.2%} {vol_note}")

# =========================
# Decision Matrix
# =========================
MATRIX = {
    ("Bullish", "Low"): ["Long Call", "Bull Call Debit Spread"],
    ("Bullish", "Normal"): ["Bull Call Debit Spread", "Cash-Secured Put"],
    ("Bullish", "High"): ["Cash-Secured Put", "Bull Put Credit Spread", "Covered Call"],
    ("Bearish", "Low"): ["Long Put", "Bear Put Debit Spread"],
    ("Bearish", "Normal"): ["Bear Put Debit Spread", "Call Credit Spread"],
    ("Bearish", "High"): ["Call Credit Spread", "Covered Call (if long)"] ,
    ("Neutral", "Low"): ["Long Straddle/Strangle"],
    ("Neutral", "Normal"): ["Broken Wing Butterfly", "Calendar (directional)"] ,
    ("Neutral", "High"): ["Iron Condor", "Short Strangle (defined risk)"]
}

# Normalize vol bucket keyword used as keys
if "High" in vol_regime:
    vol_key = "High"
elif "Low" in vol_regime:
    vol_key = "Low"
else:
    vol_key = "Normal"

primary_list = MATRIX.get((direction, vol_key), ["Long Call"])  # fallback

# Ownership constraint
if not own_shares:
    primary_list = [s for s in primary_list if "Covered Call" not in s]
if risk_mode == "Conservative":
    # drop undefined-risk choices
    primary_list = [s for s in primary_list if ("Credit Spread" in s) or ("Debit Spread" in s) or ("Condor" in s) or ("Calendar" in s) or ("Covered Call" in s) or ("Cash-Secured Put" in s)]

# Final pick
strategy_pick = primary_list[0] if primary_list else ("Bull Call Debit Spread" if direction=="Bullish" else "Bear Put Debit Spread")

# Show matrix table
st.subheader("Decision Matrix Output")
st.write(f"**Direction:** `{direction}`  |  **Volatility:** `{vol_regime}`  |  **Suggested Strategies:** {', '.join(primary_list)}")

# =========================
# Payoff Builder for Picked Strategy
# =========================
with st.expander("🔧 Configure Trade & Preview Payoff", expanded=True):
    col1, col2, col3 = st.columns([1,1,1])

    K = col1.number_input("Strike K", value=int(S0), step=1)
    width = col2.number_input("Spread width (if spread)", value=5, step=1, min_value=1)
    premium = col3.number_input("Net premium (debit + / credit -)", value=1.50, step=0.05, format="%.2f")

    S_min = st.slider("Price range at expiration (Sₜ)", max(1, int(S0*0.2)), int(S0*2.0), (int(S0*0.6), int(S0*1.4)), 1)
    S_grid = np.linspace(S_min[0], S_min[1], 300)

    def payoff_curve(strategy: str, S: np.ndarray, S0: float, K: float, width: float, premium: float):
        K2 = K + width
        if strategy == "Long Call":
            return np.maximum(S - K, 0) - abs(premium)
        if strategy == "Long Put":
            return np.maximum(K - S, 0) - abs(premium)
        if strategy == "Bull Call Debit Spread":
            return np.maximum(S - K, 0) - np.maximum(S - K2, 0) - abs(premium)
        if strategy == "Bear Put Debit Spread":
            return np.maximum(K - S, 0) - np.maximum(K2 - S, 0) - abs(premium)
        if strategy == "Cash-Secured Put":
            # credit => premium should be negative in our convention; profit when stays above K
            return (-premium) - np.maximum(K - S, 0)
        if strategy == "Call Credit Spread":
            # short call at K, long call at K2=K+width; credit trade (premium negative)
            return (-premium) - np.maximum(S - K, 0) + np.maximum(S - K2, 0)
        if strategy == "Iron Condor":
            # symmetric condor around K: (K-w, K) calls and (K, K+w) puts; approximate with width
            Kp1, Kp2 = K - width, K
            Kc1, Kc2 = K, K + width
            # short put spread + short call spread; credit = -premium
            put_spread = (-np.maximum(Kp1 - S, 0) + np.maximum(Kp2 - S, 0))
            call_spread = (-np.maximum(S - Kc2, 0) + np.maximum(S - Kc1, 0))
            return (-premium) + put_spread + call_spread
        if strategy == "Long Straddle/Strangle":
            # assume strangle with K (call) and K2=K-width (put lower); debit trade
            Kl = K - width
            return np.maximum(S - K, 0) + np.maximum(Kl - S, 0) - abs(premium)
        if strategy == "Covered Call":
            return (S - S0) + (-premium) - np.maximum(S - K, 0)
        if strategy == "Bull Put Credit Spread":
            # short put at K, long put at K-width; credit = -premium
            Kl = K - width
            return (-premium) - np.maximum(K - S, 0) + np.maximum(Kl - S, 0)
        # default
        return np.zeros_like(S)

    y = payoff_curve(strategy_pick, S_grid, S0, K, width, premium)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_grid, y=y, mode='lines', name=f'Payoff — {strategy_pick}'))
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=S0, line_dash="dot")
    fig.update_layout(title=f"{strategy_pick} — Payoff at Expiration", xaxis_title="Underlying Price Sₜ", yaxis_title="P/L per 1x")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Strategy Rationale
# =========================
st.subheader("📋 Strategy Rationale")
why = {
    "Long Call": "Bullish view with low/normal IV. Defined risk (debit), uncapped upside.",
    "Bull Call Debit Spread": "Bullish but cost-conscious; reduces debit and theta by selling a higher strike.",
    "Cash-Secured Put": "Get paid to potentially buy at a discount; best when IV is high.",
    "Bull Put Credit Spread": "Bullish-to-neutral with high IV; limited risk credit trade.",
    "Long Put": "Bearish view with low/normal IV; defined risk.",
    "Bear Put Debit Spread": "Bearish but reduce cost and theta via long put spread.",
    "Call Credit Spread": "Bearish-to-neutral with high IV; limited risk credit trade.",
    "Iron Condor": "Range-bound + high IV; profit if price stays inside the wings.",
    "Long Straddle/Strangle": "Expect big move with low IV; direction-agnostic.",
    "Covered Call": "Hold shares and sell calls for income; capped upside, downside remains.",
}

st.write(f"**Suggested Primary Strategy:** `{strategy_pick}`")
st.write(why.get(strategy_pick, ""))

# =========================
# Options Snapshot (optional)
# =========================
st.subheader("🔍 Options Snapshot (nearest expiry)")
if opt_snap:
    exp = opt_snap["exp"]
    calls = opt_snap["calls"].copy()
    puts = opt_snap["puts"].copy()

    # Thin the display to useful columns
    def thin(df):
        cols = [c for c in ["contractSymbol","lastPrice","bid","ask","impliedVolatility","delta","gamma","theta","vega","rho","openInterest","volume","strike"] if c in df.columns]
        out = df[cols].sort_values("openInterest", ascending=False).head(15)
        # pretty IV
        if "impliedVolatility" in out.columns:
            out["impliedVolatility"] = (out["impliedVolatility"].astype(float) * 100).round(2).astype(str) + "%"
        return out

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Calls** — Exp: `{exp}`")
        st.dataframe(thin(calls), use_container_width=True)
    with col2:
        st.markdown(f"**Puts** — Exp: `{exp}`")
        st.dataframe(thin(puts), use_container_width=True)
else:
    st.info("No options chain available (ticker may not have listed options or data fetch failed).")

# =========================
# Notes & Disclaimers
# =========================
st.caption("This tool is educational. Always manage risk. Prefer defined-risk spreads over naked selling, especially into earnings or high gap risk situations.")
