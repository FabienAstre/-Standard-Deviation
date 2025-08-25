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
    value="AAPL, MSFT, BTC-USD", 
    key="tickers_input"
)

period = st.selectbox(
    "Select price history period for analysis", 
    ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
    key="period_select"
)

show_only_undervalued = st.checkbox(
    "Show only Undervalued Assets (Z-score < -1)", 
    value=False, key="undervalued_check"
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
    """
    Fetch historical data and compute:
    - Current price, mean, std dev, Z-score
    - RSI
    - Fundamentals (P/E, P/B, Div Yield, Market Cap)
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

    # Price signal
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
# Display Screener Table
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
        
        # Std dev bands
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
# Options Analysis
# =========================
st.subheader("📊 Options Indicators")
opt_ticker = st.text_input("Ticker for options analysis", value="AAPL", key="opt_ticker")

if opt_ticker:
    try:
        opt_chain = yf.Ticker(opt_ticker)
        if not opt_chain.options:
            st.info("No options available for this ticker.")
        else:
            exp_date = st.selectbox("Select expiration date", options=opt_chain.options, key="opt_exp_date")
            chain = opt_chain.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            st.write("Top 5 Call Options by Volume")
            st.dataframe(calls.sort_values("volume", ascending=False).head(5))
            st.write("Top 5 Put Options by Volume")
            st.dataframe(puts.sort_values("volume", ascending=False).head(5))

            # Sentiment
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            if call_oi > put_oi * 1.2:
                st.success("Market Bias: Bullish (Calls dominate Open Interest)")
                opt_weight = 1
            elif put_oi > call_oi * 1.2:
                st.error("Market Bias: Bearish (Puts dominate Open Interest)")
                opt_weight = -1
            else:
                st.info("Market Bias: Neutral")
                opt_weight = 0

            avg_call_premium = calls['lastPrice'].mean()
            st.markdown(f"💡 Covered Call Tip: Avg Call Premium = ${round(avg_call_premium,2)}")

    except Exception as e:
        st.error(f"Error fetching options: {e}")

# =========================
# 🎓 Options Education & Payoff Simulator
# =========================
import numpy as np

st.subheader("🎓 Options Education & Payoff Simulator")

colA, colB = st.columns([1,1])
with colA:
    edu_ticker = st.text_input("Ticker (to auto-seed current price)", value="AAPL", key="edu_ticker")
with colB:
    use_live_price = st.checkbox("Auto-fetch current price with yfinance", value=True, key="edu_use_live")

# Get a starting price S0
def get_current_price(ticker: str, fallback: float = 100.0):
    try:
        h = yf.Ticker(ticker).history(period="5d")
        if not h.empty:
            return float(h["Close"][-1])
    except Exception:
        pass
    return fallback

S0 = get_current_price(edu_ticker, fallback=100.0) if use_live_price else 100.0

st.markdown(f"**Seed Price (S₀)**: `{round(S0, 2)}` — used to center the payoff range.")

strategy = st.selectbox(
    "Choose a strategy",
    ["Long Call", "Long Put", "Covered Call (own 100 shares)", "Cash-Secured Put", "Bull Call Spread"],
    key="edu_strategy"
)

# Price range at expiration
rng = st.slider(
    "Underlying price range at expiration (Sₜ)",
    min_value=max(1, int(S0*0.2)),
    max_value=int(S0*2.0),
    value=(int(S0*0.6), int(S0*1.4)),
    step=1,
    key="edu_range"
)
S_grid = np.linspace(rng[0], rng[1], 300)

st.divider()

# --- Strategy-specific inputs
if strategy in ["Long Call", "Bull Call Spread"]:
    K_call = st.number_input("Call strike K (buy)", value=float(round(S0*1.0)), step=1.0, key="edu_K_call_buy")
    prem_call = st.number_input("Premium paid for long call (per share)", value=2.00, step=0.10, key="edu_prem_call_buy")

if strategy == "Bull Call Spread":
    K_call_short = st.number_input("Call strike K (sell)", value=float(round(S0*1.1)), step=1.0, key="edu_K_call_sell")
    prem_call_short = st.number_input("Premium received for short call (per share)", value=1.00, step=0.10, key="edu_prem_call_sell")

if strategy in ["Long Put", "Cash-Secured Put"]:
    K_put = st.number_input("Put strike K", value=float(round(S0*0.95)), step=1.0, key="edu_K_put")
    prem_put = st.number_input("Premium (per share)", value=2.00, step=0.10, key="edu_prem_put")

if strategy == "Covered Call (own 100 shares)":
    entry_price = st.number_input("Your share cost basis (per share)", value=float(round(S0,2)), step=0.10, key="edu_cost_basis")
    K_cov = st.number_input("Covered call strike K", value=float(round(S0*1.05)), step=1.0, key="edu_K_cov")
    prem_cov = st.number_input("Premium received (per share)", value=2.00, step=0.10, key="edu_prem_cov")

# --- Compute payoff per share (except covered call we also show per 100)
payoff = None
breakeven = None
max_profit = None
max_loss = None
label = ""

if strategy == "Long Call":
    payoff = np.maximum(S_grid - K_call, 0.0) - prem_call
    breakeven = K_call + prem_call
    max_profit = "Unlimited ↑"
    max_loss = f"{prem_call:.2f} per share"
    label = f"Long Call (K={K_call:.2f}, premium={prem_call:.2f})"

elif strategy == "Long Put":
    payoff = np.maximum(K_put - S_grid, 0.0) - prem_put
    breakeven = K_put - prem_put
    max_profit = f"{(K_put - 0):.2f} − premium ≈ {(K_put - 0 - prem_put):.2f} per share"
    max_loss = f"{prem_put:.2f} per share"
    label = f"Long Put (K={K_put:.2f}, premium={prem_put:.2f})"

elif strategy == "Covered Call (own 100 shares)":
    # Position = +100 shares at entry_price, + short call at K_cov (receive premium)
    # Per-share P&L at expiry = (min(S, K) - entry_price) + premium
    per_share = np.minimum(S_grid, K_cov) - entry_price + prem_cov
    payoff = per_share  # per share
    breakeven = entry_price - prem_cov
    max_profit_val = (K_cov - entry_price + prem_cov) * 100
    max_loss_val = (0 - entry_price + prem_cov) * 100  # if S -> 0
    max_profit = f"${max_profit_val:,.2f} per 100 sh"
    max_loss = f"${max_loss_val:,.2f} per 100 sh"
    label = f"Covered Call (K={K_cov:.2f}, premium={prem_cov:.2f}, basis={entry_price:.2f})"

elif strategy == "Cash-Secured Put":
    # Short put payoff per share = premium - max(K - S, 0)
    payoff = prem_put - np.maximum(K_put - S_grid, 0.0)
    breakeven = K_put - prem_put
    max_profit = f"{prem_put:.2f} per share (kept if Sₜ ≥ K)"
    max_loss = f"≈ {K_put - prem_put:.2f} per share (if Sₜ→0)"
    label = f"Cash-Secured Put (K={K_put:.2f}, premium={prem_put:.2f})"

elif strategy == "Bull Call Spread":
    # (Buy K_call, pay prem_call) + (Sell K_call_short, receive prem_call_short)
    long_leg = np.maximum(S_grid - K_call, 0.0) - prem_call
    short_leg = -(np.maximum(S_grid - K_call_short, 0.0) - prem_call_short)
    payoff = long_leg + short_leg
    net_debit = prem_call - prem_call_short
    breakeven = K_call + net_debit
    max_profit_val = (K_call_short - K_call) - net_debit
    max_profit = f"{max_profit_val:.2f} per share"
    max_loss = f"{net_debit:.2f} per share"
    label = f"Bull Call Spread (Buy {K_call:.2f} / Sell {K_call_short:.2f}, net debit={net_debit:.2f})"

# --- Plot
if payoff is not None:
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=S_grid, y=payoff, mode="lines", name="P&L at Expiry"))
    fig_pay.add_vline(x=S0, line_dash="dot", annotation_text="S₀", annotation_position="top right")
    if breakeven is not None and breakeven >= S_grid.min() and breakeven <= S_grid.max():
        fig_pay.add_vline(x=breakeven, line_dash="dash", annotation_text="Breakeven", annotation_position="top left")
    fig_pay.update_layout(
        title=f"Payoff: {label}",
        xaxis_title="Underlying Price at Expiration (Sₜ)",
        yaxis_title="Profit / Loss (per share unless noted)",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_pay, use_container_width=True)

    # Metrics / explanation
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Breakeven (approx.)", f"{breakeven:.2f}" if isinstance(breakeven, (float,int)) else "—")
    with m2:
        st.metric("Max Profit", max_profit if isinstance(max_profit, str) else f"{max_profit:.2f}")
    with m3:
        st.metric("Max Loss", max_loss if isinstance(max_loss, str) else f"{max_loss:.2f}")

# -------------------------
# Explanations
# -------------------------
st.markdown("### 📘 Strategy Explanations")
if strategy == "Long Call":
    st.markdown("""
**Long Call** (debit): You pay a premium for upside.  
- **Use when:** Bullish.  
- **Breakeven:** `K + premium`.  
- **Max profit:** Unlimited.  
- **Max loss:** Premium paid.
""")
elif strategy == "Long Put":
    st.markdown("""
**Long Put** (debit): You pay a premium for downside protection or a bearish bet.  
- **Use when:** Bearish or hedging.  
- **Breakeven:** `K − premium`.  
- **Max profit:** Approaches `K − premium` if Sₜ → 0.  
- **Max loss:** Premium paid.
""")
elif strategy == "Covered Call (own 100 shares)":
    st.markdown("""
**Covered Call**: Own 100 shares and sell 1 call. You collect premium; upside capped at strike.  
- **Use when:** Neutral to mildly bullish; okay to sell shares at strike.  
- **Breakeven:** `Cost basis − premium`.  
- **Max profit:** `(K − cost basis + premium) × 100`.  
- **Max loss:** `(0 − cost basis + premium) × 100` if stock goes to zero (still stock risk).
""")
elif strategy == "Cash-Secured Put":
    st.markdown("""
**Cash-Secured Put**: Sell a put and hold cash to buy shares if assigned.  
- **Use when:** Neutral to mildly bullish; want to buy stock at discount.  
- **Breakeven:** `K − premium`.  
- **Max profit:** Premium received.  
- **Max loss:** Approaches `K − premium` if Sₜ → 0.
""")
elif strategy == "Bull Call Spread":
    st.markdown("""
**Bull Call Spread**: Buy a call at K₁ and sell a higher-strike call at K₂ to reduce cost.  
- **Use when:** Moderately bullish with defined risk.  
- **Breakeven:** `K₁ + net debit`.  
- **Max profit:** `(K₂ − K₁) − net debit`.  
- **Max loss:** `net debit`.
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
st.markdown("""
**Explanation:**  
- **Z-score**: Price vs historical mean  
- **RSI**: Oversold (<30) = bullish, Overbought (>70) = bearish  
- **Fib Trend**: Fibonacci-based trend assessment  
- **Options Sentiment**: Calls vs Puts open interest  
- **Combined Score**: Sum of all metric weights  
- **Recommendation**: Strong Buy / Buy / Hold / Sell
""")
