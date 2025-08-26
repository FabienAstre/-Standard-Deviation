import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Stock & Crypto Screener + Combined Dashboard", layout="wide")

# ==================
# Helper functions
# ==================
@st.cache_data
def fetch_batch_history(tickers, period, interval="1d"):
    """Fetches historical data for multiple tickers using yfinance.download."""
    try:
        data = yf.download(tickers=tickers, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return None

@st.cache_data
def fetch_option_sentiment(ticker: str):
    try:
        t = yf.Ticker(ticker)
        if not t.options:
            return None
        exp = t.options[0]
        chain = t.option_chain(exp)
        call_oi = chain.calls['openInterest'].sum()
        put_oi = chain.puts['openInterest'].sum()
        if call_oi > put_oi * 1.2:
            return 1
        elif put_oi > call_oi * 1.2:
            return -1
        else:
            return 0
    except Exception:
        return None

def compute_rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def compute_zscore(series: pd.Series, window: int = 30) -> float:
    # z-score of the last price relative to rolling mean/std over window
    if len(series) < window:
        return 0.0
    rolling_mean = series.rolling(window).mean().iloc[-1]
    rolling_std = series.rolling(window).std().iloc[-1]
    if rolling_std == 0 or np.isnan(rolling_std):
        return 0.0
    z = (series.iloc[-1] - rolling_mean) / rolling_std
    return float(z)

def compute_fib_levels(low, high):
    levels = {
        'fib_0': low,
        'fib_382': low + 0.382 * (high - low),
        'fib_50': low + 0.5 * (high - low),
        'fib_618': low + 0.618 * (high - low),
        'fib_100': high
    }
    return levels

def build_candlestick_with_rsi(df, ticker, fib_levels=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{ticker}'), row=1, col=1)

    # Add Fibonacci levels
    if fib_levels is not None:
        for name, level in fib_levels.items():
            fig.add_hline(y=level, line_dash='dash', annotation_text=name, row=1, col=1)

    # RSI
    rsi_series = compute_rsi(df['Close'].copy(), window=14)
    # For plotting RSI, compute full RSI series
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ma_up / ma_down
    rsi_full = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=df.index, y=rsi_full, name='RSI'), row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    fig.update_layout(height=500, showlegend=False, title_text=f"{ticker} Price & RSI")
    return fig

# ==================
# Sidebar / Inputs
# ==================
st.sidebar.title("Settings")
input_mode = st.sidebar.radio("Input tickers by:", ['Text input', 'Upload CSV'])
period = st.sidebar.selectbox("History period:", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=2)
interval = st.sidebar.selectbox("Interval:", ['1d', '1wk', '1mo'], index=0)
min_volume = st.sidebar.number_input("Minimum average volume (last period):", value=0, step=1000)

if input_mode == 'Text input':
    tickers_raw = st.sidebar.text_area("Tickers (comma separated)", value="AAPL,MSFT,TSLA,NVDA,GOOG")
    tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV with a 'Ticker' column", type=['csv'])
    tickers = []
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            if 'Ticker' in df_in.columns:
                tickers = [str(t).strip().upper() for t in df_in['Ticker'].unique()]
            else:
                st.sidebar.error("CSV must contain a 'Ticker' column")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.write("This app fetches data via yfinance. For large lists, fetching may take time.")

# ==================
# Fetch data
# ==================
if not tickers:
    st.info("Enter at least one ticker to analyze.")
    st.stop()

with st.spinner("Fetching historical data..."):
    raw_hist = fetch_batch_history(tickers, period=period, interval=interval)

# raw_hist format: if multiple tickers, columns are multiindex (ticker, feature)

# Prepare results list
results = []

for ticker in tickers:
    try:
        if len(tickers) == 1:
            df = raw_hist.copy()
        else:
            if ticker in raw_hist.columns.levels[0]:
                df = raw_hist[ticker].dropna()
            else:
                st.warning(f"No history found for {ticker}")
                continue

        if df.empty:
            st.warning(f"No data for {ticker}")
            continue

        close = df['Close']
        current_price = float(close.iloc[-1])
        avg_vol = None
        if 'Volume' in df.columns:
            avg_vol = int(df['Volume'].tail(30).mean())
            if avg_vol < min_volume:
                st.info(f"Skipping {ticker} due to low average volume ({avg_vol})")
                continue

        z = compute_zscore(close, window=min(60, max(10, int(len(close)/2))))
        rsi_val = compute_rsi(close, window=14)

        high_price = df['High'].max()
        low_price = df['Low'].min()
        fib = compute_fib_levels(low_price, high_price)

        # Fib trend scoring
        if current_price > fib['fib_618']:
            fib_trend = 2
        elif current_price > fib['fib_50']:
            fib_trend = 1
        elif current_price > fib['fib_382']:
            fib_trend = 0
        else:
            fib_trend = -1

        # Z-score weight
        if z < -2:
            z_weight = 2
        elif z < -1:
            z_weight = 1
        elif z > 2:
            z_weight = -2
        elif z > 1:
            z_weight = -1
        else:
            z_weight = 0

        # RSI weight
        if rsi_val < 30:
            rsi_weight = 1
        elif rsi_val > 70:
            rsi_weight = -1
        else:
            rsi_weight = 0

        opt_sent = fetch_option_sentiment(ticker)
        opt_display = opt_sent if opt_sent is not None else 'N/A'
        opt_weight = opt_sent if opt_sent is not None else 0

        total_score = z_weight + rsi_weight + fib_trend + opt_weight

        if total_score >= 3:
            rec = '🟢 Strong Buy'
        elif total_score == 2:
            rec = '🟢 Buy'
        elif total_score in [0, 1]:
            rec = '🟡 Hold'
        elif total_score <= -3:
            rec = '🔴 Strong Sell'
        else:
            rec = '🔴 Sell'

        results.append({
            'ticker': ticker,
            'current_price': current_price,
            'avg_volume': avg_vol,
            'z_score': round(z, 3),
            'rsi': round(rsi_val, 2),
            'fib_trend': fib_trend,
            'fib_levels': fib,
            'opt_sentiment': opt_display,
            'opt_weight': opt_weight,
            'total_score': total_score,
            'recommendation': rec,
            'history_df': df
        })

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")

# ==================
# Combined Recommendation Dashboard
# ==================
st.header("📊 Combined Trend & Recommendation Dashboard")
recommendations = []
for r in results:
    recommendations.append({
        'Ticker': r['ticker'],
        'Price': r['current_price'],
        'Avg Volume': r['avg_volume'],
        'Z-score': r['z_score'],
        'RSI': r['rsi'],
        'Fib Trend': r['fib_trend'],
        'Options Sentiment': r['opt_sentiment'],
        'Combined Score': r['total_score'],
        'Recommendation': r['recommendation']
    })

rec_df = pd.DataFrame(recommendations)

# color function
def highlight_rec(val):
    if 'Strong Buy' in val:
        return 'background-color: #b6f7b6'
    if val == '🟢 Buy':
        return 'background-color: #d4fcd4'
    if val == '🟡 Hold':
        return 'background-color: #fff8b3'
    if val == '🔴 Sell':
        return 'background-color: #fcd4d4'
    if 'Strong Sell' in val:
        return 'background-color: #f7b6b6'
    return ''

st.dataframe(rec_df.style.applymap(highlight_rec, subset=['Recommendation']), use_container_width=True)

# Download recommendations
csv = rec_df.to_csv(index=False)
st.download_button(label="Download recommendations CSV", data=csv, file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')

# ==================
# Detailed per-ticker view
# ==================
st.header("🔎 Ticker Details")
for r in results:
    ticker = r['ticker']
    df = r['history_df']
    fib = r['fib_levels']
    st.subheader(f"{ticker} — {r['recommendation']} — Price: {r['current_price']}")
    col1, col2 = st.columns([3,1])
    with col1:
        fig = build_candlestick_with_rsi(df, ticker, fib_levels=fib)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric(label="Combined Score", value=r['total_score'])
        st.metric(label="RSI", value=r['rsi'])
        st.metric(label="Z-score", value=r['z_score'])
        st.write("**Fibonacci levels**")
        st.write(pd.Series(fib))
        st.write("**Options Sentiment**")
        st.write(r['opt_sentiment'])

# ==================
# Footer / Notes
# ==================
st.write("---")
st.caption("Signals are algorithmic and for informational purposes only. Not financial advice.")

