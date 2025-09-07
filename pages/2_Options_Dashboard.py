import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from utils.options import payoff_long_call, payoff_long_put

st.title("📉 Options Dashboard & Simulator")

ticker = st.text_input("Enter ticker", "AAPL").upper()
try:
    t = yf.Ticker(ticker)
    st.write(f"**Current Price:** {t.fast_info['last_price']:.2f}")
    expirations = t.options
except Exception:
    st.error("Ticker not found or no options data.")
    st.stop()

expiration = st.selectbox("Select expiration", expirations)
opt_chain = t.option_chain(expiration)
calls, puts = opt_chain.calls, opt_chain.puts

st.subheader("Top Calls")
st.dataframe(calls.head(10))
st.subheader("Top Puts")
st.dataframe(puts.head(10))

# Strategy simulator
st.markdown("### 📊 Payoff Simulator")
strategy = st.selectbox("Strategy", ["Long Call", "Long Put"])
strike = st.number_input("Strike Price", value=float(calls["strike"].iloc[0]))
premium = st.number_input("Premium", value=1.0)

price_range = np.linspace(0.5 * float(t.fast_info['last_price']),
                          1.5 * float(t.fast_info['last_price']), 100)

if strategy == "Long Call":
    payoff = payoff_long_call(price_range, strike, premium)
else:
    payoff = payoff_long_put(price_range, strike, premium)

fig, ax = plt.subplots()
ax.plot(price_range, payoff, label=strategy)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
st.pyplot(fig)
