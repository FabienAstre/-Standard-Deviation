import streamlit as st

st.set_page_config(
    page_title="Stock & Options Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock & Options Analyzer")
st.markdown("""
Welcome! This app has three modules:

1. **Stock Screener** – fundamentals, RSI, Z-score, signals.  
2. **Options Dashboard** – options chains, Greeks, payoff simulator.  
3. **Strategy Decision Matrix** – strategy recommender & payoff builder.  

👉 Use the sidebar to switch between modules.
""")
