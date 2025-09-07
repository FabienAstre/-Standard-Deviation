import yfinance as yf
import pandas as pd

def fetch_price_data(ticker, period="1y"):
    return yf.download(ticker, period=period, interval="1d")

def compute_rsi(series, window=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(window).mean() / down.rolling(window).mean()
    return 100 - (100 / (1 + rs)).iloc[-1]

def compute_zscore(series):
    return (series.iloc[-1] - series.mean()) / series.std()
