import ccxt
import pandas as pd

exchange = ccxt.binance()

def fetch_data(symbol, timeframe, limit=200):
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df