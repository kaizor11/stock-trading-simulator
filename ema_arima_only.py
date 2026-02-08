import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamz import Stream
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time
import sys

warnings.filterwarnings("ignore")

# TICKER = "TSLA"
# PERIOD = "1y"
# INTERVAL = "1d"
# FEE = 0.0005                # 0.05% per trade

# EMA_SPAN = 20                    
# ARIMA_ORDER = (1, 0, 0)     # (p, d, q)
# STREAM_WINDOW = 50          # number of data points used for ARIMA

# INITIAL_CAPITAL = 100_000

from config import *

def compute_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def arima_forecast(price_series):
    model = ARIMA(price_series, order=ARIMA_ORDER)
    model = model.fit()
    forecast = model.forecast(steps=1)
    return forecast.iloc[0]

def trading_strategy(df):
    """
    BUY if
    - currently FLAT and price > EMA and ARIMA predicts higher price

    SELL if
    - currently LONG and price < EMA or ARIMA predicts lower price
    """

    signal_price = float(df["Close"].iloc[-1])
    execution_price = float(df["Open"].iloc[-1])

    current_ema = float(df["EMA"].iloc[-1])
    arima_pred = float(df["ARIMA_Pred"].iloc[-1])

    timestamp = df.index[-1]
    signal = "HOLD"

    # execute on previous signal to avoid look-ahead bias
    if portfolio.get("pending_signal") == "BUY":
        shares = int(portfolio["cash"] // (execution_price * (1 + FEE)))
        cost = shares * execution_price * (1 + FEE)
        portfolio["shares"] = shares
        portfolio["cash"] -= cost
        portfolio["position"] = "LONG"

    elif portfolio.get("pending_signal") == "SELL":
        proceeds = portfolio["shares"] * execution_price * (1 - FEE)
        portfolio["cash"] += proceeds
        portfolio["shares"] = 0
        portfolio["position"] = "FLAT"

    portfolio["pending_signal"] = None

    # buy / sell logic
    if (
        portfolio["position"] == "FLAT"
        and signal_price > current_ema
        and arima_pred > signal_price
    ):
        signal = "BUY"
        portfolio["pending_signal"] = "BUY"

    elif (
        portfolio["position"] == "LONG"
        and (signal_price < current_ema or arima_pred < signal_price)
    ):
        signal = "SELL"
        portfolio["pending_signal"] = "SELL"

    equity = portfolio["cash"] + portfolio["shares"] * execution_price

    portfolio_history.append({
        "timestamp": timestamp,
        "close_price": signal_price,
        "open_price": execution_price,
        "signal": signal,
        "position": portfolio["position"],
        "shares": portfolio["shares"],
        "cash": portfolio["cash"],
        "equity": equity,
    })

def process_stream(row):
    buffer.append(row)

    # keep buffer size fixed
    if len(buffer) > STREAM_WINDOW:
        buffer.pop(0)

    df = pd.DataFrame(buffer).set_index("timestamp")
    df["EMA"] = compute_ema(df["Close"], EMA_SPAN)

    # ARIMA works better on stationary data, so we use log returns
    log_returns = np.log(df["Close"]).diff().dropna()
    # only run ARIMA if enough data
    if len(log_returns) >= STREAM_WINDOW - 1:
        df["ARIMA_Pred"] = np.nan
        current_price = float(df["Close"].iloc[-1])
        arima_pred_returns = arima_forecast(log_returns)
        arima_price = current_price * np.exp(arima_pred_returns)
        df.iloc[-1, df.columns.get_loc("ARIMA_Pred")] = arima_price

        trading_strategy(df)

def metrics():
    returns = np.asarray(portfolio_history['returns'])
    equity = np.asarray(portfolio_history['equity'])
    
    # annualize returns
    total_return = np.prod(1 + returns) - 1
    periods_per_year = 252  # approx 252 trading days per year
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # sharpe ratio: measures risk-adjusted relative returns
    risk_free_rate = 0.02
    volatility = np.std(returns, ddof=1)
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # max_drawdown: greatest movement from a high point to a low point
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    max_drawdown = drawdowns.min()

    # print results
    print(f"\nEMA + ARIMA:")
    print(f"Initial Capital: {INITIAL_CAPITAL * EMA_ARIMA_SPLIT}")
    print(f"Final Equity: ${equity[-1]:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

def main():
    global data
    global portfolio
    global portfolio_history
    global buffer
    
    start = time.time()

    ticker = yf.Ticker(TICKER)
    data = ticker.history(
        period=PERIOD,
        interval=INTERVAL
    )
    data = data[["Open", "Close", "Volume"]]

    portfolio = {
        "cash": INITIAL_CAPITAL * EMA_ARIMA_SPLIT,
        "shares": 0,
        "position": "FLAT"  # FLAT or LONG
    }

    portfolio_history = []
    buffer = []

    price_stream = Stream()

    price_stream.sink(process_stream)

    for timestamp, row in data.iterrows():
        price_stream.emit({
        "timestamp": timestamp,
        "Open": float(row["Open"]),
        "Close": float(row["Close"]),
        "Volume": float(row.get("Volume", 0))
    })
    
    # results
    portfolio_history = pd.DataFrame(portfolio_history)
    portfolio_history.set_index("timestamp", inplace=True)
    portfolio_history['returns'] = portfolio_history['equity'].pct_change().fillna(0)
    metrics()
    print(f"Execution Time: {time.time() - start:.2f} seconds")

    # plot results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # PLOT: portfolio value over time
    axes[0].plot(
        portfolio_history.index,
        portfolio_history["equity"],
        label="Equity"
    )
    axes[0].set_title("Equity Over Time")
    axes[0].set_xlabel("Time")

    # PLOT: price and signals
    axes[1].plot(
        portfolio_history.index,
        portfolio_history["close_price"],
        label=f"{TICKER} Close Price"
    )
    axes[1].set_title(f"{TICKER} Price Over Time")
    axes[1].set_xlabel("Time")

    buy_signals = portfolio_history[portfolio_history["signal"] == "BUY"]
    sell_signals = portfolio_history[portfolio_history["signal"] == "SELL"]

    axes[1].scatter(
        buy_signals.index,
        buy_signals["close_price"],
        marker="^",
        label="BUY"
    )
    axes[1].scatter(
        sell_signals.index,
        sell_signals["close_price"],
        marker="v",
        label="SELL"
    )

    axes[1].legend()

    plt.tight_layout()
    # plt.show()

    return portfolio_history

if __name__ == "__main__":
    main()