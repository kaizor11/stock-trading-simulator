import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamz import Stream
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time

warnings.filterwarnings("ignore")

TICKER = "AAPL"
PERIOD = "6mo"
INTERVAL = "1h"
FEE = 0.0005        # 0.05% per trade

EMA_SPAN = int(20 * 6.5)         # 20 days * 6.5 trading hours per day
ARIMA_ORDER = (2, 0, 1)          # (p, d, q)
STREAM_WINDOW = int(50 * 6.5)    # number of data points used for ARIMA

INITIAL_CAPITAL = 100_000

def compute_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def arima_forecast(price_series):
    model = ARIMA(price_series, order=ARIMA_ORDER)
    model = model.fit()
    forecast = model.forecast(steps=1)
    return forecast.iloc[0]

def trading_strategy(df):
    """
    - BUY signal:
        price > EMA AND ARIMA forecast > current price
    - SELL signal:
        price < EMA OR ARIMA forecast < current price
    """

    current_price = float(df["Close"].iloc[-1])
    current_ema = float(df["EMA"].iloc[-1])
    arima_pred = float(df["ARIMA_Pred"].iloc[-1])

    timestamp = df.index[-1]
    signal = "HOLD"

    # BUY signal
    if (
        portfolio["position"] == "FLAT"
        and current_price > current_ema
        and arima_pred > current_price
    ):
        shares_to_buy = int(portfolio["cash"] // (current_price * (1 + FEE)))
        cost = shares_to_buy * current_price * (1 + FEE)
        portfolio["shares"] = shares_to_buy
        portfolio["cash"] -= cost
        portfolio["position"] = "LONG"
        signal = "BUY"
        # print(f"BUY @ {current_price:.2f}")

    # SELL signal
    elif (
        portfolio["position"] == "LONG"
        and (current_price < current_ema or arima_pred < current_price)
    ):
        proceeds = portfolio["shares"] * current_price * (1 - FEE)
        portfolio["cash"] += proceeds
        portfolio["shares"] = 0
        portfolio["position"] = "FLAT"
        signal = "SELL"
        # print(f"SELL @ {current_price:.2f}")

    total_value = portfolio["cash"] + portfolio["shares"] * current_price
    # print(f"Portfolio Value: {total_value:.2f}")

    portfolio_history.append({
        "timestamp": timestamp,
        "close_price": current_price,
        "signal": signal,
        "position": portfolio["position"],
        "shares": portfolio["shares"],
        "cash": portfolio["cash"],
        "portfolio_value": total_value,
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
    portfolio_return = (portfolio_history[-1]["portfolio_value"] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annualized_return = (1 + portfolio_return) ** ((252 * 6.5) / len(portfolio_history)) - 1 # approx 252 trading days per year
    print(f"Porfolio Return: {portfolio_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")

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
    data = data.between_time("09:30", "16:00")  # filter for regular trading hours
    data = data[["Close"]].dropna()

    portfolio = {
        "cash": INITIAL_CAPITAL,
        "shares": 0,
        "position": "FLAT"  # FLAT or LONG
    }

    portfolio_history = []

    price_stream = Stream()

    buffer = []

    price_stream.sink(process_stream)

    for timestamp, row in data.iterrows():
        price_stream.emit({
        "timestamp": timestamp,
        "Close": float(row["Close"])
    })

    metrics()
    print(f"Execution Time: {time.time() - start:.2f} seconds")

    # plot results
    history_df = pd.DataFrame(portfolio_history)
    history_df.set_index("timestamp", inplace=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # PLOT: portfolio value over time
    axes[0].plot(
        history_df.index,
        history_df["portfolio_value"],
        label="Portfolio Value"
    )
    axes[0].set_title("Portfolio Value Over Time")
    axes[0].set_xlabel("Time")

    # PLOT: price and signals
    axes[1].plot(
        history_df.index,
        history_df["close_price"],
        label=f"{TICKER} Close Price"
    )
    axes[1].set_title(f"{TICKER} Price Over Time")
    axes[1].set_xlabel("Time")

    buy_signals = history_df[history_df["signal"] == "BUY"]
    sell_signals = history_df[history_df["signal"] == "SELL"]

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
    plt.show()

if __name__ == "__main__":
    main()