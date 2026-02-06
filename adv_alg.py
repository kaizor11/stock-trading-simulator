import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamz import Stream
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

TICKER = "AAPL"
PERIOD = "1y"
INTERVAL = "1d"

EMA_SPAN = 20          # EMA window
ARIMA_ORDER = (5, 1, 0)  # (p, d, q)
STREAM_WINDOW = 50     # number of data points used for ARIMA

INITIAL_CAPITAL = 100_000

def compute_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def arima_forecast(price_series, order=(5, 1, 0)):
    model = ARIMA(price_series, order=order)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=1)
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
        shares_to_buy = portfolio["cash"] // current_price
        portfolio["shares"] = shares_to_buy
        portfolio["cash"] -= shares_to_buy * current_price
        portfolio["position"] = "LONG"
        signal = "BUY"
        # print(f"BUY @ {current_price:.2f}")

    # SELL signal
    elif (
        portfolio["position"] == "LONG"
        and (current_price < current_ema or arima_pred < current_price)
    ):
        portfolio["cash"] += portfolio["shares"] * current_price
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

    # Keep buffer size fixed
    if len(buffer) > STREAM_WINDOW:
        buffer.pop(0)

    df = pd.DataFrame(buffer).set_index("timestamp")
    df["EMA"] = compute_ema(df["Close"], EMA_SPAN)

    # Only run ARIMA if we have enough data
    if len(df) >= STREAM_WINDOW:
        df["ARIMA_Pred"] = np.nan
        arima_price = arima_forecast(df["Close"], ARIMA_ORDER)
        df.iloc[-1, df.columns.get_loc("ARIMA_Pred")] = arima_price

        trading_strategy(df)

def main():
    global data
    global portfolio
    global portfolio_history
    global buffer
    
    ticker = yf.Ticker(TICKER)
    data = ticker.history(
        period=PERIOD,
        interval=INTERVAL
    )
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