import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def exponential_moving_average(data, window_size):
    alpha = 2 / (window_size + 1)
    ema = []

    sma = sum(data[:window_size]) / window_size
    ema.append(sma)

    for price in data[window_size:]:
        new_ema = alpha * price + (1 - alpha) * ema[-1]
        ema.append(new_ema)

    return ema

def main():
    global data
    global capital
    global capital_hist

    ticker = yf.Ticker("TSLA")
    data = ticker.history(
        period="1y",
        interval="1h"
    )
    # print(data.shape)
    # print(data.head())
    closing_prices = data["Close"].tolist()

    # main trading logic
    capital = 100000
    capital_hist = [capital]
    window_size = 130 # 20 trading days * 6.5 hours per day = 130 hours
    ema = exponential_moving_average(closing_prices, window_size)

    position = 0

    for i in range(1, len(ema)):
        price_now = closing_prices[i + window_size - 1]
        price_prev = closing_prices[i + window_size - 2]

        # crossover detection
        if price_prev <= ema[i - 1] and price_now > ema[i]:
            if capital >= price_now:
                capital -= price_now
                position += 1

        elif price_prev >= ema[i - 1] and price_now < ema[i]:
            if position > 0:
                capital += price_now
                position -= 1

        capital_hist.append(capital + position * price_now)

    # results
    print(f"Final Capital: ${capital:.2f}")
    plt.figure(1)
    pd.Series(capital_hist).plot()
    plt.title(f"Capital Over Time\nTicker: {ticker.ticker}, Window Size: {window_size} hours")
    plt.xlabel("Time (hours)")
    plt.ylabel("Capital ($)")

    plt.figure(2)
    pd.Series(closing_prices).plot()
    pd.Series(ema).plot()
    plt.title(f"EMA for {ticker.ticker}, Window Size: {window_size} hours")
    plt.xlabel("Time (hours)")
    plt.ylabel("Closing Price ($)")
    plt.legend(["Closing Price", "EMA"])
    plt.show()

if __name__ == "__main__":
    main()