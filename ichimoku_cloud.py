import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from streamz import Stream
import time

# TICKER = "AAPL"
# PERIOD = "1y"
# INTERVAL = "1d"
# FEE = 0.0005        # 0.05% per trade
# STREAM_WINDOW = 52

# INITIAL_CAPITAL = 100_000
from config import *

def add_n_periods(n, start, interval="d"):
    prev = start
    new_time_vals = []
    if interval == "d": #daily data
        for i in range(n):
            prev = prev + pd.offsets.BDay(1)
            new_time_vals.append(prev)
    elif interval == "h": #hourly data
        for i in range(n):
            # if before 15:30: +1 hour
            if prev.hour < 15:
                prev = prev + pd.Timedelta(hours=1)

            # if 15:30: +1 biz day, set to 9:30
            else:
                prev = prev + pd.offsets.BDay(1)
                prev = prev.replace(hour=9,minute=30)
            new_time_vals.append(prev)
    return pd.Series(new_time_vals)


def run_ichimoku_trade(df, senkou_df):
    #assume all-in trading
    # conservative strategy
    # BUY if:
    #   opening price above conversion line
    #   base line below conversion line
    #   senkou span A above B
    # SELL if:
    #   opening price below conversion line 
    #   base line above conversion line
    #   senkou span B above A
    # else HOLD

    current_price = float(df["Open"].iloc[-1])
    conversion = float(df["Conversion_Line"].iloc[-1])
    base = float(df["Base_Line"].iloc[-1])
    timestamp = df.index[-1]
    signal = "HOLD"

    senkou_A = senkou_df["Senkou_A"].iloc[-27]
    senkou_B = senkou_df["Senkou_B"].iloc[-27]

    # indicates crossing of price into cloud
    # 0 = no cross
    # -1 = entering downwards
    # 1 = exiting upwards
    cross_cloud = 0

    # likewise, indicates crossing of base line and conversion line
    # 0 = no cross
    # -1 = conversion line crosses under
    # 1 = conversion line crosses over
    cross_lines = 0

    # check if value in the cloud
    if (current_price >= senkou_A and current_price <= senkou_B) or (current_price <= senkou_A and current_price >= senkou_B):
        # check if new crossover
        #   price from day before would be outside upper or lower boundary
        if df["Open"].iloc[-2] >= max(senkou_df["Senkou_A"].iloc[-28], senkou_df["Senkou_B"].iloc[-28]): # outside upper boundary, indicates decline and sell signal
            cross_cloud = -1
        # elif  df["Open"].iloc[-2] >= min(senkou_A, senkou_B): # outside lower boundary, indicates increase
        #     pass
    else:
        # check if price is newly leaving
        #   price from day before would be within bounds
        if df["Open"].iloc[-2] <= min(senkou_df["Senkou_A"].iloc[-28], senkou_df["Senkou_B"].iloc[-28]):
            cross_cloud = 1

    # see if lines crossed 
    if conversion > base:
        if df["Conversion_Line"].iloc[-2] <  df["Base_Line"].iloc[-2]: #indicates cross in positive direction
            cross_lines = 1
    elif base > conversion:
          if df["Conversion_Line"].iloc[-2] >  df["Base_Line"].iloc[-2]: #indicates cross in negative direction
            cross_lines = -1      
    
    # BUY signal
    if (
        portfolio["position"] == "FLAT"
        #and conversion > base
        #and current_price < conversion
        and (cross_cloud == 1 or cross_lines == 1)
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
        #and (current_price < conversion)
        and (conversion < base)
        #and (cross_cloud == -1 or cross_lines == -1)
        and current_price < min(senkou_A, senkou_B)
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
        "equity": total_value,
    })

def process_stream(row):
    buffer.append(row)

    # keep buffer size fixed
    if len(buffer) > STREAM_WINDOW:
        buffer.pop(0)

    df = pd.DataFrame(buffer).set_index("timestamp")
    # print(df.head())
 
    # only run ichimoku if enough data
    if df.shape[0] >= STREAM_WINDOW - 26:
        df["Conversion_Line"] = df["Base_Line"] = np.nan

        # conversion line/tenkan-sen
        # (max(high) + min(low))/2
        # of the previous 9 periods
        #print(tenkan_df.head(10))
        max_highs = df['High'].rolling(9).max()
        min_lows = df['Low'].rolling(9).min()
        # min_low = tenkan_df['Low'].max()
        df['Conversion_Line'] = (max_highs + min_lows)/2.0
        #print(data.head(30))

        # baseline/kijun-sen
        # (max(high) + min(low))/2 
        # of the prev 26 periods
        max_highs = df['High'].rolling(26).max()
        min_lows = df['Low'].rolling(26).min()
        df['Base_Line'] = (max_highs + min_lows)/2.0
        #print(df.head(30))

        # leading span A (senkou span A)
        leading_span_index = pd.concat([pd.Series(df.index), add_n_periods(26, df.index[-1])])
        #print(pd.Series(df.index))
        #print(leading_span_index)
        senkou_df = df[['Conversion_Line', 'Base_Line']]
        senkou_df['Senkou_A'] = (senkou_df['Conversion_Line'] + senkou_df['Base_Line']) / 2.0

        # leading span B
        max_highs = df['High'].rolling(52).max()
        min_lows = df['Low'].rolling(52).min()
        senkou_df['Senkou_B'] = (max_highs + min_lows) / 2.0

        senkou_df = senkou_df.reindex(leading_span_index)
        senkou_df['Senkou_A'] = senkou_df['Senkou_A'].shift(26)
        senkou_df['Senkou_B'] = senkou_df['Senkou_B'].shift(26)
        senkou_df = senkou_df[['Senkou_A', 'Senkou_B']]
        #print(senkou_df.tail(50))

        # lagging span (chikou span)
        df['Lagging_Span'] = df['Close'].shift(-26)
        
        run_ichimoku_trade(df, senkou_df)

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
    print(f"\nICHIMOKU:")
    print(f"Initial Capital: {INITIAL_CAPITAL * ICHIMOKU_SPLIT}")
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
    buffer = []
    
    ticker = yf.Ticker(TICKER)
    data = ticker.history(
        period=PERIOD,
        interval=INTERVAL
    )
    #data = data[["Close"]].dropna()

    portfolio = {
        "cash": INITIAL_CAPITAL * ICHIMOKU_SPLIT,
        "shares": 0,
        "position": "FLAT"  # FLAT or LONG
    }

    portfolio_history = []

    price_stream = Stream()

    price_stream.sink(process_stream)

    for timestamp, row in data.iterrows():
        price_stream.emit({
        "timestamp": timestamp,
        "High": float(row["High"]),
        "Low": float(row["Low"]),
        "Close": float(row["Close"]),
        "Open": float(row["Open"])
    })

    # # results
    # fig, ax = plt.subplots(figsize=(12,12))
    # sns.set_style("whitegrid")
    # plot_df = data[['Open', 'Conversion_Line', 'Base_Line', 'Lagging_Span']]

    # sns.lineplot(data=plot_df,ax=ax)
    # sns.lineplot(data=senkou_df,ax=ax)
    # plt.xlabel("Time (hours)")
    # plt.xlim(data.index[-50], senkou_df.index[-20])
    # plt.grid()

    # #fill green where leading signals indicate uptrend
    # plt.fill_between(senkou_df.index, 
    #                  senkou_df['Senkou_A'], 
    #                  senkou_df['Senkou_B'], 
    #                  where=np.where(senkou_df['Senkou_A'] > senkou_df['Senkou_B'],1,0),
    #                  color="green", 
    #                  alpha=.2)
    
    # #fill red where leading signals indicate downtrend
    # plt.fill_between(senkou_df.index, 
    #                 senkou_df['Senkou_A'], 
    #                 senkou_df['Senkou_B'], 
    #                 where=np.where(senkou_df['Senkou_A'] <= senkou_df['Senkou_B'],1,0),
    #                 color="red", 
    #                 alpha=.2)
    # plt.ylabel("Price ($)")

    # plt.show()

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