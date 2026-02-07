import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def add_n_hr_periods(n, start):
    prev = start
    new_time_vals = []
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


def main():
    global data
    global capital
    global capital_hist

    ticker = yf.Ticker("AAPL")
    data = ticker.history(
        period="200d",       
        interval="1h"
    )
    # print(data.shape)
    #print(data.head(10))
    closing_prices = data["Close"].tolist()

    # conversion line/tenkan-sen
    # (max(high) + min(low))/2
    # of the previous 9 periods
    #print(tenkan_df.head(10))
    max_highs = data['High'].rolling(9).max()
    min_lows = data['Low'].rolling(9).min()
    # min_low = tenkan_df['Low'].max()
    data['Conversion_Line'] = (max_highs + min_lows)/2.0
    #print(data.head(30))
    

    # baseline/kijun-sen
    # (max(high) + min(low))/2 
    # of the prev 26 periods
    max_highs = data['High'].rolling(26).max()
    min_lows = data['Low'].rolling(26).min()
    data['Base_Line'] = (max_highs + min_lows)/2.0
    print(data.head(30))


    # leading span A (senkou span A)
    leading_span_index = pd.concat([pd.Series(data.index), add_n_hr_periods(26, data.index[-1])])
    print(pd.Series(data.index))
    print(leading_span_index)
    senkou_df = data[['Conversion_Line', 'Base_Line']]
    senkou_df['Senkou_A'] = (senkou_df['Conversion_Line'] + senkou_df['Base_Line']) / 2.0

    # leading span B
    max_highs = data['High'].rolling(52).max()
    min_lows = data['Low'].rolling(52).min()
    senkou_df['Senkou_B'] = (max_highs + min_lows) / 2.0

    senkou_df = senkou_df.reindex(leading_span_index)
    senkou_df['Senkou_A'] = senkou_df['Senkou_A'].shift(26)
    senkou_df['Senkou_B'] = senkou_df['Senkou_B'].shift(26)
    senkou_df = senkou_df[['Senkou_A', 'Senkou_B']]
    #print(senkou_df.tail(50))

    # lagging span (chikou span)
    data['Lagging_Span'] = (max_highs + min_lows) / 2.0
    data['Lagging_Span'] = data['Lagging_Span'].shift(26)

    # results
    fig, ax = plt.subplots(figsize=(12,12))
    sns.set_style("whitegrid")
    plot_df = data[['Close', 'Conversion_Line', 'Base_Line', 'Lagging_Span']]

    sns.lineplot(data=plot_df,ax=ax)
    sns.lineplot(data=senkou_df,ax=ax)
    plt.xlabel("Time (hours)")
    plt.xlim(data.index[-50], senkou_df.index[-20])
    plt.grid()

    #fill green where leading signals indicate uptrend
    plt.fill_between(senkou_df.index, 
                     senkou_df['Senkou_A'], 
                     senkou_df['Senkou_B'], 
                     where=np.where(senkou_df['Senkou_A'] > senkou_df['Senkou_B'],1,0),
                     color="green", 
                     alpha=.2)
    
    #fill red where leading signals indicate downtrend
    plt.fill_between(senkou_df.index, 
                    senkou_df['Senkou_A'], 
                    senkou_df['Senkou_B'], 
                    where=np.where(senkou_df['Senkou_A'] <= senkou_df['Senkou_B'],1,0),
                    color="red", 
                    alpha=.2)
    plt.ylabel("Price ($)")

    plt.show()
    

if __name__ == "__main__":
    main()