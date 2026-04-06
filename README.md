# Algorithmic Stock Trading Simulator
Use a combination of three algorithms to simulate trading over stock ticker.

### config.py
Used to configure the settings for the combined stock trading algorithm.
Input:
- Stock to trade over
- Period of historical data to use
- Interval of each observation
- Estimated broker fee
- Initial capital amount
- Size of streaming data window
- Proportion of funds to allocate in EMA/ARIMA strategy
- Proportion of funds to allocate in Ichimoku Cloud strategy
- Proportion of funds to allocate in Mainwave strategy

### combined_alg.py
Run combined_alg.py to perform the trading outlined in config.py. 
Displays a graph of buy/sell points along the ticker line and outputs these final metrics:
- Final Equity
- Total Return
- Annualized Return
- Sharpe Ratio
- Max Drawdown

### Individual algorithms
Refer to ema_arima_only.py, mainwave_only.py, and ichimoku_cloud.py for calculations and trading logic for each individual strategy.