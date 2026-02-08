import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamz import Stream
from statsmodels.tsa.arima.model import ARIMA
import warnings
import time

warnings.filterwarnings("ignore")

from config import *

def main():
    from ema_arima_only import main as ema_arima_main
    from ichimoku_cloud import main as ichimoku_main
    from mainwave_only import main as mainwave_main

    ema_arima_portfolio = ema_arima_main()
    ichimoku_portfolio = ichimoku_main()
    mainwave_portflio = mainwave_main()

    combined_portfolio = pd.DataFrame({
        "ema_arima": ema_arima_portfolio["returns"],
        "ichimoku": ichimoku_portfolio["returns"],
        "mainwave": mainwave_portflio["returns"]
    })
    
    print(combined_portfolio.corr())
    vol_ema_arima = combined_portfolio["ema_arima"].std()
    vol_ichimoku = combined_portfolio["ichimoku"].std()
    vol_mainwave = combined_portfolio["mainwave"].std()
    print(f"\nVolatility (std of returns):")    
    print(f"EMA + ARIMA: {vol_ema_arima:.4f}")
    print(f"ICHIMOKU: {vol_ichimoku:.4f}")
    print(f"MAINWAVE: {vol_mainwave:.4f}")

if __name__ == "__main__":
    main()