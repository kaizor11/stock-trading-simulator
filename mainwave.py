import pandas as pd
import numpy as np
import yfinance as yf
class MainWaveDetector:
    def __init__(self, df):
        self.df = df.copy()
        
    def add_indicators(self):
        # smoothed average 
        self.df['MA5'] = self.df['Close'].rolling(window=5).mean()
        self.df['MA10'] = self.df['Close'].rolling(window=10).mean()
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA60'] = self.df['Close'].rolling(window=60).mean()
 
        self.df['Vol_MA5'] = self.df['Volume'].rolling(window=5).mean()
        self.df['Vol_MA20'] = self.df['Volume'].rolling(window=20).mean()
    
        self.df['Slope_20'] = np.degrees(np.arctan(self.df['MA20'].pct_change() * 100))

    def detect_main_wave(self, strict=True):
        
        self.add_indicators()
        df = self.df
        
        #increasing faster than before
        cond_alignment = (df['MA5'] > df['MA10']) & \
                         (df['MA10'] > df['MA20']) & \
                         (df['MA20'] > df['MA60'])
        
        #strength
        cond_strength = df['Close'] > df['MA10']
        
        #volume
        cond_volume = df['Vol_MA5'] > df['Vol_MA20']
        
       
        cond_angle = df['MA20'] > df['MA20'].shift(1)

        
        if strict:
           
            df['is_main_wave'] = cond_alignment & cond_strength & cond_volume & cond_angle
        else:
           
            df['is_main_wave'] = (df['Close'] > df['MA20']) & cond_volume & cond_angle
            
        return df
if  __name__ == '__main__':

    ticker = yf.Ticker("AAPL")
    df = ticker.history(
            period="200d",       
    )
    detector = MainWaveDetector(df)
    result_df = detector.detect_main_wave(strict=True)
    cols = ['Close', 'MA5', 'MA20', 'is_main_wave']
    result_df.to_csv('result.csv')
    print(result_df[cols].tail(10))