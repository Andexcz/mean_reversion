import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t 

class MeanReversionBacktester:
    def __init__(self, ticker, start_date, end_date, window=20, transaction_cost=0.001, degrees_of_freedom=3):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.window = window
        self.tc = transaction_cost
        self.df_stat = degrees_of_freedom
        self.data = None

    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        raw = yf.download(self.ticker, start=self.start, end=self.end, progress=False, auto_adjust=True)
        
        #Just to fix the yfinance new structure (AI saved me here)
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                raw = raw.xs(self.ticker, level=1, axis=1)
            except:
                raw.columns = raw.columns.droplevel(1)

        if 'Close' in raw.columns:
            close_prices = raw['Close']
        else:
            close_prices = raw.iloc[:, 0]
            
        close_prices = close_prices.squeeze()
        
        # Rebuild clean dataframe
        df = pd.DataFrame(close_prices)
        df.columns = ['Close']
        
        df['Mean'] = df['Close'].rolling(window=self.window).mean()
        df['Std'] = df['Close'].rolling(window=self.window).std()
        df['t_stat'] = (df['Close'] - df['Mean']) / df['Std']
        df['Prob'] = t.cdf(df['t_stat'], df=self.df_stat)
        
        self.data = df.dropna()
        print(f"Data ready. Rows: {len(self.data)}")

    def run_backtest(self):
        if self.data is None: raise Exception("Data not loaded.")
        
        position = 0 
        positions = []
        probs = self.data['Prob'].values

        for p in probs:        
            # LOOKING FOR ENTRY
            if position == 0:
                if p < 0.10:
                    position = 1   # Buy
                elif p > 0.90:
                    position = -1  # Short

            # LOOKING FOR EXIT (Long-case)
            elif position == 1:
                if p > 0.5:
                    position = 0   
                elif p < 0.01:    
                    position = 0

             # LOOKING FOR EXIT (Short-case)
            elif position == -1:
                if p <= 0.5:
                    position = 0  
                elif p > 0.99:    
                    position = 0

            positions.append(position)
        
        self.data['Position'] = positions
        self.data['Position_Lagged'] = self.data['Position'].shift(1)
        self._calculate_returns()

    def _calculate_returns(self):
        self.data['Market_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Strategy_Gross'] = self.data['Position_Lagged'] * self.data['Market_Returns']
        trades = self.data['Position_Lagged'].diff().abs()
        self.data['Strategy_Net'] = self.data['Strategy_Gross'] - (trades * self.tc)
        self.data['Cumulative_Market'] = self.data['Market_Returns'].cumsum().apply(np.exp)
        self.data['Cumulative_Strategy'] = self.data['Strategy_Net'].cumsum().apply(np.exp)

    def plot_results(self):
        # DATA PREPARATION
        # Convert Log Returns back to Simple Percentage for plotting (easier to read)
        daily_pct = np.exp(self.data['Strategy_Net']) - 1
        
        # Calculate Drawdown
        peak = self.data['Cumulative_Strategy'].cummax()
        drawdown = (self.data['Cumulative_Strategy'] - peak) / peak

        # 2. SETUP PLOT 
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # --- PANEL 1: EQUITY CURVE ---
        ax1.plot(self.data['Cumulative_Market'], label='Buy & Hold (SPY)', color='gray', alpha=0.5, linewidth=1.5)
        ax1.plot(self.data['Cumulative_Strategy'], label='Student-t Strategy', color='blue', linewidth=2)
        ax1.set_title(f'Equity Curve: {self.ticker}')
        ax1.set_ylabel('Growth of $1')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # --- PANEL 2: DAILY PnL ---
        colors = ['green' if x >= 0 else 'red' for x in daily_pct]
        ax2.bar(daily_pct.index, daily_pct, color=colors, width=1.0)
        ax2.set_title('Daily Profit/Loss (%)')
        ax2.set_ylabel('Daily Return')
        ax2.grid(True, alpha=0.3)
        # Add a horizontal line at 0
        ax2.axhline(0, color='black', linewidth=0.5)

        # --- PANEL 3: DRAWDOWN ---
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown (Risk)')
        ax3.set_ylabel('% Below Peak')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        # Print Summary Stats
        total_return = self.data['Cumulative_Strategy'].iloc[-1] - 1
        max_dd = drawdown.min()
        print(f"--- FINAL STATS ---")
        print(f"Total Return: {total_return:.2%}")
        print(f"Max Drawdown: {max_dd:.2%}")

# --- EXECUTION ---
if __name__ == "__main__":
    backtest = MeanReversionBacktester("SPY", "2020-01-01", "2026-01-01")
    backtest.fetch_data()
    backtest.run_backtest()
    backtest.plot_results()