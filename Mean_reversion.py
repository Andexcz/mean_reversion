import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t 
import os

class FinalLongDurationBacktester:
    def __init__(self, start_capital=10000, window=30, transaction_cost=0.0002, sl_pct=0.02):
        self.start_capital = start_capital
        self.window = window
        self.tc = transaction_cost 
        self.sl_pct = sl_pct  
        self.tp_pct = sl_pct * 2  # RR 1:2
        self.data = None
        self.trades_list = [] 

    def fetch_data(self):
        print("Stahuji data...")
        path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")
        file_path = os.path.join(path, "30m_data.csv")
        raw = pd.read_csv(file_path, sep='\t')
        raw.columns = raw.columns.str.strip()
        raw['DateTime'] = pd.to_datetime(raw['DateTime'])
        raw = raw.set_index('DateTime').sort_index()
        
        df = raw.copy()
        df['Mean'] = df['Close'].rolling(window=self.window).mean()
        df['Std'] = df['Close'].rolling(window=self.window).std()
        df['t_stat'] = (df['Close'] - df['Mean']) / df['Std']
        df['Prob'] = t.cdf(df['t_stat'], df=5) 
        df['SMA_Trend'] = df['Close'].rolling(window=500).mean()
        
        self.data = df.dropna()
        print(f"Data připravena. Svíček: {len(self.data)}")

    def run_backtest(self):
        capital = self.start_capital
        position = 0
        entry_price = 0
        shares = 0
        
        equity_curve = []
        self.trades_list = []
        
        closes = self.data['Close'].values
        opens = self.data['Open'].values
        highs = self.data['High'].values
        lows = self.data['Low'].values
        probs = self.data['Prob'].values
        trends = self.data['SMA_Trend'].values
        dates = self.data.index

        for i in range(len(self.data)):
            if i == 0: 
                equity_curve.append(capital)
                continue

            if position == 1:
                sl_price = entry_price * (1 - self.sl_pct)
                tp_price = entry_price * (1 + self.tp_pct)
                
                # Exit na Stop Loss
                if lows[i] <= sl_price:
                    exit_price = sl_price * (1 - self.tc)
                    capital = shares * exit_price
                    self.trades_list.append({'Date': dates[i], 'PnL': capital - prev_cap, 'Type': 'SL'})
                    position = 0
                    shares = 0
                
                # Exit na Take Profit
                elif highs[i] >= tp_price:
                    exit_price = tp_price * (1 - self.tc)
                    capital = shares * exit_price
                    self.trades_list.append({'Date': dates[i], 'PnL': capital - prev_cap, 'Type': 'TP'})
                    position = 0
                    shares = 0

            if position == 0:
                if probs[i-1] < 0.02 and closes[i-1] > trends[i-1]:
                    entry_price = opens[i] * (1 + self.tc)
                    shares = capital / entry_price
                    prev_cap = capital
                    position = 1

            equity_curve.append(shares * closes[i] if position == 1 else capital)

        self.data['Equity'] = equity_curve

    def plot_results(self):
        trades_df = pd.DataFrame(self.trades_list)
        if trades_df.empty: return

        # Výpočty metrik
        peak = self.data['Equity'].cummax()
        drawdown = (self.data['Equity'] - peak) / peak
        max_dd = drawdown.min() * 100
        
        returns = self.data['Equity'].pct_change().dropna()
        # Sharpe Ratio: (průměrný výnos / std výnosu) * odmocnina z (počet svíček za rok)
        # Předpokládáme 252 dní * cca 14 hodin obchodování (28 svíček) = 7056 svíček/rok
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 28) if returns.std() != 0 else 0

        # Grafy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(self.data.index, self.data['Equity'], label='Equity Curve', color='blue')
        ax1.set_title(f"Equity (Zhodnocení: {(self.data['Equity'].iloc[-1]/self.start_capital-1)*100:.2f}%)")
        ax1.legend()
        
        ax2.fill_between(self.data.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax2.set_title(f"Drawdown (Max DD: {max_dd:.2f}%)")
        ax2.set_ylabel("%")
        
        plt.tight_layout()
        plt.show()

        print(f"--- FINÁLNÍ STATISTIKY ---")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Počet obchodů: {len(trades_df)}")
        print(f"Win Rate:     {(len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)) * 100:.2f}%")

if __name__ == "__main__":
    bt = FinalLongDurationBacktester(sl_pct=0.02) # SL 2%, TP 4%
    bt.fetch_data()
    bt.run_backtest()
    bt.plot_results()