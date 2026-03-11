import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t 
import os

class FinalLongDurationBacktester:
    def __init__(self, start_capital=10000, window=30, transaction_cost=0.0002, 
                 risk_pct=0.015, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=4.0):
        
        self.start_capital = start_capital
        self.window = window
        self.tc = transaction_cost 
        
        # --- NOVÉ PARAMETRY PRO PROFESIONÁLNÍ RISK MANAGEMENT ---
        self.risk_pct = risk_pct          # Risk na obchod (např. 0.015 = 1.5 % účtu)
        self.atr_period = atr_period      # Perioda pro výpočet volatility
        self.sl_atr_mult = sl_atr_mult    # Stop Loss vzdálenost = 2x ATR
        self.tp_atr_mult = tp_atr_mult    # Take Profit vzdálenost = 4x ATR (RRR 1:2)
        
        self.data = None
        self.trades_list = [] 

    def fetch_data(self):
        print("Stahuji a připravuji data...")
        path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")
        file_path = os.path.join(path, "30m_data.csv")
        raw = pd.read_csv(file_path, sep='\t')
        
        raw.columns = raw.columns.str.strip()
        raw['DateTime'] = pd.to_datetime(raw['DateTime'])
        raw = raw.set_index('DateTime').sort_index()
        
        df = raw.copy()
        
        # Výpočty pro Studentovo t-rozdělení (statistická odchylka)
        df['Mean'] = df['Close'].rolling(window=self.window).mean()
        df['Std'] = df['Close'].rolling(window=self.window).std()
        df['t_stat'] = (df['Close'] - df['Mean']) / df['Std']
        df['Prob'] = t.cdf(df['t_stat'], df=5)
        
        # Trendový filtr
        df['SMA_Trend'] = df['Close'].rolling(window=500).mean()
        
        # --- NOVÉ: Výpočet ATR pro dynamický SL/TP ---
        df['Prev_Close'] = df['Close'].shift(1)
        df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(
            lambda row: max(row['High'] - row['Low'], 
                            abs(row['High'] - row['Prev_Close']), 
                            abs(row['Low'] - row['Prev_Close'])), axis=1
        )
        df['ATR'] = df['TR'].rolling(window=self.atr_period).mean()
        
        self.data = df.dropna().copy()
        print(f"Data připravena. Počet platných M30 svíček: {len(self.data)}")

    def run_backtest(self):
        # Správné oddělení hotovosti (cash) a počtu držených akcií (shares)
        cash = self.start_capital
        position = 0
        entry_price = 0
        shares = 0
        sl_price = 0
        tp_price = 0
        
        equity_curve = []
        self.trades_list = []
        
        # Konverze sloupců na numpy arrays pro maximální rychlost smyčky
        closes = self.data['Close'].values
        opens = self.data['Open'].values
        highs = self.data['High'].values
        lows = self.data['Low'].values
        probs = self.data['Prob'].values
        trends = self.data['SMA_Trend'].values
        atrs = self.data['ATR'].values
        dates = self.data.index

        for i in range(len(self.data)):
            if i == 0: 
                equity_curve.append(cash)
                continue

            if position == 1:
                # --- LOGIKA VÝSTUPU ---
                if lows[i] <= sl_price:
                    # SL zasažen: Prodáváme za cenu SL mínus poplatky
                    exit_price = sl_price * (1 - self.tc)
                    cash += shares * exit_price
                    
                    pnl = (exit_price - entry_price) * shares
                    self.trades_list.append({'Date': dates[i], 'PnL': pnl, 'Type': 'SL'})
                    position = 0
                    shares = 0
                
                elif highs[i] >= tp_price:
                    # TP zasažen: Prodáváme za cenu TP mínus poplatky
                    exit_price = tp_price * (1 - self.tc)
                    cash += shares * exit_price
                    
                    pnl = (exit_price - entry_price) * shares
                    self.trades_list.append({'Date': dates[i], 'PnL': pnl, 'Type': 'TP'})
                    position = 0
                    shares = 0

            if position == 0:
                # --- LOGIKA VSTUPU ---
                if probs[i-1] < 0.02 and closes[i-1] > trends[i-1]:
                    entry_price = opens[i] * (1 + self.tc)
                    current_atr = atrs[i-1]
                    
                    # 1. Výpočet vzdálenosti Stop Lossu v bodech
                    sl_dist = current_atr * self.sl_atr_mult
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + (current_atr * self.tp_atr_mult)
                    
                    if sl_dist > 0:
                        # 2. Position Sizing: Výpočet riskované částky v USD
                        current_equity = cash
                        risk_usd = current_equity * self.risk_pct
                        
                        # 3. Kolik 'shares' si můžeme dovolit, abychom při zasažení SL ztratili přesně risk_usd
                        shares = risk_usd / sl_dist
                        
                        # Ochrana proti páce: Nesmíme nakoupit za víc hotovosti, než máme
                        max_shares = cash / entry_price
                        if shares > max_shares:
                            shares = max_shares 
                        
                        # Odečteme hotovost použitou na nákup
                        cash -= shares * entry_price 
                        position = 1

            # Průběžný záznam hodnoty účtu (hotovost + aktuální tržní hodnota pozice)
            current_equity = cash + (shares * closes[i] if position == 1 else 0)
            equity_curve.append(current_equity)

        self.data['Equity'] = equity_curve

    def plot_results(self):
        trades_df = pd.DataFrame(self.trades_list)
        if trades_df.empty: 
            print("ERROR: Žádné obchody nebyly realizovány.")
            return

        peak = self.data['Equity'].cummax()
        drawdown = (self.data['Equity'] - peak) / peak
        max_dd = drawdown.min() * 100
        
        # Anualizace pro cca 14 svíček denně (M30 pro US seanci) * 252 dní
        returns = self.data['Equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(3528) if returns.std() != 0 else 0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(self.data.index, self.data['Equity'], label='Equity Curve', color='blue')
        ax1.set_title(f"Vývoj účtu (Celkově: {(self.data['Equity'].iloc[-1]/self.start_capital-1)*100:.2f}%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.fill_between(self.data.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax2.set_title(f"Drawdown (Nejhorší propad: {max_dd:.2f}%)")
        ax2.set_ylabel("%")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        print("=== VÝSLEDKY BACKTESTU ===")
        print(f"Max Drawdown:  {max_dd:.2f}%")
        print(f"Sharpe Ratio:  {sharpe:.2f}")
        print(f"Počet obchodů: {len(trades_df)}")
        print(f"Win Rate:      {(len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)) * 100:.2f}%")

if __name__ == "__main__":
    # Risk 1.5% na obchod, SL = 2x ATR, TP = 4x ATR
    bt = FinalLongDurationBacktester(risk_pct=0.015, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=4.0)
    bt.fetch_data()
    bt.run_backtest()
    bt.plot_results()