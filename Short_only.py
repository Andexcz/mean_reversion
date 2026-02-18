import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. NASTAVENÍ STRATEGIE
# ==========================================
SYMBOL = "NQ_100"
LOOKBACK_WINDOW = 120     
ATR_PERIOD = 5          
SL_MULTIPLIER = 0.24
INITIAL_CAPITAL = 10000.0 
RISK_PER_TRADE = 0.012

# --- 1.1 NASTAVENÍ ČASOVÉHO RÁMCE (ZDE ZMĚŇ DATUM) ---
START_DATE = "2016-01-01"  # Formát YYYY-MM-DD
END_DATE   = "2026-2-15"  # Formát YYYY-MM-DD
# -----------------------------------------------------

START_TIME = "16:30"      
LAST_SIGNAL_TIME = "22:00" 

print(f"--- START BACKTESTU ({START_DATE} až {END_DATE}) ---")

# ==========================================
# 2. NAČTENÍ DAT
# ==========================================
path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")
file_path = os.path.join(path, "30m_data.csv")

df = pd.read_csv(file_path, sep='\t')
df.columns = df.columns.str.strip()
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.set_index('DateTime').sort_index()

# --- FILTROVÁNÍ PODLE DATA ---
df = df.loc[START_DATE:END_DATE]
# -----------------------------

print(f"Data načtena: od {df.index.min()} do {df.index.max()}")

# ==========================================
# 3. INDIKÁTORY
# ==========================================
# Denní ATR
# Resample uděláme, ale pozor - denní svíčky se teď budou počítat podle času dat.
# Pro účely ATR (volatility) je to OK.
daily_df = df.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
daily_df['TR'] = np.maximum(daily_df['High'] - daily_df['Low'], 
                            np.maximum(abs(daily_df['High'] - daily_df['Close'].shift(1)), 
                                       abs(daily_df['Low'] - daily_df['Close'].shift(1))))
daily_df['ATR'] = daily_df['TR'].rolling(window=ATR_PERIOD).mean()
daily_df['Prev_Day_ATR'] = daily_df['ATR'].shift(1)

df['Date_Only'] = df.index.normalize()
df = df.merge(daily_df[['Prev_Day_ATR']], left_on='Date_Only', right_index=True, how='left')

# Percentil a Čas
df['Bar_Size'] = abs(df['Close'] - df['Open'])
df['Time_Str'] = df.index.strftime('%H:%M')

print("Počítám statistiky (to chvíli potrvá)...")
df['Rolling_95_Pct'] = (
    df.groupby('Time_Str')['Bar_Size']
    .transform(lambda x: x.shift(1).rolling(window=LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).quantile(0.95))
)

# ==========================================
# 4. GENERACE SIGNÁLŮ
# ==========================================
df['Signal_Short'] = (
    (df['Bar_Size'] > df['Rolling_95_Pct']) &      # Velká svíčka
    (df['Close'] < df['Open']) &                   # Červená
    (df['Prev_Day_ATR'].notna()) &                 # Máme ATR
    (df['Time_Str'] >= START_TIME) &               # <--- 16:30
    (df['Time_Str'] <= LAST_SIGNAL_TIME)           # <--- 22:00
)

print(f"Počet nalezených signálů: {df['Signal_Short'].sum()}")

# ==========================================
# 5. BACKTEST LOOP
# ==========================================
trades = []
current_capital = INITIAL_CAPITAL

for i in range(len(df) - 1):
    current_bar = df.iloc[i]
    
    if current_bar['Signal_Short']:
        next_bar = df.iloc[i + 1]
        
        # Kontrola kontinuity (aby obchod nebyl přes víkend)
        if (next_bar.name - current_bar.name).total_seconds() > 1800:
            continue

        entry_price = next_bar['Open']
        atr = current_bar['Prev_Day_ATR']
        
        # Pojistka proti NaN v ATR
        if pd.isna(atr) or atr == 0:
            continue

        sl_distance = SL_MULTIPLIER * atr
        stop_loss_price = entry_price + sl_distance
        
        # Risk management
        risk_dollars = current_capital * RISK_PER_TRADE
        if sl_distance > 0:
            position_size = risk_dollars / sl_distance
        else:
            continue 

        # Průběh obchodu
        if next_bar['High'] >= stop_loss_price:
            exit_price = stop_loss_price
        else:
            exit_price = next_bar['Close']
            
        pnl_dollars = (entry_price - exit_price) * position_size
        current_capital += pnl_dollars
        
        trades.append({
            'Date': next_bar.name,
            'PnL_USD': pnl_dollars,
            'Capital': current_capital
        })

# ==========================================
# 6. VÝSLEDKY
# ==========================================
if len(trades) > 0:
    results = pd.DataFrame(trades)
    
    # Statistiky
    results['Peak_Capital'] = results['Capital'].cummax()
    results['Drawdown_Pct'] = ((results['Capital'] - results['Peak_Capital']) / results['Peak_Capital']) * 100
    total_ret = ((current_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    max_dd = results['Drawdown_Pct'].min()
    
    print(f"\n--- VÝSLEDKY STRATEGIE ---")
    print(f"Počáteční kapitál: ${INITIAL_CAPITAL}")
    print(f"Konečný kapitál:   ${current_capital:.2f}")
    print(f"Zisk:              {total_ret:.2f}%")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print(f"Počet obchodů:     {len(results)}")

    # Grafy
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Equity
    ax1.plot(results['Date'], results['Capital'], color='blue')
    ax1.set_title('Equity Curve (Vývoj kapitálu)')
    ax1.set_ylabel('USD')
    
    # Drawdown
    ax2.fill_between(results['Date'], results['Drawdown_Pct'], 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown (Poklesy)')
    ax2.set_ylabel('%')
    
    # PnL
    colors = ['green' if x >= 0 else 'red' for x in results['PnL_USD']]
    ax3.bar(results['Date'], results['PnL_USD'], color=colors, width=0.05)
    ax3.set_title('Zisk/Ztráta na obchod')
    ax3.set_ylabel('USD')
    
    plt.tight_layout()
    plt.show()

else:
    print("Žádné obchody. Zkontroluj, jestli je LOOKBACK_WINDOW (120) menší než délka dostupných dat.")

import numpy as np

# ==========================================
# DETAILNÍ OPTIMALIZACE SL_MULTIPLIER
# ==========================================

# Generujeme rozsah od 0.1 do 1.0 s krokem 0.01
# np.arange(start, stop, step) - stop je exkluzivní, proto 1.01
SL_TEST_RANGE = np.around(np.arange(0.1, 1.01, 0.01), 2)

optimization_results = []

print(f"Spouštím detailní optimalizaci pro {len(SL_TEST_RANGE)} variant...")

for current_sl_mult in SL_TEST_RANGE:
    current_cap = INITIAL_CAPITAL
    trade_count = 0
    equity_history = [INITIAL_CAPITAL]
    
    # Pro každou variantu projedeme historii
    for i in range(len(df) - 1):
        bar = df.iloc[i]
        
        if bar['Signal_Short']:
            next_bar = df.iloc[i + 1]
            # Kontrola víkendu / kontinuity dat
            if (next_bar.name - bar.name).total_seconds() > 1800: continue
            
            entry = next_bar['Open']
            atr = bar['Prev_Day_ATR']
            if pd.isna(atr) or atr == 0: continue
            
            # Dynamický SL podle aktuálního násobku
            dist = current_sl_mult * atr
            sl_price = entry + dist
            
            # Risk Management (Fixed fractional risk)
            pos_size = (current_cap * RISK_PER_TRADE) / dist
            
            # Logika výstupu: Pokud High svíčky lízne SL, končíme na SL. Jinak na Close.
            if next_bar['High'] >= sl_price:
                exit_p = sl_price
            else:
                exit_p = next_bar['Close']
            
            pnl = (entry - exit_p) * pos_size
            current_cap += pnl
            trade_count += 1
            equity_history.append(current_cap)
            
    # Výpočet Max Drawdownu pro tuto variantu
    equity_series = pd.Series(equity_history)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() * 100
    
    total_profit = ((current_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    optimization_results.append({
        'SL_Mult': current_sl_mult, 
        'Profit': total_profit, 
        'MaxDD': max_dd,
        'Trades': trade_count
    })

# Převedeme na DataFrame pro snadnou analýzu
opt_df = pd.DataFrame(optimization_results)
best = opt_df.loc[opt_df['Profit'].idxmax()]

print("\n" + "="*45)
print(f"VÝSLEDEK OPTIMALIZACE (Rozsah 0.1 - 1.0)")
print(f"Nejlepší SL_MULTIPLIER: {best['SL_Mult']}")
print(f"Maximální zisk:        {best['Profit']:.2f}%")
print(f"Max Drawdown u vítěze: {best['MaxDD']:.2f}%")
print(f"Počet obchodů:         {int(best['Trades'])}")
print("="*45)

# Vizualizace
fig, ax1 = plt.subplots(figsize=(12, 6))

# Graf zisku
ax1.set_xlabel('ATR Multiplier (SL)')
ax1.set_ylabel('Celkový zisk (%)', color='tab:blue')
ax1.plot(opt_df['SL_Mult'], opt_df['Profit'], color='tab:blue', linewidth=2, label='Zisk')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

# Druhá osa pro Drawdown
ax2 = ax1.twinx()
ax2.set_ylabel('Max Drawdown (%)', color='tab:red')
ax2.plot(opt_df['SL_Mult'], opt_df['MaxDD'], color='tab:red', linestyle='--', alpha=0.6, label='Max DD')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Optimalizace: Zisk vs. Riziko (Krok 0.01)')
fig.tight_layout()
plt.show()

# ... (kód po dokončení optimalizační smyčky)

# Převedeme na DataFrame
opt_df = pd.DataFrame(optimization_results)

# Najdeme nejlepší výsledek pro zisk
best_index = opt_df['Profit'].idxmax()
best_sl = opt_df.loc[best_index, 'SL_Mult']
best_profit = opt_df.loc[best_index, 'Profit']
best_dd = opt_df.loc[best_index, 'MaxDD']

# Vytvoření grafu
fig, ax1 = plt.subplots(figsize=(12, 7))

# 1. osa pro Zisk
ax1.set_xlabel('ATR Multiplier (Stop Loss)')
ax1.set_ylabel('Celkový zisk (%)', color='tab:blue', fontweight='bold')
line1, = ax1.plot(opt_df['SL_Mult'], opt_df['Profit'], color='tab:blue', linewidth=2.5, label='Zisk (%)')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3, linestyle='--')

# 2. osa pro Drawdown
ax2 = ax1.twinx()
ax2.set_ylabel('Max Drawdown (%)', color='tab:red', fontweight='bold')
line2, = ax2.plot(opt_df['SL_Mult'], opt_df['MaxDD'], color='tab:red', linestyle='-', alpha=0.5, label='Max Drawdown (%)')
ax2.tick_params(axis='y', labelcolor='tab:red')

# --- PŘIDÁNÍ KOLMÉ ČÁRY PRO OPTIMUM ---
plt.axvline(x=best_sl, color='red', linestyle='--', linewidth=2, label=f'Optimum (SL {best_sl})')

# Sjednocení legendy do jedné
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title(f'Optimalizace: Hledání rovnováhy mezi ziskem a rizikem\nBest SL: {best_sl} | Profit: {best_profit:.2f}% | DD: {best_dd:.2f}%', 
          pad=20, fontweight='bold')

fig.tight_layout()
plt.show()

# Výpis do terminálu
print(f"\n" + "="*45)
print(f"ANALÝZA DOKONČENA")
print(f"Optimální SL Multiplier: {best_sl}")
print(f"Odpovídající zisk:       {best_profit:.2f}%")
print(f"Odpovídající Drawdown:   {best_dd:.2f}%")
print("="*45)