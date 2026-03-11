import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ==========================================
# 1. KONFIGURACE A BEZPEČNOSTNÍ LIMITY
# ==========================================
SYMBOL = "NDX"
MAGIC_NUMBER = 123456
LOOKBACK_WINDOW = 120
ATR_PERIOD = 5
SL_MULTIPLIER = 0.24
RISK_PER_TRADE = 0.012  # Riziko 1,2 % z účtu

START_TIME = "16:30"
LAST_SIGNAL_TIME = "23:00"
HARD_CLOSE_TIME = "23:30"


MAX_RETRIES = 5           # Kolikrát zkusit odeslat příkaz při selhání (requotes)
MAX_SLIPPAGE = 50         # Maximální povolený skluz ceny (50 pointů = 5 USD)
MAX_ENTRY_DELAY = 30     # Max zpoždění vstupu od uzavření svíčky (v sekundách)

# Globální paměť bota pro sledování svíček
last_processed_bar_time = None

# ==========================================
# 2. BEZPEČNÉ POMOCNÉ FUNKCE
# ==========================================
def calculate_lot_size(sl_points):
    """Bezpečný výpočet lotu. Selže a nepustí obchod, pokud chybí data."""
    account = mt5.account_info()
    symbol_info = mt5.symbol_info(SYMBOL)
    
    if account is None or symbol_info is None:
        print("KRITICKÁ CHYBA: Nelze načíst účet nebo symbol. Obchod zrušen.")
        return None

    risk_usd = account.equity * RISK_PER_TRADE
    risk_per_1_lot = sl_points * symbol_info.trade_contract_size

    if risk_per_1_lot <= 0: 
        print("CHYBA: Risk na 1 lot je 0. Zkontroluj Stop Loss vzdálenost.")
        return None

    raw_lots = risk_usd / risk_per_1_lot
    step = symbol_info.volume_step
    lots = round(raw_lots / step) * step
    
    # Zarovnání do povolených mezí brokera (min/max lot)
    lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))
    return round(lots, 2)

def execute_with_retries(request, action_name):
    """Odešle příkaz brokerovi a v případě selhání (např. spread gap) to zkusí znovu."""
    for attempt in range(1, MAX_RETRIES + 1):
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f">>> {action_name} ÚSPĚŠNÁ! (Ticket: {result.deal})")
            return True
        else:
            print(f"Pokus {attempt}/{MAX_RETRIES} o {action_name} selhal: {result.comment} (Kód: {result.retcode})")
            time.sleep(1)
    
    print(f"[!!!] FATÁLNÍ CHYBA: {action_name} se nezdařila ani po {MAX_RETRIES} pokusech.")
    return False

def close_all_my_positions(comment="Vystup_po_1_svicce"):
    """Zavře všechny naše pozice s retry smyčkou (ochrana proti nezavření)."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            tick = mt5.symbol_info_tick(SYMBOL)
            type_dict = {mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL, 
                         mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY}
            price_dict = {mt5.POSITION_TYPE_BUY: tick.bid, 
                          mt5.POSITION_TYPE_SELL: tick.ask}
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": MAX_SLIPPAGE,
                "magic": MAGIC_NUMBER,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            execute_with_retries(request, f"Zavření pozice {pos.ticket}")

def open_sell_order(price, sl_points, volume):
    """Otevře SELL objednávku pouze se Stop Lossem (ochrana proti slippage gapům)."""
    sl_price = price + sl_points
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "deviation": MAX_SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "Nasdaq_V3_Algo",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    execute_with_retries(request, f"Otevření SELL {volume} lotů")

# ==========================================
# 3. HLAVNÍ LOGIKA STRATEGIE
# ==========================================
def run_strategy():
    global last_processed_bar_time

    current_tick = mt5.symbol_info_tick(SYMBOL)
    if current_tick is None:
        return
        
    current_broker_time_str = pd.to_datetime(current_tick.time, unit='s').strftime('%H:%M')

    if current_broker_time_str >= HARD_CLOSE_TIME:
        positions = mt5.positions_get(symbol=SYMBOL)
        active_my_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
        if len(active_my_positions) > 0:
            print(f"[{current_broker_time_str}] Dosažen HARD CLOSE. Násilně zavírám pozice před Rolloverem!")
            close_all_my_positions(comment="Hard_Close_2330")
        return # Funkce se okamžitě ukončí, bot nebude hledat další vstupy

    rates_light = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, 2)
    if rates_light is None or len(rates_light) < 2:
        return

    last_closed_bar_light = rates_light[0] # Svíčka, která se právě dokreslila
    
    if last_processed_bar_time == last_closed_bar_light['time']:
        return 

    # === MÁME NOVOU SVÍČKU ===
    last_processed_bar_time = last_closed_bar_light['time']
    closed_time_str = pd.to_datetime(last_closed_bar_light['time'], unit='s').strftime('%H:%M')
    print(f"\n[{current_broker_time_str}] Nový M30 bar. Čas uzavřené svíčky: {closed_time_str}")

    # --- 2. KONTROLA ZPOŽDĚNÍ (Freshness Check) ---
    # 1800s = odečteme 30min trvání svíčky, abychom zjistili, před kolika sekundami přesně skončila
    seconds_since_close = current_tick.time - last_closed_bar_light['time'] - 1800 

    # --- 3. VÝSTUPNÍ LOGIKA (Zavření na konci svíčky jako v backtestu) ---
    positions = mt5.positions_get(symbol=SYMBOL)
    active_my_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
    
    if len(active_my_positions) > 0:
        print("Zavírám předchozí obchod (konec svíčky)...")
        close_all_my_positions(comment="Vystup_Na_Close")
        time.sleep(1) 
        # Znovunačtení pozic po případném uzavření
        positions = mt5.positions_get(symbol=SYMBOL)
        active_my_positions = [p for p in positions if p.magic == MAGIC_NUMBER]

    # --- 4. TĚŽKÁ MATEMATIKA A PANDAS ---
    rates_m30 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M30, 0, 8000)
    df_m30 = pd.DataFrame(rates_m30)
    df_m30['time'] = pd.to_datetime(df_m30['time'], unit='s')
    df_m30['bar_size'] = abs(df_m30['close'] - df_m30['open'])
    df_m30['time_str'] = df_m30['time'].dt.strftime('%H:%M')
    
    last_closed_bar = df_m30.iloc[-2]

    # D1 ATR (Volatilita)
    rates_d1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, ATR_PERIOD + 5)
    df_d1 = pd.DataFrame(rates_d1)
    df_d1['prev_close'] = df_d1['close'].shift(1)
    df_d1['TR'] = df_d1[['high', 'low', 'prev_close']].apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['prev_close']), abs(row['low'] - row['prev_close'])), axis=1
    )
    df_d1['ATR'] = df_d1['TR'].rolling(window=ATR_PERIOD).mean()
    prev_day_atr = df_d1['ATR'].iloc[-2]
    
    if pd.isna(prev_day_atr) or prev_day_atr == 0:
        print("[!] Nelze spočítat ATR pro tento den.")
        return

    # Percentil - vytažení z historie POUZE pro tento konkrétní čas v rámci dne
    history_same_time = df_m30.iloc[:-2] # Odřízneme poslední zavřenou a právě tvořenou svíčku
    history_same_time = history_same_time[history_same_time['time_str'] == closed_time_str]
    
    if len(history_same_time) < LOOKBACK_WINDOW:
        print(f"[!] Málo historických dat ({len(history_same_time)}) pro čas {closed_time_str}.")
        return
        
    threshold = history_same_time['bar_size'].tail(LOOKBACK_WINDOW).quantile(0.95)

    # --- 5. GENEROVÁNÍ SIGNÁLU K OBCHODU ---
    is_time_valid = START_TIME <= closed_time_str <= LAST_SIGNAL_TIME
    is_bearish = last_closed_bar['close'] < last_closed_bar['open']
    is_big_bar = last_closed_bar['bar_size'] > threshold

    print(f"[{current_broker_time_str}] Čas v okně: {is_time_valid} | Bearish svíčka: {is_bearish} | Bar velikost: {last_closed_bar['bar_size']:.2f} (Limit 95.p: {threshold:.2f})")

    if is_time_valid and is_bearish and is_big_bar and len(active_my_positions) == 0:
        if seconds_since_close > MAX_ENTRY_DELAY:
            print(f"[!] Vstup zrušen: Svíčka se uzavřela před {seconds_since_close}s (Limit je {MAX_ENTRY_DELAY}s). Chráníme cenu.")
            return

        entry_price = current_tick.bid 
        sl_points = prev_day_atr * SL_MULTIPLIER
        
        volume = calculate_lot_size(sl_points)
        if volume is not None:
            open_sell_order(entry_price, sl_points, volume)

# ==========================================
# START BOTA
# ==========================================
if not mt5.initialize():
    print("MT5 Error - nelze inicializovat spojení s terminálem. Je zapnutý Auto Trading?")
    quit()

print(f"==================================================")
print(f"*** BOT V3 FINAL (OSTRÝ PROVOZ) ***")
print(f"Účet: {mt5.account_info().login} | Server: {mt5.account_info().server}")
print(f"Spuštěna ochrana a Hard Close systém (Zavíráme ve {HARD_CLOSE_TIME}).")
print(f"==================================================")

try:
    while True:
        run_strategy()
        time.sleep(1)
except KeyboardInterrupt:
    print("\nBot manuálně ukončen uživatelem.")
    mt5.shutdown()