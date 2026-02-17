import kagglehub
import os

#This whole file is unimportant, Im just looking for data in kaggle
path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")

print(f"\nScanning the file: {path}")
print("-" * 50)

files = [f for f in os.listdir(path) if f.endswith('.csv')]

for f in files:
    size_mb = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
    print(f"Soubor: {f}  --- Velikost: {size_mb:.2f} MB")

    import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Načtení dat
path = kagglehub.dataset_download("novandraanugrah/nasdaq-100-nas100-historical-price-data")
file_path = os.path.join(path, "30m_data.csv")

print(f"Analyzuji soubor: {file_path}")
df = pd.read_csv(file_path, sep='\t')
df.columns = df.columns.str.strip()
df['DateTime'] = pd.to_datetime(df['DateTime'])

# 2. Vytáhneme si pouze čas (hodina:minuta)
df['Time_Str'] = df['DateTime'].dt.strftime('%H:%M')

# 3. Spočítáme průměrné TickVolume pro každý časový slot
# (Používáme TickVolume, protože u CFD/Forex dat často reálné Volume chybí nebo je nula)
volume_profile = df.groupby('Time_Str')['TickVolume'].mean()

# 4. Najdeme čas s největší aktivitou
peak_time = volume_profile.idxmax()
peak_volume = volume_profile.max()

print("\n=== VÝSLEDEK ANALÝZY ČASU ===")
print(f"Čas s nejvyšší aktivitou (Peak Volume): {peak_time}")
print("-" * 40)

# Interpretace
if peak_time == "09:30":
    print("ZÁVĚR: Data jsou v NEW YORK TIME (EST/EDT).")
    print("-> Nemusíš nic posouvat, filtry 09:30-16:00 budou fungovat přesně.")
    
elif peak_time in ["13:30", "14:30"]: # 13:30 (letní), 14:30 (zimní)
    print("ZÁVĚR: Data jsou pravděpodobně v GMT/UTC.")
    print("-> Musíš posunout časy v kódu. Místo 09:30 hledej 13:30/14:30.")
    
elif peak_time in ["16:30"]:
    print("ZÁVĚR: Data jsou v BROKER TIME (pravděpodobně GMT+2/GMT+3).")
    print("-> Musíš posunout časy. NY Open (09:30) zde odpovídá 16:30.")
    
else:
    print(f"ZÁVĚR: Neznámé pásmo. Peak je v {peak_time}.")
    print("Porovnej to s 09:30 NY času a spočítej posun.")

# 5. Vykreslení grafu aktivity během dne
plt.figure(figsize=(12, 6))
volume_profile.plot(kind='bar', color='orange')
plt.title('Průměrná aktivita (Volume) během dne')
plt.xlabel('Čas svíčky')
plt.ylabel('Průměrné Volume')
plt.xticks(rotation=90)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()