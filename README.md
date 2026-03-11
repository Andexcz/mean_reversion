# Portfolio algoritmických strategií (US indexy)

Zde budu tvořit své portfolio strategií na algoritmický trading. Nechci, aby se z tohoto projektu stalo jen divadlo, ve kterém ukazuji donekonečna over-optimized strategie, aby má equity křivka krásně rostla až do nebe. Raději strategie ponechávám v realistické formě, která je zobchodovatelná i live. Je to reálná ukázka toho, jak stavím algo systémy pro US indexy (hlavně tedy NQ).

Momentálně (11. 3. 2026) zde najdete dva modely. První strategie byla pouze zkouška, abych se naučil trochu tvořit, takže jsem začal jednoduše s mean-reversion strategií a přesně, jak jsem očekával, základní technická analýza prostě není profitabilní v 99 % případech. Podle toho i dopadl backtest s -42.49% ztrátou po X letech. Druhá strategie už byla vytvořená na základě reálnější a funkčnější myšlenky. Strategie se mi natolik zalíbila, že jsem ji momentálně nasadil na forward-test na [Darwinex demo účet (D.414773)](https://www.darwinex.com/account/D.414773).

## 📂 Co tu najdete
* `Short_only.py` - Backtester funkční breakout strategie postavené na volatilitě.
* `mt5ATR.py` - Live bot, který tuto úspěšnou strategii reálně obchoduje přes MT5.
* `Mean_reversion.py` - Mean-reversion fat-tail backtester (nefunkční strategie).
* `Kaggle_data.py` - Jednoduchý skript na stahování historických M30 dat a celkově si tam hraju s Kaggle daty (spíše technický soubor bokem).

---

## 🏆 Výherní strategie live na Darwinexu: Volatility Fading (`Short_only.py` & `mt5ATR.py`)
Základní myšlenka je jednoduchá: nesnažím se hádat směr trhu každý den. Tahle strategie prostě sedí a čeká na extrémní statistické úlety v průběhu US seance. 

**Jak to funguje:**
* **Kontext:** Model bere M30 svíčky a srovnává je jen s historií (120 dní nazpět) ve stejný čas (např. 16:30 porovnává se všemi 16:30 za posledních 120 dní). Tím vyfiltruju běžný šum od skutečných extrémů. Zkoušel jsem vytvářet model pro oba scénáře (long i short) – longový scénář mi nepřišel nijak zajímavý ani stabilní. Zatímco short scénář vykazuje docela pěkné výsledky.
* **Signál pro vstup:** Když svíčka zavře bearish a její velikost (počítám pouze tělo, tedy Open až Close) překročí 95. percentil této specifické historie, bot to nebere tak, že byl pohyb už moc veliký, ale naopak tak, že se chce svézt na pokračujícím prodejním tlaku.
* **Risk a Výstup:** Pevný risk 1.2 % na obchod (s tímhle si chci ještě pohrát, protože za 10 let se 2x vyskytl 12% max drawdown a já bych to chtěl dostat pod 10 %). Stop loss je dynamický a vypočítaný jako `0.24 * ATR(5)`. Tento multiplikátor lambda (těch 0.24) je jediná věc, co jsem se snažil co nejvíce zoptimalizovat (původně algoritmus běžel s lambdou 0.5). A to nejdůležitější – time-based exit. Pozice se drží přesně jednu svíčku (30 minut) a poté se ihned bez emocí zavře.

**Live Bot (`mt5ATR.py`):**
Napsat backtest je jedna věc, nasadit to live na MT5 druhá. Skript obsahuje asynchronní mikro-polling, aby trefil close svíčky na milisekundu přesně. Zároveň řeší klasické nástrahy reálného tradingu – zamítnutí příkazu brokerem, timeouty a má v sobě pojistku (Hard Close, která zavře všechny obchody před koncem NY seance). Díky tomu nezůstanu zaseklý v obchodu přes noc a neplatím zbytečně swapy. Rozhodně také nechcete takový obchod držet přes víkend, obzvlášť v dnešní době plné makroekonomických a geopolitických nejistot.

---

## 💀 `Mean_reversion.py` - Totální selhání
Z historie commitů je asi jasné, jak tohle dopadlo. Chtěl jsem být chytřejší a postavit model na Studentově t-rozdělení (df=5), abych líp podchytil "fat tails" a extrémní odchylky, které jsou ignorovány při použití klasického normálního rozdělení.

**Logika:**
Počítám t-statistiku z 30denního průměru a volatility. Když pravděpodobnost spadne pod 2 % a trh se drží nad SMA 500, znamená to přeprodaný trh a nakupuju. Ošetřeno dynamickým sizingem a 2x ATR stop lossem.

**Proč to chcíplo (a proč to nesmažu):**
Matematicky to dávalo super smysl. V reálu to byl průšvih. Když Nasdaq trefí 2% extrém, málokdy to znamená, že je jen "přeprodaný" a teď se hezky odrazí. Většinou to znamená, že panika právě začala. Bot sice hezky nakupoval dipy, akorát že to dělal přesně vteřinu před tím, než to spadlo o dalších 5 %.

Nechal jsem to tu schválně. Je to perfektní připomínka toho, že čistá statistika bez pochopení tržního kontextu a momenta je k ničemu. Zároveň to alespoň ukazuje, že základní technická analýza sama o sobě stojí za h****. :)

---
**Disclaimer 1:** *Tento kód slouží pouze pro edukační účely a jako ukázka mého přístupu k algotradingu. Rozhodně to není investiční doporučení.*

**Disclaimer 2:** *Abych byl upřímný, tento text v README souboru je z velké části vygenerovaný umělou inteligencí. Doplňoval jsem pouze své myšlenky a detaily, které jsem nechtěl, aby zanikly. Text vygenerovalo AI na základě mých poznámek, promptů a samotného kódu, protože se mi to nechtělo psát úplně od nuly. Logika, matematika a kód samotný jsou ale čistě moje práce (kódy byly AI pouze okomentovány a upraveny, aby je dokázal čistě číst i někdo externí)! :)*