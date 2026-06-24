import os
import sys
import time
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from database import DB_FILE, create_tables


# -------------------------
# SETTINGS
# -------------------------

BATCH_SIZE = 500
BATCH_NUMBER = int(sys.argv[1]) if len(sys.argv) > 1 else 1

REQUEST_DELAY_SECONDS = 0.23

MIN_PRICE = 1.00
MIN_DOLLAR_VOLUME = 500_000


# -------------------------
# FOLDERS / FILES
# -------------------------

OUTPUT_DIR = "outputs"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
BATCHES_DIR = os.path.join(OUTPUT_DIR, "batches")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BATCHES_DIR, exist_ok=True)

TECHNICAL_BATCH_FILE = os.path.join(
    BATCHES_DIR,
    f"technical_batch_{BATCH_NUMBER}.csv"
)

# -------------------------
# TICKER LOADER
# -------------------------

def load_us_stock_universe():
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    tickers = []

    for url in urls:
        try:
            df = pd.read_csv(url, sep="|")

            if "Symbol" in df.columns:
                symbol_col = "Symbol"
            elif "ACT Symbol" in df.columns:
                symbol_col = "ACT Symbol"
            else:
                continue

            df = df[df[symbol_col].notna()]
            df = df[~df[symbol_col].astype(str).str.contains("File Creation Time", na=False)]

            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]

            if "ETF" in df.columns:
                df = df[df["ETF"] == "N"]

            symbols = df[symbol_col].astype(str).str.strip().str.upper()

            if "Security Name" in df.columns:
                names = df["Security Name"].astype(str).str.upper()
            else:
                names = pd.Series([""] * len(df))

            mask = (
                symbols.notna()
                & ~symbols.str.contains(r"\$|\.|/", regex=True)
                & ~symbols.str.endswith(("W", "U", "R"))
                & ~names.str.contains(
                    "WARRANT|RIGHT|UNIT|ETF|ETN|NOTE|PREFERRED|PREF",
                    regex=True,
                    na=False
                )
            )

            tickers.extend(symbols[mask].tolist())

            print(f"Loaded {mask.sum()} clean tickers from {url}")

        except Exception as e:
            print(f"Failed loading Nasdaq Trader file {url}: {e}")

    return sorted(list(set(tickers)))


# -------------------------
# SCORING
# -------------------------

def calculate_technical_panic_score(
    price_change_5d,
    price_change_10d,
    volume_spike,
    dollar_volume
):
    score = 0

    # Panic selling requires price weakness.
    # This score should reward downside movement, not upward momentum.
    if pd.notna(price_change_5d):
        if price_change_5d <= -25:
            score += 5
        elif price_change_5d <= -20:
            score += 4
        elif price_change_5d <= -15:
            score += 3
        elif price_change_5d <= -10:
            score += 2
        elif price_change_5d <= -5:
            score += 1

    if pd.notna(price_change_10d):
        if price_change_10d <= -30:
            score += 4
        elif price_change_10d <= -20:
            score += 3
        elif price_change_10d <= -10:
            score += 2

    # Volume matters, but it should not create panic by itself.
    if pd.notna(volume_spike):
        if volume_spike >= 5:
            score += 5
        elif volume_spike >= 3:
            score += 4
        elif volume_spike >= 2:
            score += 3
        elif volume_spike >= 1.5:
            score += 1

    # Liquidity bonus
    if pd.notna(dollar_volume):
        if dollar_volume >= 50_000_000:
            score += 3
        elif dollar_volume >= 10_000_000:
            score += 2
        elif dollar_volume >= 2_000_000:
            score += 1

    return score

## Batch Downloader

def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


TECHNICAL_SQLITE_COLUMNS = [
    "ticker",
    "price",
    "price_change_5d",
    "price_change_10d",
    "latest_volume",
    "avg_volume_20d",
    "volume_spike",
    "dollar_volume",
    "distance_from_20d_avg",
    "technical_panic_score",
    "technical_panic_flag",
    "volume_spike_flag",
    "oversold_flag",
    "momentum_spike_flag",
    "batch_number",
    "last_updated",
]


TECHNICAL_SQLITE_COLUMN_MAP = {
    "Ticker": "ticker",
    "Price": "price",
    "5D Price Change %": "price_change_5d",
    "10D Price Change %": "price_change_10d",
    "Latest Volume": "latest_volume",
    "Avg Volume 20D": "avg_volume_20d",
    "Volume Spike": "volume_spike",
    "Dollar Volume": "dollar_volume",
    "Distance From 20D Avg %": "distance_from_20d_avg",
    "Technical Panic Score": "technical_panic_score",
    "Technical Panic Flag": "technical_panic_flag",
    "Volume Spike Flag": "volume_spike_flag",
    "Oversold Flag": "oversold_flag",
    "Momentum Spike Flag": "momentum_spike_flag",
    "Batch Number": "batch_number",
    "Last Updated": "last_updated",
}


def prepare_technical_sqlite_df(results_df):
    sqlite_df = results_df.rename(columns=TECHNICAL_SQLITE_COLUMN_MAP).copy()

    missing_columns = [
        column for column in TECHNICAL_SQLITE_COLUMNS
        if column not in sqlite_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing columns for technical_results SQLite write: "
            + ", ".join(missing_columns)
        )

    sqlite_df = sqlite_df[TECHNICAL_SQLITE_COLUMNS]

    bool_columns = [
        "technical_panic_flag",
        "volume_spike_flag",
        "oversold_flag",
        "momentum_spike_flag",
    ]
    for column in bool_columns:
        sqlite_df[column] = sqlite_df[column].fillna(False).astype(int)

    sqlite_df["last_updated"] = pd.to_datetime(
        sqlite_df["last_updated"],
        errors="coerce"
    ).astype(str)

    sqlite_df = sqlite_df.dropna(subset=["ticker", "batch_number"])

    return sqlite_df.drop_duplicates(
        subset=["ticker", "batch_number"],
        keep="last"
    )


def save_technical_results_to_sqlite(results_df):
    if results_df.empty:
        print("SQLite write skipped: no technical results to save.")
        return

    sqlite_df = prepare_technical_sqlite_df(results_df)

    if sqlite_df.empty:
        print("SQLite write skipped: no valid technical rows to save.")
        return

    try:
        create_tables()

        with sqlite3.connect(DB_FILE) as conn:
            tickers = sqlite_df["ticker"].dropna().unique().tolist()
            if not tickers:
                print("SQLite write skipped: no tickers available to save.")
                return

            placeholders = ",".join(["?"] * len(tickers))
            conn.execute(
                f"""
                DELETE FROM technical_results
                WHERE batch_number = ?
                AND ticker IN ({placeholders})
                """,
                [BATCH_NUMBER] + tickers
            )

            sqlite_df.to_sql(
                "technical_results",
                conn,
                if_exists="append",
                index=False
            )

        print(f"Saved {len(sqlite_df)} technical rows to SQLite: {DB_FILE}")

    except sqlite3.Error as e:
        print(f"SQLite write failed for technical_results: {e}")
    except ValueError as e:
        print(f"SQLite data preparation failed for technical_results: {e}")
# -------------------------
# MAIN
# -------------------------

print("Loading technical scanner universe...")
all_tickers = load_us_stock_universe()
print(f"Loaded {len(all_tickers)} total tickers.")

start = (BATCH_NUMBER - 1) * BATCH_SIZE
end = start + BATCH_SIZE

TICKERS = all_tickers[start:end]

print(f"Scanning technical batch {BATCH_NUMBER}")
print(f"Tickers {start + 1} through {end}")
print(f"Scanning {len(TICKERS)} tickers.")

print("Downloading price/volume data in batch...")

data = print("Downloading price/volume data in smaller batches...")

PRICE_BATCH_SIZE = 100
results = []

for price_batch in chunk_list(TICKERS, PRICE_BATCH_SIZE):
    print(f"Downloading {len(price_batch)} tickers...")

    try:
        data = yf.download(
            tickers=price_batch,
            period="60d",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False
        )

        # Then your existing `for ticker in price_batch:` processing loop goes here,
        # using `data[ticker]` instead of per-ticker history calls.

    except Exception as e:
        print(f"Batch download error: {e}")

results = []

for ticker in TICKERS:
    try:
        if ticker not in data.columns.get_level_values(0):
            continue

        hist = data[ticker].dropna()

        if hist.empty or len(hist) < 20:
            continue

        latest_close = hist["Close"].iloc[-1]
        latest_volume = hist["Volume"].iloc[-1]

        if latest_close <= MIN_PRICE:
            continue

        close_5d_ago = hist["Close"].iloc[-6] if len(hist) >= 6 else np.nan
        close_10d_ago = hist["Close"].iloc[-11] if len(hist) >= 11 else np.nan

        avg_volume_20d = hist["Volume"].tail(20).mean()
        avg_close_20d = hist["Close"].tail(20).mean()

        dollar_volume = latest_close * latest_volume

        if dollar_volume < MIN_DOLLAR_VOLUME:
            continue

        price_change_5d = (
            ((latest_close - close_5d_ago) / close_5d_ago) * 100
            if pd.notna(close_5d_ago) and close_5d_ago > 0
            else np.nan
        )

        price_change_10d = (
            ((latest_close - close_10d_ago) / close_10d_ago) * 100
            if pd.notna(close_10d_ago) and close_10d_ago > 0
            else np.nan
        )

        volume_spike = (
            latest_volume / avg_volume_20d
            if avg_volume_20d > 0
            else np.nan
        )

        distance_from_20d_avg = (
            ((latest_close - avg_close_20d) / avg_close_20d) * 100
            if avg_close_20d > 0
            else np.nan
        )

        technical_panic_score = calculate_technical_panic_score(
            price_change_5d,
            price_change_10d,
            volume_spike,
            dollar_volume
        )

        technical_panic_flag = (
            technical_panic_score >= 7
            and pd.notna(price_change_5d)
            and price_change_5d <= -10
        )

        volume_spike_flag = (
            volume_spike >= 2
            if pd.notna(volume_spike)
            else False
        )

        oversold_flag = (
            price_change_5d <= -10
            if pd.notna(price_change_5d)
            else False
        )

        momentum_spike_flag = (
            price_change_5d >= 20
            and volume_spike >= 2
            if pd.notna(price_change_5d) and pd.notna(volume_spike)
            else False
        )

        results.append({
            "Ticker": ticker,
            "Price": latest_close,
            "5D Price Change %": price_change_5d,
            "10D Price Change %": price_change_10d,
            "Latest Volume": latest_volume,
            "Avg Volume 20D": avg_volume_20d,
            "Volume Spike": volume_spike,
            "Dollar Volume": dollar_volume,
            "Distance From 20D Avg %": distance_from_20d_avg,
            "Technical Panic Score": technical_panic_score,
            "Technical Panic Flag": technical_panic_flag,
            "Volume Spike Flag": volume_spike_flag,
            "Oversold Flag": oversold_flag,
            "Momentum Spike Flag": momentum_spike_flag,
            "Batch Number": BATCH_NUMBER,
            "Last Updated": datetime.now()
        })

        print(f"Processed technicals: {ticker}")

    except Exception as e:
        print(f"Technical error {ticker}: {e}")


results_df = pd.DataFrame(results)

if results_df.empty:
    print("No technical results found.")
else:
    results_df = results_df.sort_values(
        by=[
            "Technical Panic Flag",
            "Technical Panic Score",
            "Volume Spike",
            "5D Price Change %"
        ],
        ascending=[False, False, False, True]
    )

results_df.to_csv(TECHNICAL_BATCH_FILE, index=False)
print("Saving technical results to SQLite...")
save_technical_results_to_sqlite(results_df)
print()
print("Done.")
print(f"Saved technical batch file to: {TECHNICAL_BATCH_FILE}")
print("Total technical results:", len(results_df))
print("Technical panic flags:", len(results_df[results_df["Technical Panic Flag"] == True]))
print("Volume spikes:", len(results_df[results_df["Volume Spike Flag"] == True]))
print("Oversold candidates:", len(results_df[results_df["Oversold Flag"] == True]))
print("Momentum spikes:", len(results_df[results_df["Momentum Spike Flag"] == True]))

print()
print(results_df[[
        "Ticker",
        "Price",
        "5D Price Change %",
        "Volume Spike",
        "Dollar Volume",
        "Technical Panic Score",
        "Technical Panic Flag",
        "Momentum Spike Flag"
    ]].head(25))
