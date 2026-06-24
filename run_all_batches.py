import os
import math
import subprocess
import sys
import pandas as pd


# -------------------------
# SETTINGS
# -------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

RUN_BATCH_SCRIPT = os.path.join(PROJECT_DIR, "run_batch.py")

BATCH_SIZE = 500


# -------------------------
# LOAD TOTAL TICKERS
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
            df = df[
                ~df[symbol_col]
                .astype(str)
                .str.contains("File Creation Time", na=False)
            ]

            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]

            if "ETF" in df.columns:
                df = df[df["ETF"] == "N"]

            symbols = (
                df[symbol_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            if "Security Name" in df.columns:
                names = (
                    df["Security Name"]
                    .astype(str)
                    .str.upper()
                )
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

            print(f"Loaded {mask.sum()} tickers from {url}")

        except Exception as e:
            print(f"Failed loading {url}: {e}")

    return sorted(list(set(tickers)))


# -------------------------
# MAIN
# -------------------------

all_tickers = load_us_stock_universe()

total_tickers = len(all_tickers)

total_batches = math.ceil(total_tickers / BATCH_SIZE)

print()
print("=" * 60)
print(f"Total tickers: {total_tickers}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total batches: {total_batches}")
print("=" * 60)

for batch_number in range(1, total_batches + 1):

    print()
    print("=" * 60)
    print(f"STARTING BATCH {batch_number} / {total_batches}")
    print("=" * 60)

    command = [
        sys.executable,
        RUN_BATCH_SCRIPT,
        str(batch_number)
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True
    )

    if result.returncode != 0:
        print()
        print(f"Batch {batch_number} failed.")
        print(f"Exit code: {result.returncode}")

        answer = input(
            "Continue to next batch anyway? (y/n): "
        ).strip().lower()

        if answer != "y":
            print("Stopping run.")
            sys.exit(result.returncode)

print()
print("=" * 60)
print("ALL BATCHES COMPLETE")
print("=" * 60)