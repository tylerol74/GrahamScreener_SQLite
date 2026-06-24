"""
graham_screener.py - cached version v5 batch-price + fundamentals-cache debug

Graham-style undervalued stock screener with:
- Nasdaq Trader ticker universe
- Manual batch controls
- Fundamentals cache so EPS/book/current ratio/debt are not re-downloaded every run
- Fresh price check each run using yf.download() in chunks
- CSV outputs in the outputs/ folder

How to use:
1. Save this file as graham_screener.py in your project folder.
2. Change BATCH_NUMBER to 1, 2, 3, etc. as needed.
3. Run: py graham_screener.py

Important:
- Price/volume is downloaded first in large yf.download() chunks.
- First run for a ticker will download fundamentals and save them to outputs/fundamentals_cache.csv.
- Later runs reuse cached fundamentals until they are older than CACHE_MAX_AGE_DAYS.
- Price is intentionally refreshed every run because price changes daily.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# SETTINGS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Manual batch controls.
# Change BATCH_NUMBER to 2, 3, 4, etc. to scan the next group.
BATCH_SIZE = 500
BATCH_NUMBER = 1

# Fundamental cache controls.
# Fundamentals do not need daily refreshing. 60 days is a reasonable starting point.
FUNDAMENTALS_CACHE = os.path.join(OUTPUT_DIR, "fundamentals_cache.csv")
CACHE_MAX_AGE_DAYS = 9999
FORCE_REFRESH_FUNDAMENTALS = False

# Output files.
GRAHAM_RESULTS = os.path.join(OUTPUT_DIR, "graham_results.csv")
GRAHAM_OPPORTUNITIES = os.path.join(OUTPUT_DIR, "graham_opportunities.csv")
GRAHAM_BATCH_RESULTS = os.path.join(OUTPUT_DIR, f"graham_batch_{BATCH_NUMBER}_results.csv")
GRAHAM_BATCH_OPPORTUNITIES = os.path.join(OUTPUT_DIR, f"graham_batch_{BATCH_NUMBER}_opportunities.csv")
CACHE_MISSES = os.path.join(OUTPUT_DIR, f"graham_batch_{BATCH_NUMBER}_cache_misses.csv")
CACHE_AUDIT = os.path.join(OUTPUT_DIR, f"graham_batch_{BATCH_NUMBER}_cache_audit.csv")

# Basic filters.
MIN_PRICE = 1.00
MIN_DOLLAR_VOLUME = 0  # Set higher later if you want liquidity filter, e.g. 1_000_000

# Graham-ish thresholds.
MAX_PE = 15
MAX_PB = 1.5
MIN_MARGIN_OF_SAFETY = 30
MIN_CURRENT_RATIO = 1.5
MAX_DEBT_TO_EQUITY = 100

# Delay only when downloading new fundamentals.
# You lowered this before; keep it polite enough to avoid throttling.
DOWNLOAD_DELAY_SECONDS = 0.5
PRICE_DOWNLOAD_CHUNK_SIZE = 100


# =========================
# HELPERS
# =========================

def safe_float(value):
    """Convert a value to float, returning np.nan if it cannot be converted."""
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def load_us_stock_universe():
    """
    Load common-stock-like tickers from Nasdaq Trader files.
    This intentionally filters out many ETFs, warrants, rights, units, notes, and preferreds.
    Later, if you want Graham-style preferred/warrant comparison, build a separate universe for those.
    """
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
                print(f"Could not find symbol column in {url}")
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
                names = pd.Series([""] * len(df), index=df.index)

            mask = (
                symbols.notna()
                & ~symbols.str.contains(r"\$|\.|/", regex=True, na=False)
                & ~symbols.str.endswith(("W", "U", "R"), na=False)
                & ~names.str.contains(
                    "WARRANT|RIGHT|UNIT|ETF|ETN|NOTE|PREFERRED|PREF|DEPOSITARY|REIT",
                    regex=True,
                    na=False,
                )
            )

            clean_symbols = symbols[mask].tolist()
            tickers.extend(clean_symbols)
            print(f"Loaded {len(clean_symbols)} clean tickers from {url}")

        except Exception as e:
            print(f"Failed loading Nasdaq Trader file {url}: {e}")

    return sorted(list(set(tickers)))


def load_fundamentals_cache():
    """Load the fundamentals cache if it exists. Supports several older column names."""
    if not os.path.exists(FUNDAMENTALS_CACHE):
        print("CACHE STATUS: no fundamentals_cache.csv found yet.")
        return pd.DataFrame()

    try:
        cache = pd.read_csv(FUNDAMENTALS_CACHE)

        # Normalize ticker column if an older file used Symbol/ticker instead of Ticker.
        if "Ticker" not in cache.columns:
            for possible in ["ticker", "Symbol", "symbol"]:
                if possible in cache.columns:
                    cache = cache.rename(columns={possible: "Ticker"})
                    break

        if "Ticker" in cache.columns:
            cache["Ticker"] = cache["Ticker"].astype(str).str.upper().str.strip()
        else:
            print("CACHE STATUS: cache file exists, but it has no Ticker/Symbol column.")
            print(f"Cache columns are: {list(cache.columns)}")
            return pd.DataFrame()

        # Normalize common older field names.
        rename_map = {
            "BookValue": "Book Value",
            "bookValue": "Book Value",
            "book_value": "Book Value",
            "CurrentRatio": "Current Ratio",
            "currentRatio": "Current Ratio",
            "DebtToEquity": "Debt To Equity",
            "debtToEquity": "Debt To Equity",
            "Trailing EPS": "EPS",
            "trailingEps": "EPS",
        }
        cache = cache.rename(columns={k: v for k, v in rename_map.items() if k in cache.columns})

        if "Last Fundamental Update" not in cache.columns:
            for possible in ["Last Updated", "Downloaded At", "Last Price Update", "Date", "date"]:
                if possible in cache.columns:
                    cache["Last Fundamental Update"] = cache[possible]
                    break
            else:
                # Existing cache with no date: treat as usable so we don't redownload everything.
                cache["Last Fundamental Update"] = datetime.now()

        cache["Last Fundamental Update"] = pd.to_datetime(cache["Last Fundamental Update"], errors="coerce")
        cache["Last Fundamental Update"] = cache["Last Fundamental Update"].fillna(pd.Timestamp(datetime.now()))

        # Drop duplicate ticker rows and keep the newest one.
        cache = cache.sort_values("Last Fundamental Update").drop_duplicates("Ticker", keep="last")

        return cache

    except Exception as e:
        print(f"CACHE STATUS: could not read fundamentals cache: {e}")
        return pd.DataFrame()

def get_cached_fundamentals(ticker, cache):
    """
    Return cached fundamentals for ticker if present and fresh enough.
    Also returns a readable reason when it misses.
    """
    if FORCE_REFRESH_FUNDAMENTALS:
        return None, "force refresh is turned on"

    if cache.empty:
        return None, "cache file is empty or was not loaded"

    if "Ticker" not in cache.columns:
        return None, "cache has no Ticker column"

    ticker = str(ticker).upper().strip()
    rows = cache[cache["Ticker"] == ticker]

    if rows.empty:
        return None, "ticker not found in cache"

    row = rows.iloc[0]
    last_update = row.get("Last Fundamental Update", pd.NaT)

    if pd.isna(last_update):
        return None, "cache row has no usable date"

    age_days = (datetime.now() - last_update.to_pydatetime()).days

    if age_days > CACHE_MAX_AGE_DAYS:
        return None, f"cache row is stale: {age_days} days old"

    return row.to_dict(), "cache hit"

def upsert_fundamentals_cache(cache, ticker, data):
    """
    Insert or replace one ticker's fundamentals in both the in-memory cache and the CSV file.
    Returns the updated in-memory cache.
    """
    ticker = str(ticker).upper().strip()

    data = dict(data)
    data["Ticker"] = ticker
    data["Last Fundamental Update"] = datetime.now()

    new_row = pd.DataFrame([data])

    if cache.empty:
        updated = new_row
    else:
        if "Ticker" in cache.columns:
            cache = cache[cache["Ticker"] != ticker]
        updated = pd.concat([cache, new_row], ignore_index=True)

    updated.to_csv(FUNDAMENTALS_CACHE, index=False)
    return updated


def download_fundamentals(ticker):
    """
    Download fundamental fields from yfinance.
    This version NEVER hides a failed attempt. It returns a row even if yfinance fails,
    so the ticker can still be written to the cache with a status.
    """
    base = {
        "Short Name": None,
        "Sector": None,
        "Industry": None,
        "EPS": np.nan,
        "Forward EPS": np.nan,
        "Book Value": np.nan,
        "Current Ratio": np.nan,
        "Debt To Equity": np.nan,
        "Total Debt": np.nan,
        "Total Cash": np.nan,
        "Market Cap": np.nan,
        "Profit Margins": np.nan,
        "Return On Equity": np.nan,
        "Revenue Growth": np.nan,
        "Shares Outstanding": np.nan,
        "Download Status": "not_attempted",
        "Download Error": "",
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not isinstance(info, dict) or len(info) == 0:
            base["Download Status"] = "empty_info"
            base["Download Error"] = "yfinance returned empty info"
            return base

        base.update({
            "Short Name": info.get("shortName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "EPS": safe_float(info.get("trailingEps")),
            "Forward EPS": safe_float(info.get("forwardEps")),
            "Book Value": safe_float(info.get("bookValue")),
            "Current Ratio": safe_float(info.get("currentRatio")),
            "Debt To Equity": safe_float(info.get("debtToEquity")),
            "Total Debt": safe_float(info.get("totalDebt")),
            "Total Cash": safe_float(info.get("totalCash")),
            "Market Cap": safe_float(info.get("marketCap")),
            "Profit Margins": safe_float(info.get("profitMargins")),
            "Return On Equity": safe_float(info.get("returnOnEquity")),
            "Revenue Growth": safe_float(info.get("revenueGrowth")),
            "Shares Outstanding": safe_float(info.get("sharesOutstanding")),
            "Download Status": "ok",
            "Download Error": "",
        })
        return base

    except Exception as e:
        base["Download Status"] = "error"
        base["Download Error"] = str(e)[:250]
        return base

def get_batch_latest_prices(tickers, chunk_size=PRICE_DOWNLOAD_CHUNK_SIZE):
    """
    Refresh latest price and volume using yf.download() in chunks.

    This is the fast part: prices/volume can be downloaded for many tickers at once.
    Fundamentals cannot reliably be batch-downloaded through yfinance .info, so those are
    cached separately and only downloaded when missing/stale.
    """
    price_rows = []

    tickers = [str(t).upper().strip() for t in tickers]

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        print(f"Batch downloading prices {start + 1}-{start + len(chunk)} of {len(tickers)}...")

        try:
            data = yf.download(
                tickers=" ".join(chunk),
                period="10d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            if data is None or data.empty:
                for ticker in chunk:
                    price_rows.append({
                        "Ticker": ticker,
                        "Price": np.nan,
                        "Latest Volume": np.nan,
                        "Dollar Volume": np.nan,
                    })
                continue

            # Multiple tickers usually return columns like (Ticker, Field).
            # One ticker can return simple columns like Close, Volume.
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker not in data.columns.get_level_values(0):
                            close = np.nan
                            volume = np.nan
                        else:
                            tdf = data[ticker].dropna(how="all")
                            close_col = "Close" if "Close" in tdf.columns else "Adj Close"
                            close = safe_float(tdf[close_col].dropna().iloc[-1]) if close_col in tdf.columns and not tdf[close_col].dropna().empty else np.nan
                            volume = safe_float(tdf["Volume"].dropna().iloc[-1]) if "Volume" in tdf.columns and not tdf["Volume"].dropna().empty else np.nan
                    else:
                        tdf = data.dropna(how="all")
                        close_col = "Close" if "Close" in tdf.columns else "Adj Close"
                        close = safe_float(tdf[close_col].dropna().iloc[-1]) if close_col in tdf.columns and not tdf[close_col].dropna().empty else np.nan
                        volume = safe_float(tdf["Volume"].dropna().iloc[-1]) if "Volume" in tdf.columns and not tdf["Volume"].dropna().empty else np.nan

                    dollar_volume = close * volume if pd.notna(close) and pd.notna(volume) else np.nan

                    price_rows.append({
                        "Ticker": ticker,
                        "Price": close,
                        "Latest Volume": volume,
                        "Dollar Volume": dollar_volume,
                    })

                except Exception:
                    price_rows.append({
                        "Ticker": ticker,
                        "Price": np.nan,
                        "Latest Volume": np.nan,
                        "Dollar Volume": np.nan,
                    })

        except Exception as e:
            print(f"Price batch download failed for chunk starting at {start + 1}: {e}")
            for ticker in chunk:
                price_rows.append({
                    "Ticker": ticker,
                    "Price": np.nan,
                    "Latest Volume": np.nan,
                    "Dollar Volume": np.nan,
                })

    prices_df = pd.DataFrame(price_rows)

    if prices_df.empty:
        return {}

    prices_df = prices_df.drop_duplicates("Ticker", keep="last")
    return prices_df.set_index("Ticker").to_dict(orient="index")


def graham_number(eps, book_value):
    """
    Benjamin Graham-style number:
    sqrt(22.5 * EPS * Book Value)
    Only valid when EPS and book value are positive.
    """
    eps = safe_float(eps)
    book_value = safe_float(book_value)

    if pd.isna(eps) or pd.isna(book_value):
        return np.nan

    if eps <= 0 or book_value <= 0:
        return np.nan

    return np.sqrt(22.5 * eps * book_value)


def calculate_score(row):
    """
    Simple Graham-style score.
    Higher score = more interesting.
    This is not investment advice; it is only a ranking tool for research.
    """
    score = 0

    margin = row.get("Margin Of Safety %", np.nan)
    pe = row.get("P/E", np.nan)
    pb = row.get("P/B", np.nan)
    current_ratio = row.get("Current Ratio", np.nan)
    debt_to_equity = row.get("Debt To Equity", np.nan)

    if pd.notna(margin):
        if margin >= 50:
            score += 3
        elif margin >= 30:
            score += 2
        elif margin > 0:
            score += 1

    if pd.notna(pe):
        if pe <= 10:
            score += 2
        elif pe <= 15:
            score += 1

    if pd.notna(pb):
        if pb <= 1:
            score += 2
        elif pb <= 1.5:
            score += 1

    if pd.notna(current_ratio):
        if current_ratio >= 2:
            score += 1
        elif current_ratio >= 1.5:
            score += 0.5

    if pd.notna(debt_to_equity):
        if debt_to_equity <= 50:
            score += 1
        elif debt_to_equity <= 100:
            score += 0.5

    return score


def is_graham_opportunity(row):
    """Conservative first-pass flag for potential Graham-style opportunities."""
    price = row.get("Price", np.nan)
    eps = row.get("EPS", np.nan)
    book_value = row.get("Book Value", np.nan)
    margin = row.get("Margin Of Safety %", np.nan)
    pe = row.get("P/E", np.nan)
    pb = row.get("P/B", np.nan)
    current_ratio = row.get("Current Ratio", np.nan)
    debt_to_equity = row.get("Debt To Equity", np.nan)
    dollar_volume = row.get("Dollar Volume", np.nan)

    if pd.isna(price) or price < MIN_PRICE:
        return False

    if pd.isna(eps) or eps <= 0:
        return False

    if pd.isna(book_value) or book_value <= 0:
        return False

    if pd.isna(margin) or margin < MIN_MARGIN_OF_SAFETY:
        return False

    if pd.isna(pe) or pe > MAX_PE:
        return False

    if pd.isna(pb) or pb > MAX_PB:
        return False

    # If these fields are missing, we do not automatically reject the stock.
    # Some yfinance records are incomplete. Instead, the score will penalize missing data.
    if pd.notna(current_ratio) and current_ratio < MIN_CURRENT_RATIO:
        return False

    if pd.notna(debt_to_equity) and debt_to_equity > MAX_DEBT_TO_EQUITY:
        return False

    if MIN_DOLLAR_VOLUME > 0 and (pd.isna(dollar_volume) or dollar_volume < MIN_DOLLAR_VOLUME):
        return False

    return True


# =========================
# MAIN PROGRAM
# =========================

def main():
    print("RUNNING GRAHAM SCREENER CACHED VERSION v5.2 BATCH PRICE + FUNDAMENTALS CACHE")
    print("If you do not see this exact line, you are running the wrong file.")
    print()
    print("Loading Graham screener universe...")
    tickers = load_us_stock_universe()
    print(f"Loaded {len(tickers)} total tickers.")

    start = (BATCH_NUMBER - 1) * BATCH_SIZE
    end = start + BATCH_SIZE
    batch_tickers = tickers[start:end]

    print()
    print(f"Scanning Graham batch {BATCH_NUMBER}")
    print(f"Tickers {start + 1} through {min(end, len(tickers))}")
    print(f"Scanning {len(batch_tickers)} tickers.")
    print()

    fundamentals_cache = load_fundamentals_cache()
    print(f"Current working folder: {os.getcwd()}")
    print(f"Script folder: {BASE_DIR}")
    print(f"Fundamentals cache file: {FUNDAMENTALS_CACHE}")
    print(f"Loaded {len(fundamentals_cache)} cached fundamental records.")
    print()

    print("Downloading current prices/volume in large batches first...")
    price_lookup = get_batch_latest_prices(batch_tickers)
    print(f"Loaded current price/volume rows for {len(price_lookup)} tickers.")
    print()

    results = []
    cache_misses = []

    for i, ticker in enumerate(batch_tickers, start=1):
        try:
            print(f"[{i}/{len(batch_tickers)}] Processing {ticker}")

            cached, cache_reason = get_cached_fundamentals(ticker, fundamentals_cache)

            if cached is not None:
                print(f"CACHE HIT fundamentals: {ticker}")
                fundamentals = cached
            else:
                print(f"CACHE MISS fundamentals: {ticker} — {cache_reason}")
                print(f"Downloading fundamentals: {ticker}")
                time.sleep(DOWNLOAD_DELAY_SECONDS)
                fundamentals = download_fundamentals(ticker)

                # IMPORTANT: save immediately, even if yfinance returned missing fields or an error.
                # This prevents the same ticker from being re-downloaded forever.
                fundamentals_cache = upsert_fundamentals_cache(fundamentals_cache, ticker, fundamentals)
                cache_misses.append({
                    "Ticker": ticker,
                    "Downloaded At": datetime.now(),
                    "Download Status": fundamentals.get("Download Status"),
                    "Download Error": fundamentals.get("Download Error"),
                })
                print(f"SAVED CACHE ROW: {ticker} — {fundamentals.get('Download Status')}")

            price_data = price_lookup.get(str(ticker).upper().strip(), {})
            price = safe_float(price_data.get("Price"))
            latest_volume = safe_float(price_data.get("Latest Volume"))
            dollar_volume = safe_float(price_data.get("Dollar Volume"))

            eps = safe_float(fundamentals.get("EPS"))
            book_value = safe_float(fundamentals.get("Book Value"))
            current_ratio = safe_float(fundamentals.get("Current Ratio"))
            debt_to_equity = safe_float(fundamentals.get("Debt To Equity"))
            market_cap = safe_float(fundamentals.get("Market Cap"))

            if pd.isna(price) or price <= 0:
                print(f"Skipping {ticker}: no usable price")
                continue

            if price < MIN_PRICE:
                print(f"Skipping {ticker}: price below minimum")
                continue

            g_number = graham_number(eps, book_value)

            pe = price / eps if pd.notna(eps) and eps > 0 else np.nan
            pb = price / book_value if pd.notna(book_value) and book_value > 0 else np.nan

            margin_of_safety = (
                ((g_number - price) / price) * 100
                if pd.notna(g_number) and price > 0
                else np.nan
            )

            row = {
                "Ticker": ticker,
                "Short Name": fundamentals.get("Short Name"),
                "Sector": fundamentals.get("Sector"),
                "Industry": fundamentals.get("Industry"),
                "Price": price,
                "Latest Volume": latest_volume,
                "Dollar Volume": dollar_volume,
                "Market Cap": market_cap,
                "EPS": eps,
                "Forward EPS": safe_float(fundamentals.get("Forward EPS")),
                "Book Value": book_value,
                "Graham Number": g_number,
                "Margin Of Safety %": margin_of_safety,
                "P/E": pe,
                "P/B": pb,
                "Current Ratio": current_ratio,
                "Debt To Equity": debt_to_equity,
                "Total Debt": safe_float(fundamentals.get("Total Debt")),
                "Total Cash": safe_float(fundamentals.get("Total Cash")),
                "Profit Margins": safe_float(fundamentals.get("Profit Margins")),
                "Return On Equity": safe_float(fundamentals.get("Return On Equity")),
                "Revenue Growth": safe_float(fundamentals.get("Revenue Growth")),
                "Shares Outstanding": safe_float(fundamentals.get("Shares Outstanding")),
                "Download Status": fundamentals.get("Download Status"),
                "Download Error": fundamentals.get("Download Error"),
                "Last Fundamental Update": fundamentals.get("Last Fundamental Update"),
                "Last Price Update": datetime.now(),
            }

            row["Graham Score"] = calculate_score(row)
            row["Graham Opportunity"] = is_graham_opportunity(row)

            results.append(row)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print()
        print("No Graham results found for this batch.")
        return

    results_df = results_df.sort_values(
        by=["Graham Opportunity", "Graham Score", "Margin Of Safety %"],
        ascending=[False, False, False],
    )

    opportunities_df = results_df[results_df["Graham Opportunity"] == True].copy()

    # Save batch-specific outputs.
    results_df.to_csv(GRAHAM_BATCH_RESULTS, index=False)
    opportunities_df.to_csv(GRAHAM_BATCH_OPPORTUNITIES, index=False)

    # Save general/latest outputs too, so you always know which files to look at.
    results_df.to_csv(GRAHAM_RESULTS, index=False)
    opportunities_df.to_csv(GRAHAM_OPPORTUNITIES, index=False)

    if cache_misses:
        pd.DataFrame(cache_misses).to_csv(CACHE_MISSES, index=False)

    # Save a copy of the cache after this run so you can verify which tickers are stored.
    if not fundamentals_cache.empty:
        fundamentals_cache.to_csv(CACHE_AUDIT, index=False)

    print()
    print("Done.")
    print(f"Total results this batch: {len(results_df)}")
    print(f"Graham opportunities this batch: {len(opportunities_df)}")
    print(f"New fundamental downloads this batch: {len(cache_misses)}")
    print()
    print(f"Saved batch results to: {GRAHAM_BATCH_RESULTS}")
    print(f"Saved batch opportunities to: {GRAHAM_BATCH_OPPORTUNITIES}")
    print(f"Saved latest results to: {GRAHAM_RESULTS}")
    print(f"Saved latest opportunities to: {GRAHAM_OPPORTUNITIES}")
    print(f"Saved/updated fundamentals cache at: {FUNDAMENTALS_CACHE}")
    print(f"Saved cache audit copy to: {CACHE_AUDIT}")
    print()

    display_columns = [
        "Ticker",
        "Price",
        "EPS",
        "Book Value",
        "Graham Number",
        "Margin Of Safety %",
        "P/E",
        "P/B",
        "Current Ratio",
        "Debt To Equity",
        "Graham Score",
        "Graham Opportunity",
    ]

    existing_display_columns = [col for col in display_columns if col in results_df.columns]
    print(results_df[existing_display_columns].head(25))


if __name__ == "__main__":
    main()
