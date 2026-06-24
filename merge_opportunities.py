import glob
import os
import re
import sqlite3
from datetime import date, datetime

import pandas as pd

from database import DB_FILE, create_tables


# -------------------------
# FOLDERS
# -------------------------

OUTPUT_DIR = "outputs"
BATCHES_DIR = os.path.join(OUTPUT_DIR, "batches")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(BATCHES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# -------------------------
# OUTPUT REPORTS
# -------------------------

GRAHAM_MASTER_FILE = os.path.join(REPORTS_DIR, "graham_results_master.csv")
TECHNICAL_MASTER_FILE = os.path.join(REPORTS_DIR, "technical_results_master.csv")
COMBINED_FILE = os.path.join(REPORTS_DIR, "combined_opportunities.csv")
GRAHAM_PANIC_FILE = os.path.join(REPORTS_DIR, "graham_panic_opportunities.csv")
WATCHLIST_FILE = os.path.join(REPORTS_DIR, "research_watchlist.csv")
TOP_WATCHLIST_FILE = os.path.join(REPORTS_DIR, "top_research_watchlist.csv")
DAILY_SUMMARY_FILE = os.path.join(REPORTS_DIR, "daily_summary.txt")
IGNORE_LIST_FILE = os.path.join(OUTPUT_DIR, "ignore_list.csv")

TOP_WATCHLIST_LIMIT = 50
MAX_PER_SECTOR = 10
MAX_PER_INDUSTRY = 5
MIN_DOLLAR_VOLUME = 500_000
LOW_DOLLAR_VOLUME = 1_000_000
PENNY_STOCK_PRICE = 1.00
LOW_PRICE_PENALTY = 2.00

BAD_TICKER_PATTERN = re.compile(r"(?:\.|/|\$|-|WS$|WT$|W$|U$|R$|P$|PR$)", re.IGNORECASE)
BAD_SECURITY_PATTERN = re.compile(
    r"WARRANT|RIGHT|UNIT|PREFERRED|PREF|DEPOSITARY|SPAC|ACQUISITION CORP|"
    r"ACQUISITION COMPANY|SHELL COMPANIES",
    re.IGNORECASE,
)


# -------------------------
# HELPERS
# -------------------------

def load_batch_files(pattern, label):
    files = glob.glob(os.path.join(BATCHES_DIR, pattern))

    if not files:
        print(f"No {label} batch files found.")
        return pd.DataFrame()

    frames = []

    for file in files:
        try:
            df = pd.read_csv(file)
            df["Source Batch File"] = os.path.basename(file)
            frames.append(df)
            print(f"Loaded {label}: {file}")
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if "Ticker" in combined.columns:
        combined = combined.drop_duplicates(subset=["Ticker"], keep="last")

    return combined


def ensure_ignore_list():
    # Users can add ticker,reason,ignored_at rows here. Any listed ticker is
    # excluded from final research reports, while broad legacy CSVs still run.
    if not os.path.exists(IGNORE_LIST_FILE):
        pd.DataFrame(columns=["ticker", "reason", "ignored_at"]).to_csv(
            IGNORE_LIST_FILE,
            index=False,
        )


def load_ignored_tickers():
    ensure_ignore_list()

    try:
        ignore_df = pd.read_csv(IGNORE_LIST_FILE)
    except Exception as e:
        print(f"Could not read ignore list {IGNORE_LIST_FILE}: {e}")
        return set()

    if "ticker" not in ignore_df.columns:
        return set()

    return set(ignore_df["ticker"].dropna().astype(str).str.upper().str.strip())


def to_bool(series):
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def clean_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def clip_score(series, lower=0, upper=100):
    return series.fillna(0).clip(lower=lower, upper=upper)


def assign_research_tier(row):
    # Tier assignment is intentionally simple and readable:
    # Tier 1: Graham value pass plus confirmed technical panic.
    # Tier 2: Graham value pass without panic.
    # Tier 3: Technical panic or oversold setup without a Graham pass.
    # Tier 4: Volume or momentum signal only.
    graham_pass = bool(row.get("Graham Undervalued")) or bool(row.get("Deep Graham Discount"))
    panic = bool(row.get("Technical Panic Flag")) or row.get("Technical Panic Score", 0) >= 7
    oversold = bool(row.get("Oversold Flag"))
    volume = bool(row.get("Volume Spike Flag")) or row.get("Volume Spike", 0) >= 2
    momentum = bool(row.get("Momentum Spike Flag"))

    if graham_pass and panic:
        return "Tier 1"
    if graham_pass:
        return "Tier 2"
    if panic or oversold:
        return "Tier 3"
    if volume or momentum:
        return "Tier 4"
    return ""


def assign_reason(row):
    graham_pass = bool(row.get("Graham Undervalued")) or bool(row.get("Deep Graham Discount"))
    panic = bool(row.get("Technical Panic Flag")) or row.get("Technical Panic Score", 0) >= 7
    oversold = bool(row.get("Oversold Flag"))
    volume = bool(row.get("Volume Spike Flag")) or row.get("Volume Spike", 0) >= 2
    momentum = bool(row.get("Momentum Spike Flag"))
    margin = row.get("Discount %", 0)

    if graham_pass and panic:
        return "Cheap by Graham metrics + panic selloff"
    if graham_pass and margin >= 40:
        return "High Graham margin of safety"
    if graham_pass:
        return "Graham value pass"
    if oversold and volume:
        return "Oversold with volume spike"
    if panic:
        return "Technical panic selloff"
    if oversold:
        return "Oversold technical setup"
    if volume or momentum:
        return "Unusual volume/momentum only"
    return "Research candidate"


def normalize_company_name(name):
    text = str(name or "").upper()
    text = re.sub(r"\b(CLASS|CL|COM|COMMON|ORDINARY|SHARES|STOCK|A|B|C)\b", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_scores(merged):
    # Scoring is component based so the final rank is explainable:
    # value_score rewards the existing Graham score and positive margin of safety.
    # technical_score rewards panic, oversold, selloff, volume, and momentum signals.
    # liquidity_score rewards higher dollar volume and penalizes very thin trading.
    # data_quality_score rewards usable price, EPS, book value, sector, and industry.
    # combined_opportunity_score blends those components and adds a small tier bonus.
    margin = clip_score(merged["Discount %"])
    existing_value = clip_score(merged["Value Score"])
    merged["value_score"] = (existing_value * 0.60 + margin * 0.40).round(2)

    panic_score = clip_score(merged["Technical Panic Score"] * 6.0)
    selloff_bonus = clip_score(merged["5D Price Change %"].abs(), upper=25) * 0.40
    merged["technical_score"] = (
        panic_score
        + to_bool(merged["Oversold Flag"]).astype(int) * 15
        + to_bool(merged["Volume Spike Flag"]).astype(int) * 10
        + to_bool(merged["Momentum Spike Flag"]).astype(int) * 5
        + selloff_bonus
    ).clip(0, 100).round(2)

    dollar_volume = merged["Dollar Volume"].fillna(0)
    merged["liquidity_score"] = 0
    merged.loc[dollar_volume >= 250_000, "liquidity_score"] = 35
    merged.loc[dollar_volume >= MIN_DOLLAR_VOLUME, "liquidity_score"] = 55
    merged.loc[dollar_volume >= LOW_DOLLAR_VOLUME, "liquidity_score"] = 75
    merged.loc[dollar_volume >= 5_000_000, "liquidity_score"] = 90
    merged.loc[dollar_volume >= 20_000_000, "liquidity_score"] = 100

    quality = pd.Series(100, index=merged.index)
    for col in ["Price_graham", "EPS", "Book Value Per Share", "Sector", "Industry"]:
        quality = quality - merged[col].isna().astype(int) * 15
    quality = quality - (merged["EPS"] <= 0).fillna(False).astype(int) * 25
    quality = quality - (merged["Price_graham"] < LOW_PRICE_PENALTY).fillna(False).astype(int) * 15
    quality = quality - (merged["Dollar Volume"] < LOW_DOLLAR_VOLUME).fillna(False).astype(int) * 15
    merged["data_quality_score"] = quality.clip(0, 100).round(2)

    tier_bonus = merged["research_tier"].map({
        "Tier 1": 20,
        "Tier 2": 12,
        "Tier 3": 6,
        "Tier 4": 0,
    }).fillna(0)

    merged["combined_opportunity_score"] = (
        merged["value_score"] * 0.45
        + merged["technical_score"] * 0.30
        + merged["liquidity_score"] * 0.15
        + merged["data_quality_score"] * 0.10
        + tier_bonus
    ).round(2)


def add_exclusion_flags(merged, ignored_tickers):
    ticker = merged["Ticker"].astype(str).str.upper().str.strip()
    company = merged["Company"].fillna("")
    industry = merged["Industry"].fillna("")

    merged["Excluded From Final"] = False
    merged["Exclusion Reason"] = ""

    exclusions = [
        (ticker.isin(ignored_tickers), "ignore list"),
        (merged["Price_graham"].isna() & merged["Price_technical"].isna(), "missing price"),
        (merged["EPS"].isna() | merged["Book Value Per Share"].isna(), "missing fundamentals"),
        (merged["EPS"] <= 0, "negative EPS"),
        (merged["Price_graham"] < PENNY_STOCK_PRICE, "extreme penny stock"),
        (merged["Dollar Volume"].fillna(0) < MIN_DOLLAR_VOLUME, "low liquidity"),
        (ticker.str.contains(BAD_TICKER_PATTERN, na=False), "warrant/unit/right/preferred ticker"),
        (
            company.str.contains(BAD_SECURITY_PATTERN, na=False)
            | industry.str.contains(BAD_SECURITY_PATTERN, na=False),
            "warrant/unit/right/preferred/SPAC-like security",
        ),
    ]

    for mask, reason in exclusions:
        mask = mask.fillna(False)
        merged.loc[mask, "Excluded From Final"] = True
        merged.loc[mask & (merged["Exclusion Reason"] == ""), "Exclusion Reason"] = reason


def apply_sector_industry_caps(final_candidates):
    # Sector/industry caps are applied after ranking. This keeps a hot industry
    # from crowding out the whole report while preserving the best-ranked names.
    selected = []
    sector_counts = {}
    industry_counts = {}

    for _, row in final_candidates.iterrows():
        sector = str(row.get("Sector") or "Unknown")
        industry = str(row.get("Industry") or "Unknown")

        if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        if industry_counts.get(industry, 0) >= MAX_PER_INDUSTRY:
            continue

        selected.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

        if len(selected) >= TOP_WATCHLIST_LIMIT:
            break

    if not selected:
        return pd.DataFrame(columns=final_candidates.columns)

    return pd.DataFrame(selected)


def remove_duplicate_share_classes(final_candidates):
    final_candidates = final_candidates.copy()
    final_candidates["Company Key"] = final_candidates["Company"].apply(normalize_company_name)
    final_candidates = final_candidates.sort_values(
        by=["combined_opportunity_score", "Dollar Volume"],
        ascending=[False, False],
    )
    return final_candidates.drop_duplicates(subset=["Company Key"], keep="first").drop(
        columns=["Company Key"]
    )


def update_watchlist_history(top_watchlist):
    # History is updated every time this merge script runs. Current watchlist
    # names are inserted or refreshed; absent names keep their prior last_seen.
    create_tables()

    today = date.today().isoformat()
    history_rows = []

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()

        for _, row in top_watchlist.iterrows():
            ticker = str(row["ticker"]).upper()
            latest_score = float(row["combined_opportunity_score"])
            latest_tier = row["research_tier"]
            latest_reason = row["reason"]

            cur.execute(
                """
                SELECT first_seen, highest_score_seen
                FROM watchlist_history
                WHERE ticker = ?
                """,
                (ticker,),
            )
            existing = cur.fetchone()

            if existing:
                first_seen, highest_score_seen = existing
                first_date = datetime.strptime(first_seen, "%Y-%m-%d").date()
                days_on_watchlist = (date.today() - first_date).days + 1
                highest_score_seen = max(float(highest_score_seen or 0), latest_score)
                cur.execute(
                    """
                    UPDATE watchlist_history
                    SET last_seen = ?,
                        days_on_watchlist = ?,
                        highest_score_seen = ?,
                        latest_score = ?,
                        latest_tier = ?,
                        latest_reason = ?
                    WHERE ticker = ?
                    """,
                    (
                        today,
                        days_on_watchlist,
                        highest_score_seen,
                        latest_score,
                        latest_tier,
                        latest_reason,
                        ticker,
                    ),
                )
            else:
                first_seen = today
                days_on_watchlist = 1
                highest_score_seen = latest_score
                cur.execute(
                    """
                    INSERT INTO watchlist_history (
                        ticker,
                        first_seen,
                        last_seen,
                        days_on_watchlist,
                        highest_score_seen,
                        latest_score,
                        latest_tier,
                        latest_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        today,
                        today,
                        days_on_watchlist,
                        highest_score_seen,
                        latest_score,
                        latest_tier,
                        latest_reason,
                    ),
                )

            history_rows.append({
                "ticker": ticker,
                "days_on_watchlist": days_on_watchlist,
            })

        conn.commit()

    return pd.DataFrame(history_rows)


def write_daily_summary(merged, top_watchlist, previous_tickers):
    current_tickers = set(top_watchlist["ticker"].dropna().astype(str).str.upper())
    new_names = sorted(current_tickers - previous_tickers)
    dropped_names = sorted(previous_tickers - current_tickers)

    tier_counts = top_watchlist["research_tier"].value_counts().to_dict()
    top_10 = top_watchlist["ticker"].head(10).tolist()

    lines = [
        f"Daily research summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total scanned: {len(merged)}",
        f"Graham candidates: {int(to_bool(merged['Graham Undervalued']).sum())}",
        f"Panic candidates: {int(to_bool(merged['Technical Panic Flag']).sum())}",
        f"Final watchlist count: {len(top_watchlist)}",
        f"Tier 1 count: {tier_counts.get('Tier 1', 0)}",
        f"Tier 2 count: {tier_counts.get('Tier 2', 0)}",
        f"Tier 3 count: {tier_counts.get('Tier 3', 0)}",
        f"Tier 4 count: {tier_counts.get('Tier 4', 0)}",
        f"Top 10 tickers: {', '.join(top_10) if top_10 else 'None'}",
        f"New names versus previous run: {', '.join(new_names) if new_names else 'None'}",
        f"Dropped names versus previous run: {', '.join(dropped_names) if dropped_names else 'None'}",
    ]

    with open(DAILY_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_top_watchlist(merged):
    previous_tickers = set()
    if os.path.exists(TOP_WATCHLIST_FILE):
        try:
            previous = pd.read_csv(TOP_WATCHLIST_FILE)
            if "ticker" in previous.columns:
                previous_tickers = set(previous["ticker"].dropna().astype(str).str.upper())
        except Exception as e:
            print(f"Could not read previous top watchlist: {e}")

    ignored_tickers = load_ignored_tickers()

    merged["research_tier"] = merged.apply(assign_research_tier, axis=1)
    merged["reason"] = merged.apply(assign_reason, axis=1)
    add_scores(merged)
    add_exclusion_flags(merged, ignored_tickers)

    final_candidates = merged[
        (merged["Research Watchlist"] == True)
        & (merged["research_tier"] != "")
        & (merged["Excluded From Final"] == False)
    ].copy()

    final_candidates = remove_duplicate_share_classes(final_candidates)
    final_candidates = final_candidates.sort_values(
        by=["research_tier", "combined_opportunity_score", "value_score", "technical_score"],
        ascending=[True, False, False, False],
    )
    final_candidates = apply_sector_industry_caps(final_candidates)

    top_watchlist = pd.DataFrame({
        "ticker": final_candidates["Ticker"],
        "company_name": final_candidates["Company"],
        "sector": final_candidates["Sector"],
        "industry": final_candidates["Industry"],
        "price": final_candidates["Price_graham"].combine_first(final_candidates["Price_technical"]),
        "research_tier": final_candidates["research_tier"],
        "reason": final_candidates["reason"],
        "margin_of_safety": final_candidates["Discount %"],
        "graham_score": final_candidates["Value Score"],
        "price_change_5d": final_candidates["5D Price Change %"],
        "volume_spike": final_candidates["Volume Spike"],
        "technical_panic_score": final_candidates["Technical Panic Score"],
        "dollar_volume": final_candidates["Dollar Volume"],
        "value_score": final_candidates["value_score"],
        "technical_score": final_candidates["technical_score"],
        "liquidity_score": final_candidates["liquidity_score"],
        "data_quality_score": final_candidates["data_quality_score"],
        "combined_opportunity_score": final_candidates["combined_opportunity_score"],
    })

    history_days = update_watchlist_history(top_watchlist)
    top_watchlist = top_watchlist.merge(history_days, on="ticker", how="left")
    top_watchlist["days_on_watchlist"] = top_watchlist["days_on_watchlist"].fillna(1).astype(int)

    ordered_columns = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "price",
        "research_tier",
        "reason",
        "margin_of_safety",
        "graham_score",
        "price_change_5d",
        "volume_spike",
        "technical_panic_score",
        "dollar_volume",
        "value_score",
        "technical_score",
        "liquidity_score",
        "data_quality_score",
        "combined_opportunity_score",
        "days_on_watchlist",
    ]

    top_watchlist[ordered_columns].to_csv(TOP_WATCHLIST_FILE, index=False)
    write_daily_summary(merged, top_watchlist, previous_tickers)

    return top_watchlist


# -------------------------
# LOAD ALL BATCHES
# -------------------------

graham = load_batch_files("graham_batch_*.csv", "Graham")
technical = load_batch_files("technical_batch_*.csv", "Technical")

if graham.empty:
    print("No Graham data found. Run graham_screener.py first.")

if technical.empty:
    print("No technical data found. Run technical_scanner.py first.")

if graham.empty or technical.empty:
    print("Merge stopped.")
else:
    graham.to_csv(GRAHAM_MASTER_FILE, index=False)
    technical.to_csv(TECHNICAL_MASTER_FILE, index=False)

    merged = pd.merge(
        graham,
        technical,
        on="Ticker",
        how="inner",
        suffixes=("_graham", "_technical"),
    )

    if merged.empty:
        print("No overlapping tickers found between Graham and technical data.")
    else:
        numeric_cols = [
            "Discount %",
            "Value Score",
            "Technical Panic Score",
            "5D Price Change %",
            "Volume Spike",
            "Dollar Volume",
            "Price_graham",
            "Price_technical",
            "EPS",
            "Book Value Per Share",
        ]
        clean_numeric(merged, numeric_cols)

        for flag_col in [
            "Graham Undervalued",
            "Deep Graham Discount",
            "Technical Panic Flag",
            "Oversold Flag",
            "Volume Spike Flag",
            "Momentum Spike Flag",
        ]:
            if flag_col in merged.columns:
                merged[flag_col] = to_bool(merged[flag_col])

        # True Graham panic opportunity:
        # Undervalued by Graham AND actually selling off.
        merged["Graham Panic Opportunity"] = (
            (merged["Graham Undervalued"] == True)
            & (merged["Discount %"] > 0)
            & (merged["Technical Panic Score"] >= 7)
            & (merged["5D Price Change %"] <= -10)
        )

        # Softer legacy research watchlist. The new top_research_watchlist.csv
        # applies stricter exclusions, tiers, scores, and group caps afterward.
        merged["Research Watchlist"] = (
            (merged["Graham Undervalued"] == True)
            | (merged["Deep Graham Discount"] == True)
            | (merged["Technical Panic Flag"] == True)
            | (merged["Oversold Flag"] == True)
            | (merged["Volume Spike Flag"] == True)
            | (merged["Momentum Spike Flag"] == True)
        )

        # Legacy score kept for existing combined_opportunities.csv consumers.
        merged["Combined Opportunity Score"] = (
            merged["Value Score"].fillna(0) * 0.45
            + merged["Discount %"].fillna(0) * 0.35
            + merged["Technical Panic Score"].fillna(0) * 4
        )

        merged = merged.sort_values(
            by=[
                "Graham Panic Opportunity",
                "Combined Opportunity Score",
                "Discount %",
                "Technical Panic Score",
            ],
            ascending=[False, False, False, False],
        )

        top_watchlist = build_top_watchlist(merged)

        merged.to_csv(COMBINED_FILE, index=False)

        merged[merged["Graham Panic Opportunity"] == True].to_csv(
            GRAHAM_PANIC_FILE,
            index=False,
        )

        merged[merged["Research Watchlist"] == True].to_csv(
            WATCHLIST_FILE,
            index=False,
        )

        print()
        print("Done.")
        print("Graham master rows:", len(graham))
        print("Technical master rows:", len(technical))
        print("Merged overlapping tickers:", len(merged))
        print("Graham panic opportunities:", len(merged[merged["Graham Panic Opportunity"] == True]))
        print("Research watchlist:", len(merged[merged["Research Watchlist"] == True]))
        print("Top research watchlist:", len(top_watchlist))

        columns_to_show = [
            "Ticker",
            "Company",
            "Price_graham",
            "Graham Number",
            "Discount %",
            "Value Score",
            "5D Price Change %",
            "Volume Spike",
            "Technical Panic Score",
            "Combined Opportunity Score",
            "combined_opportunity_score",
            "research_tier",
            "Graham Panic Opportunity",
        ]

        available_columns = [
            col for col in columns_to_show
            if col in merged.columns
        ]

        print()
        print(merged[available_columns].head(25))
