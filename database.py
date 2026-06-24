import sqlite3

DB_FILE = "value_screener.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scan_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_type TEXT,
        batch_number INTEGER,
        batch_size INTEGER,
        started_at TEXT,
        finished_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        ticker TEXT PRIMARY KEY,
        company_name TEXT,
        exchange TEXT,
        last_updated TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS graham_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        price REAL,
        eps REAL,
        book_value REAL,
        graham_number REAL,
        margin_of_safety REAL,
        graham_pass INTEGER,
        batch_number INTEGER,
        last_updated TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS technical_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        price REAL,
        price_change_5d REAL,
        price_change_10d REAL,
        latest_volume REAL,
        avg_volume_20d REAL,
        volume_spike REAL,
        dollar_volume REAL,
        distance_from_20d_avg REAL,
        technical_panic_score INTEGER,
        technical_panic_flag INTEGER,
        volume_spike_flag INTEGER,
        oversold_flag INTEGER,
        momentum_spike_flag INTEGER,
        batch_number INTEGER,
        last_updated TEXT,
        UNIQUE(ticker, batch_number)
    )
    """)

    cur.execute("PRAGMA table_info(technical_results)")
    technical_columns = {row[1] for row in cur.fetchall()}
    if "momentum_spike_flag" not in technical_columns:
        cur.execute("""
        ALTER TABLE technical_results
        ADD COLUMN momentum_spike_flag INTEGER
        """)

    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_technical_results_ticker_batch
    ON technical_results (ticker, batch_number)
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS opportunities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        price REAL,
        graham_number REAL,
        margin_of_safety REAL,
        graham_pass INTEGER,
        technical_panic_score INTEGER,
        technical_panic_flag INTEGER,
        volume_spike REAL,
        dollar_volume REAL,
        batch_number INTEGER,
        last_updated TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS watchlist_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        first_seen TEXT,
        last_seen TEXT,
        days_on_watchlist INTEGER,
        highest_score_seen REAL,
        latest_score REAL,
        latest_tier TEXT,
        latest_reason TEXT
    )
    """)

    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_history_ticker
    ON watchlist_history (ticker)
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    print("Database and tables created successfully.")
