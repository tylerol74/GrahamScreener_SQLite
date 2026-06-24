import sqlite3
from datetime import datetime

conn = sqlite3.connect("value_screener.db")
cur = conn.cursor()

cur.execute("""
INSERT INTO technical_results (
    ticker,
    price,
    price_change_5d,
    price_change_10d,
    latest_volume,
    avg_volume_20d,
    volume_spike,
    dollar_volume,
    distance_from_20d_avg,
    technical_panic_score,
    technical_panic_flag,
    volume_spike_flag,
    oversold_flag,
    batch_number,
    last_updated
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    "TEST",
    10.50,
    -12.3,
    -18.7,
    1000000,
    500000,
    2.0,
    10500000,
    -9.5,
    8,
    1,
    1,
    1,
    1,
    datetime.now().isoformat()
))

conn.commit()

cur.execute("""
SELECT *
FROM technical_results
WHERE ticker = 'TEST'
""")

row = cur.fetchone()

print(row)

conn.close()