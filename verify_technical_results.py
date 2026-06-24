import sqlite3

from database import DB_FILE


def main():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS row_count FROM technical_results")
        row_count = cur.fetchone()["row_count"]

        cur.execute("SELECT MAX(batch_number) AS latest_batch FROM technical_results")
        latest_batch = cur.fetchone()["latest_batch"]

        cur.execute("""
        SELECT
            ticker,
            price,
            price_change_5d,
            price_change_10d,
            latest_volume,
            volume_spike,
            technical_panic_score,
            technical_panic_flag,
            volume_spike_flag,
            oversold_flag,
            momentum_spike_flag,
            batch_number,
            last_updated
        FROM technical_results
        ORDER BY id
        LIMIT 10
        """)
        rows = cur.fetchall()

    print(f"technical_results row count: {row_count}")
    print(f"most recent batch_number: {latest_batch}")
    print()
    print("first 10 rows:")

    if not rows:
        print("(no rows)")
        return

    for row in rows:
        print(dict(row))


if __name__ == "__main__":
    main()
