# check_technical_rows.py

import sqlite3

conn = sqlite3.connect("value_screener.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM technical_results")

print("Rows:", cur.fetchone()[0])

conn.close()