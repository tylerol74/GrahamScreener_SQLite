import sqlite3

conn = sqlite3.connect("value_screener.db")
cur = conn.cursor()

cur.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

tables = cur.fetchall()

print("\nTables:\n")

for table in tables:
    print(table[0])

conn.close()