import sqlite3
conn = sqlite3.connect("data/nfl.db")
cols = [r[1] for r in conn.execute("PRAGMA table_info(games)")]
print("spread_line" in cols)
print(cols)
conn.close()