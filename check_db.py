import sqlite3
conn = sqlite3.connect("eeg_auth.db")
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM users")
print(f"Users: {c.fetchone()[0]}")
c.execute("SELECT COUNT(*) FROM brainprints")
print(f"Brainprints: {c.fetchone()[0]}")
conn.close()
