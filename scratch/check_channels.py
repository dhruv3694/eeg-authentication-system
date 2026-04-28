import sqlite3, json
conn=sqlite3.connect('eeg_auth.db')
c=conn.cursor()
c.execute('SELECT user_id, raw_data_json FROM brainprints')
for u, r in c.fetchall():
    if r:
        raw = json.loads(r)
        chans = list(raw["channels"].keys())
        print(f"UID {u}: {len(chans)} channels")
    else:
        print(f"UID {u}: NO RAW DATA")
