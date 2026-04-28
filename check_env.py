import sys
try:
    import numpy as np
    print(f"numpy {np.__version__}")
except ImportError:
    print("numpy: NOT FOUND")

try:
    import sklearn
    print(f"sklearn {sklearn.__version__}")
except ImportError:
    print("sklearn: NOT FOUND")

try:
    import torch
    print(f"torch {torch.__version__}")
except ImportError:
    print("torch: NOT FOUND")

import sqlite3, json, os
conn = sqlite3.connect("eeg_auth.db")
c = conn.cursor()
c.execute("SELECT uid, name FROM users")
users = c.fetchall()
print(f"\nUsers in DB: {users}")
c.execute("SELECT user_id, length(raw_data_json), length(features_json) FROM brainprints")
bps = c.fetchall()
print(f"Brainprints: {[(uid, f'raw={rlen}b', f'feat={flen}b') for uid,rlen,flen in bps]}")
conn.close()
