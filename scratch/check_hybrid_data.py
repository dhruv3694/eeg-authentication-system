import dl_auth
import json

data = dl_auth.load_hybrid_data_from_db()
print(f"Found {len(data)} users with hybrid data")
for uid, d in data.items():
    print(f"  UID {uid}: {len(d['windows'])} windows, handcrafted shape {d['handcrafted'].shape}")
