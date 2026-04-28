"""
Diagnostic: check all 3 users' features in DB and test similarity scores
"""
import json, sqlite3, math

DB = "eeg_auth.db"
conn = sqlite3.connect(DB)
c = conn.cursor()

c.execute("SELECT u.uid, u.name, b.features_json FROM users u LEFT JOIN brainprints b ON u.uid=b.user_id")
rows = c.fetchall()
conn.close()

KEYS = [
    "alpha_peak_est_hz","rms_uv","variance_uv2","line_length","zero_cross_rate",
    "hjorth_activity","hjorth_mobility","hjorth_complexity","spectral_entropy",
    "theta_alpha_ratio","beta_alpha_ratio","gamma_theta_ratio","delta_gamma_ratio",
    "total_spectral_power","estimated_interhemispheric_coherence",
    "posterior_alpha_asym","frontal_beta_asym",
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "delta_rel","theta_rel","alpha_rel","beta_rel","gamma_rel",
]

def vec(feat):
    try:
        return [float(feat[k]) for k in KEYS]
    except:
        return []

def cosine(v1, v2):
    if not v1 or not v2: return 0.0
    dot = sum(a*b for a,b in zip(v1,v2))
    m1 = math.sqrt(sum(x*x for x in v1))
    m2 = math.sqrt(sum(x*x for x in v2))
    return dot / (m1*m2 + 1e-9)

users = {}
for uid, name, feat_json in rows:
    feat = json.loads(feat_json) if feat_json else {}
    users[uid] = {"name": name, "vec": vec(feat)}
    print(f"UID {uid} ({name}): {len(users[uid]['vec'])} features, alpha_peak={feat.get('alpha_peak_est_hz','?')}")

print("\n--- Cosine similarity matrix ---")
uids = list(users.keys())
for i, u1 in enumerate(uids):
    for u2 in uids:
        v1 = users[u1]['vec']
        v2 = users[u2]['vec']
        cos = cosine(v1, v2)
        same = "SAME" if u1 == u2 else "DIFF"
        print(f"  UID {u1} ({users[u1]['name'][:12]}) vs UID {u2} ({users[u2]['name'][:12]}): cosine={cos:.4f} [{same}]")
