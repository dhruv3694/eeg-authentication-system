"""
Test the new similarity math against all user pairs from the DB
"""
import json, sqlite3, math

DB = "eeg_auth.db"
KEYS = [
    "alpha_peak_est_hz","rms_uv","variance_uv2","line_length","zero_cross_rate",
    "hjorth_activity","hjorth_mobility","hjorth_complexity","spectral_entropy",
    "theta_alpha_ratio","beta_alpha_ratio","gamma_theta_ratio","delta_gamma_ratio",
    "total_spectral_power","estimated_interhemispheric_coherence",
    "posterior_alpha_asym","frontal_beta_asym",
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "delta_rel","theta_rel","alpha_rel","beta_rel","gamma_rel",
]

def vectorize(feat):
    try: return [float(feat[k]) for k in KEYS]
    except: return []

def _mag(v): return math.sqrt(sum(x*x for x in v)) or 1.0

def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1,v2))
    raw = dot / (_mag(v1)*_mag(v2))
    return round(max(0.0, min(1.0, (raw+1.0)/2.0)), 4)

def _zn(v):
    n=len(v); m=sum(v)/n; s=math.sqrt(sum((x-m)**2 for x in v)/n) or 1.0
    return [(x-m)/s for x in v]

def euclidean_similarity(v1, v2):
    z1,z2 = _zn(v1),_zn(v2)
    dist = math.sqrt(sum((a-b)**2 for a,b in zip(z1,z2)))
    return round(max(0.0, 1.0-dist/(2.0*math.sqrt(len(v1)))), 4)

def per_feature_relative_score(v1, v2):
    scores = []
    for a,b in zip(v1,v2):
        if abs(a)<1e-9 and abs(b)<1e-9: scores.append(1.0)
        elif a>0 and b>0: scores.append(min(a,b)/max(a,b))
        else:
            mx=max(abs(a),abs(b),1e-9); scores.append(max(0.0,1.0-abs(a-b)/mx))
    return round(sum(scores)/len(scores), 4)

def ensemble(v1, v2):
    cos=cosine_similarity(v1,v2); euc=euclidean_similarity(v1,v2)
    feat=per_feature_relative_score(v1,v2)
    return round(0.35*cos + 0.35*euc + 0.30*feat, 4)

conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("SELECT u.uid, u.name, b.features_json FROM users u JOIN brainprints b ON u.uid=b.user_id")
rows = c.fetchall()
conn.close()

users = {uid: {"name": name, "vec": vectorize(json.loads(fj))} for uid,name,fj in rows}
uids = sorted(users.keys())

print("="*70)
print(f"{'Pair':40s}  cos    euc    feat   ENS    TYPE")
print("="*70)
for u1 in uids:
    for u2 in uids:
        v1,v2 = users[u1]["vec"], users[u2]["vec"]
        cos  = cosine_similarity(v1,v2)
        euc  = euclidean_similarity(v1,v2)
        feat = per_feature_relative_score(v1,v2)
        ens  = ensemble(v1,v2)
        kind = "SAME" if u1==u2 else "DIFF"
        pair = f"{users[u1]['name'][:14]:14s} vs {users[u2]['name'][:14]:14s}"
        print(f"  {pair:36s}  {cos:.4f} {euc:.4f} {feat:.4f} {ens:.4f}  [{kind}]")

print("\nThreshold test at 0.72:")
for u1 in uids:
    for u2 in uids:
        v1,v2 = users[u1]["vec"], users[u2]["vec"]
        ens = ensemble(v1,v2)
        verdict = "GRANTED" if ens >= 0.72 else "DENIED"
        expected = "GRANTED" if u1==u2 else "DENIED"
        ok = "OK" if verdict==expected else "!!! WRONG !!!"
        print(f"  UID {u1} vs UID {u2}: {ens:.4f} -> {verdict} (expected {expected}) {ok}")
