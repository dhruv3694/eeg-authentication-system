"""
Test: compare LIVE features from simulator vs STORED features in DB
This mirrors exactly what auth_server.verify() does.
"""
import json, sqlite3, math, urllib.request, time

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
def _zn(v):
    n=len(v); m=sum(v)/n; s=math.sqrt(sum((x-m)**2 for x in v)/n) or 1.0
    return [(x-m)/s for x in v]
def cosine(v1,v2):
    dot=sum(a*b for a,b in zip(v1,v2)); raw=dot/(_mag(v1)*_mag(v2))
    return round(max(0.0,min(1.0,(raw+1.0)/2.0)), 4)
def euclid(v1,v2):
    z1,z2=_zn(v1),_zn(v2); d=math.sqrt(sum((a-b)**2 for a,b in zip(z1,z2)))
    return round(max(0.0,1.0-d/(2.0*math.sqrt(len(v1)))), 4)
def feat_score(v1,v2):
    s=[]
    for a,b in zip(v1,v2):
        if abs(a)<1e-9 and abs(b)<1e-9: s.append(1.0)
        elif a>0 and b>0: s.append(min(a,b)/max(a,b))
        else: mx=max(abs(a),abs(b),1e-9); s.append(max(0.0,1.0-abs(a-b)/mx))
    return round(sum(s)/len(s), 4)
def ensemble(v1,v2): return round(0.35*cosine(v1,v2)+0.35*euclid(v1,v2)+0.30*feat_score(v1,v2), 4)

# Load stored from DB
conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("SELECT u.uid, u.name, b.features_json FROM users u JOIN brainprints b ON u.uid=b.user_id")
rows = c.fetchall()
conn.close()
stored = {uid: {"name":name,"vec":vectorize(json.loads(fj))} for uid,name,fj in rows}

print("="*75)
print(f"{'Pair (live vs stored)':45s}  cos   euc   feat  ENS   verdict")
print("="*75)
THRESHOLD = 0.95

for probe_uid in sorted(stored.keys()):
    # Get live features from simulator
    url = f"http://127.0.0.1:8000/api/users/{probe_uid}/features/live"
    try:
        r = urllib.request.urlopen(url, timeout=3)
        live_feat = json.loads(r.read())
        live_vec = vectorize(live_feat)
    except Exception as e:
        print(f"  UID {probe_uid}: Cannot get live features: {e}")
        continue

    for target_uid in sorted(stored.keys()):
        stored_vec = stored[target_uid]["vec"]
        cos  = cosine(live_vec, stored_vec)
        euc  = euclid(live_vec, stored_vec)
        feat = feat_score(live_vec, stored_vec)
        ens  = ensemble(live_vec, stored_vec)
        verdict = "GRANTED" if ens >= THRESHOLD else "DENIED"
        kind = "SAME" if probe_uid==target_uid else "DIFF"
        expected = "GRANTED" if probe_uid==target_uid else "DENIED"
        ok = "" if verdict==expected else "  !!! WRONG !!!"
        pn = stored[probe_uid]["name"][:12]
        tn = stored[target_uid]["name"][:12]
        print(f"  live:{pn:12s} vs stored:{tn:12s}  {cos:.3f} {euc:.3f} {feat:.3f} {ens:.3f}  {verdict} [{kind}]{ok}")

print(f"\nThreshold: {THRESHOLD}")
