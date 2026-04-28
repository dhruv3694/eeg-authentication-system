"""
Full end-to-end test: self-test + imposter test via auth_server API
"""
import urllib.request, json, time

def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), 
                                  headers={'Content-Type': 'application/json'}, method='POST')
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

def get(url):
    return json.loads(urllib.request.urlopen(url).read())

time.sleep(1)

# Get enrolled users
enrolled = get('http://127.0.0.1:8001/api/auth/enrolled')['users']
print(f"Enrolled users: {[(u['uid'], u['name'], u['fingerprint']) for u in enrolled]}\n")

if len(enrolled) < 1:
    print("Need at least 1 enrolled user. Add users in the simulator first.")
    exit(1)

uid = enrolled[0]['uid']

# ── Self-test (should be very high similarity) ──────────────────────────────
print(f"=== SELF-TEST: UID {uid} vs its OWN stored brainprint ===")
r = post('http://127.0.0.1:8001/api/auth/verify', {'probe_uid': uid, 'target_uid': uid})
print(f"  Cosine:    {r['cosine_similarity']:.4f}")
print(f"  Euclidean: {r['euclidean_similarity']:.4f}")
print(f"  Ensemble:  {r['ensemble_score']:.4f}")
print(f"  Verdict:   {r['result']}")
print(f"  Stored FP: {r['fingerprint_stored']}")
print(f"  Live   FP: {r['fingerprint_live']}")
print()

# Top 3 most divergent features
bd = r['feature_breakdown']
worst = sorted(bd, key=lambda x: x['diff_pct'], reverse=True)[:3]
print("  Most divergent features (expected: small diffs for self-test):")
for f in worst:
    print(f"    {f['key']:40s} stored={f['stored']:.5f} live={f['live']:.5f} Δ={f['diff_pct']:.1f}%")
