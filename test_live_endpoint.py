import urllib.request, json, time

time.sleep(2)  # let simulation warm up

# Check users
r = urllib.request.urlopen('http://127.0.0.1:8000/api/users')
users = json.loads(r.read())
print("Users:", [(u['id'], u['name']) for u in users['users']])

if not users['users']:
    print("No users!")
    exit(1)

uid = users['users'][0]['id']
print(f"\nTesting live features endpoint for UID {uid}...")
r2 = urllib.request.urlopen(f'http://127.0.0.1:8000/api/users/{uid}/features/live')
feat = json.loads(r2.read())
if 'alpha_peak_est_hz' in feat:
    print(f"SUCCESS - alpha_peak_est_hz = {feat['alpha_peak_est_hz']}")
    print(f"  hjorth_mobility  = {feat.get('hjorth_mobility')}")
    print(f"  spectral_entropy = {feat.get('spectral_entropy')}")
    print(f"  ensemble keys    = {len([k for k in feat if k in ['alpha_power','beta_power','gamma_power']])}/3 band powers present")
else:
    print("FAIL - missing biometric keys:", list(feat.keys())[:5])
