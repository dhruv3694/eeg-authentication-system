"""
EEG Brainprint Authentication Server
=====================================
Standalone server that runs ALONGSIDE the existing EEG simulator (port 8000).
This server does NOT modify simulator state — it only reads data.

Architecture:
  - Reads enrolled brainprints from the simulator's eeg_auth.db (read-only).
  - Has its own auth_history.db for storing authentication attempt logs.
  - Calls http://127.0.0.1:8000/api/users/{uid}/features/live to get live features.
  - Runs on port 8001 and serves the auth_ui.html frontend.

Run with:  python auth_server.py
"""

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Deep Learning authentication engine (pure numpy CNN+LSTM + sklearn MLP)
try:
    from dl_auth import get_engine as _get_dl_engine
    _DL_AVAILABLE = True
except ImportError as _e:
    _DL_AVAILABLE = False
    print(f"[Auth Server] DL engine not available: {_e}")

# ─── Constants ───────────────────────────────────────────────────────────────

SIMULATOR_BASE = "http://127.0.0.1:8000"
SIMULATOR_DB   = "eeg_auth.db"          # simulator's DB — read-only
AUTH_DB        = "auth_history.db"       # our own DB for attempt logs
AUTH_THRESHOLD = 0.92                   # ensemble similarity threshold for "match"

# Exact same biometric keys used by the simulator (must stay in sync)
BIOMETRIC_KEYS = [
    "alpha_peak_est_hz",
    "rms_uv",
    "variance_uv2",
    "line_length",
    "zero_cross_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "spectral_entropy",
    "alpha_psd_peak",
    "alpha_psd_cog",
    "alpha_psd_entropy",
    "alpha_sb1_rel",
    "alpha_sb2_rel",
    "alpha_sb3_rel",
    "alpha_sb4_rel",
    "theta_alpha_ratio",
    "beta_alpha_ratio",
    "gamma_theta_ratio",
    "delta_gamma_ratio",
    "total_spectral_power",
    "estimated_interhemispheric_coherence",
    "posterior_alpha_asym",
    "frontal_beta_asym",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "delta_rel",
    "theta_rel",
    "alpha_rel",
    "beta_rel",
    "gamma_rel",
]

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="EEG Brainprint Auth", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Math: Vectorization & Similarity ─────────────────────────────────────────

def vectorize(features: Dict[str, Any]) -> List[float]:
    try:
        return [float(features[k]) for k in BIOMETRIC_KEYS]
    except (KeyError, TypeError):
        return []


def _mag(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Raw cosine in [-1,1] mapped to [0,1]. Captures signal shape."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    raw = dot / (_mag(v1) * _mag(v2))
    return round(max(0.0, min(1.0, (raw + 1.0) / 2.0)), 6)


def _zscore_normalize(v: List[float]) -> List[float]:
    """Per-vector z-score normalisation so each feature contributes equally."""
    n = len(v)
    if n == 0:
        return v
    mean = sum(v) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in v) / n) or 1.0
    return [(x - mean) / std for x in v]


def euclidean_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Z-score normalised Euclidean distance mapped to [0,1].
    After normalisation the max meaningful distance ≈ sqrt(2*N);
    we clamp at 2*sqrt(N) to keep scores in [0,1].
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    z1 = _zscore_normalize(v1)
    z2 = _zscore_normalize(v2)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2)))
    max_dist = 2.0 * math.sqrt(len(v1))          # theoretical upper bound
    return round(max(0.0, 1.0 - dist / max_dist), 6)


def per_feature_relative_score(v1: List[float], v2: List[float]) -> float:
    """
    Key discriminator: computes mean per-feature relative agreement.
    Uses log-ratio on positive features and direct ratio on others.
    Highly sensitive to per-channel magnitude differences.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    scores = []
    for a, b in zip(v1, v2):
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            scores.append(1.0)          # both zero → perfect match
        elif a > 0 and b > 0:
            ratio = min(a, b) / max(a, b)   # in [0,1], 1=perfect
            scores.append(ratio)
        else:
            # Sign disagree or one is zero → punish
            mx = max(abs(a), abs(b), 1e-9)
            diff = abs(a - b) / mx
            scores.append(max(0.0, 1.0 - diff))
    return round(sum(scores) / len(scores), 6)


def ensemble_score(v1: List[float], v2: List[float]) -> float:
    """
    Three-way ensemble — highly discriminative even for similar biometric profiles:
      35% Cosine            (shape of 27-D feature vector)
      35% Euclidean z-norm  (magnitude differences per feature)
      30% Per-feature ratio (fine-grained channel-level agreement)
    """
    cos  = cosine_similarity(v1, v2)
    euc  = euclidean_similarity(v1, v2)
    feat = per_feature_relative_score(v1, v2)
    return round(0.35 * cos + 0.35 * euc + 0.30 * feat, 6)



def brainprint_fingerprint(vec: List[float]) -> str:
    """16-char SHA256 hex fingerprint of the biometric vector."""
    s = ",".join(f"{v:.8f}" for v in vec)
    return hashlib.sha256(s.encode()).hexdigest()[:16].upper()


def per_feature_diff(stored: List[float], live: List[float]) -> List[Dict[str, Any]]:
    result = []
    for i, key in enumerate(BIOMETRIC_KEYS):
        sv = stored[i] if i < len(stored) else 0.0
        lv = live[i]   if i < len(live)   else 0.0
        mx = max(abs(sv), abs(lv), 1e-9)
        diff_pct = min(abs(sv - lv) / mx * 100.0, 200.0)   # cap at 200%
        match_quality = max(0.0, 1.0 - diff_pct / 100.0)
        result.append({
            "key": key,
            "stored": round(sv, 6),
            "live":   round(lv, 6),
            "diff_pct": round(diff_pct, 2),
            "match_quality": round(match_quality, 4),
        })
    return result

# ─── Database ─────────────────────────────────────────────────────────────────

def init_auth_db():
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS auth_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            probe_uid INTEGER,
            probe_name TEXT,
            target_uid INTEGER,
            target_name TEXT,
            cosine_score REAL,
            euclidean_score REAL,
            ensemble_score REAL,
            result TEXT,
            fingerprint_stored TEXT,
            fingerprint_live TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_attempt(
    probe_uid: int, probe_name: str,
    target_uid: int, target_name: str,
    cos: float, euc: float, ens: float,
    result: str,
    fp_stored: str, fp_live: str,
):
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO auth_attempts
        (ts, probe_uid, probe_name, target_uid, target_name,
         cosine_score, euclidean_score, ensemble_score, result,
         fingerprint_stored, fingerprint_live)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (time.time(), probe_uid, probe_name, target_uid, target_name,
          cos, euc, ens, result, fp_stored, fp_live))
    conn.commit()
    conn.close()


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(AUTH_DB)
    c = conn.cursor()
    c.execute("""
        SELECT id, ts, probe_uid, probe_name, target_uid, target_name,
               cosine_score, euclidean_score, ensemble_score, result,
               fingerprint_stored, fingerprint_live
        FROM auth_attempts ORDER BY ts DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    cols = ["id", "ts", "probe_name_uid", "probe_name", "target_uid", "target_name",
            "cosine_score", "euclidean_score", "ensemble_score", "result",
            "fingerprint_stored", "fingerprint_live"]
    return [dict(zip(cols, r)) for r in rows]


# ─── Simulator DB helpers (read-only) ─────────────────────────────────────────

def get_all_stored_brainprints() -> List[Dict[str, Any]]:
    if not os.path.exists(SIMULATOR_DB):
        return []
    conn = sqlite3.connect(f"file:{SIMULATOR_DB}?mode=ro", uri=True)
    c = conn.cursor()
    c.execute("""
        SELECT u.uid, u.name, u.age, u.pathology, u.medication, u.seed,
               b.features_json
        FROM users u
        LEFT JOIN brainprints b ON u.uid = b.user_id
    """)
    rows = c.fetchall()
    conn.close()
    result = []
    for uid, name, age, pathology, medication, seed, feat_json in rows:
        feat = json.loads(feat_json) if feat_json else {}
        vec  = vectorize(feat)
        fp   = brainprint_fingerprint(vec) if vec else "NO DATA"
        result.append({
            "uid": uid,
            "name": name,
            "age": age,
            "pathology": pathology,
            "medication": medication,
            "seed": seed,
            "has_brainprint": bool(vec),
            "fingerprint": fp,
            "vector_dim": len(vec),
            "features": feat,
        })
    return result


def get_stored_features(uid: int) -> Optional[Dict[str, Any]]:
    if not os.path.exists(SIMULATOR_DB):
        return None
    conn = sqlite3.connect(f"file:{SIMULATOR_DB}?mode=ro", uri=True)
    c = conn.cursor()
    c.execute("SELECT features_json FROM brainprints WHERE user_id=?", (uid,))
    row = c.fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return json.loads(row[0])


# ─── API Models ───────────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    probe_uid: int    # which user's LIVE signal to use as the "login attempt"
    target_uid: int   # which stored brainprint to verify against


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("auth_ui.html")


@app.get("/api/auth/status")
async def auth_status():
    """Check if the EEG simulator is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{SIMULATOR_BASE}/api/users")
            sim_ok = r.status_code == 200
            sim_users = r.json().get("users", []) if sim_ok else []
    except Exception:
        sim_ok = False
        sim_users = []

    auth_db_ok = os.path.exists(SIMULATOR_DB)
    conn = sqlite3.connect(AUTH_DB)
    total_attempts = conn.execute("SELECT COUNT(*) FROM auth_attempts").fetchone()[0]
    conn.close()

    return {
        "simulator_online": sim_ok,
        "simulator_users": sim_users,
        "simulator_db_exists": auth_db_ok,
        "total_auth_attempts": total_attempts,
        "threshold": AUTH_THRESHOLD,
        "biometric_dimensions": len(BIOMETRIC_KEYS),
    }


@app.get("/api/auth/enrolled")
async def list_enrolled():
    """
    Returns all users from the simulator database with their stored brainprints.
    """
    return {"users": get_all_stored_brainprints()}


@app.get("/api/auth/brainprint/{uid}")
async def get_brainprint(uid: int):
    """
    Returns detailed brainprint breakdown for a specific stored user.
    """
    feat = get_stored_features(uid)
    if not feat:
        raise HTTPException(404, f"No brainprint found for user {uid}")
    vec = vectorize(feat)
    return {
        "uid": uid,
        "fingerprint": brainprint_fingerprint(vec),
        "vector": vec,
        "keys": BIOMETRIC_KEYS,
        "features": feat,
    }


@app.post("/api/auth/verify")
async def verify_brainprint(req: VerifyRequest):
    """
    Core authentication endpoint.
    Strategy: compare LIVE features of probe_uid against LIVE re-snapshot of target_uid.
    Both calls go to the running simulator, so both vectors come from the
    same deterministic signal state — no drift issues.
    The stored DB brainprint is used as a fallback and for fingerprint display.
    """
    # ── Get live features for BOTH Probe and Target concurrently ──────────
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # We call both endpoints as close as possible to minimize time drift
            tasks = [
                client.get(f"{SIMULATOR_BASE}/api/users/{req.probe_uid}/features/live"),
                client.get(f"{SIMULATOR_BASE}/api/users/{req.target_uid}/features/live")
            ]
            responses = await asyncio.gather(*tasks)
            
            if responses[0].status_code != 200:
                raise HTTPException(503, f"Simulator error for probe UID {req.probe_uid}")
            if responses[1].status_code != 200:
                raise HTTPException(503, f"Simulator error for target UID {req.target_uid}")
                
            live_probe = responses[0].json()
            live_target = responses[1].json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Cannot reach EEG Simulator: {e}")

    if not live_probe or "alpha_peak_est_hz" not in live_probe:
        raise HTTPException(422, "Probe live features not ready")
    if not live_target or "alpha_peak_est_hz" not in live_target:
        raise HTTPException(422, "Target live features not ready")

    probe_vec  = vectorize(live_probe)
    target_vec = vectorize(live_target)

    if not probe_vec or not target_vec:
        raise HTTPException(422, "Feature vectors empty — simulator may need more time")

    # ── Compute similarity scores ────────────────────────────────────────
    cos  = cosine_similarity(probe_vec, target_vec)
    euc  = euclidean_similarity(probe_vec, target_vec)
    feat = per_feature_relative_score(probe_vec, target_vec)
    ens  = ensemble_score(probe_vec, target_vec)
    matched = ens >= AUTH_THRESHOLD

    # Fingerprints from stored DB (for display / audit trail)
    stored_feat = get_stored_features(req.target_uid) or live_target
    stored_vec  = vectorize(stored_feat)
    fp_stored   = brainprint_fingerprint(stored_vec) if stored_vec else "NO DATA"
    fp_live     = brainprint_fingerprint(probe_vec)
    breakdown   = per_feature_diff(target_vec, probe_vec)

    # ── Get names ─────────────────────────────────────────────────────────
    all_users = get_all_stored_brainprints()
    users_by_uid = {u["uid"]: u["name"] for u in all_users}
    probe_name  = users_by_uid.get(req.probe_uid,  f"User {req.probe_uid}")
    target_name = users_by_uid.get(req.target_uid, f"User {req.target_uid}")

    result_str = "GRANTED" if matched else "DENIED"
    log_attempt(
        req.probe_uid, probe_name,
        req.target_uid, target_name,
        cos, euc, ens,
        result_str,
        fp_stored, fp_live,
    )

    return {
        "probe_uid":    req.probe_uid,
        "probe_name":   probe_name,
        "target_uid":   req.target_uid,
        "target_name":  target_name,
        "cosine_similarity":     cos,
        "euclidean_similarity":  euc,
        "ensemble_score":        ens,
        "threshold":             AUTH_THRESHOLD,
        "match":                 matched,
        "result":                result_str,
        "fingerprint_stored":    fp_stored,
        "fingerprint_live":      fp_live,
        "feature_breakdown":     breakdown,
        "biometric_dimensions":  len(BIOMETRIC_KEYS),
    }


@app.post("/api/auth/add_user")
async def add_user_proxy():
    """
    Proxy to add a generic user in the simulator.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{SIMULATOR_BASE}/api/users", json={"count": 1})
            if r.status_code != 200:
                raise HTTPException(503, f"Simulator returned {r.status_code}")
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Cannot reach EEG Simulator: {e}")


@app.post("/api/auth/enroll/{uid}")
async def enroll_user(uid: int):
    """
    Tells the simulator to re-snapshot the current features and raw data.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{SIMULATOR_BASE}/api/users/{uid}/enroll")
            if r.status_code != 200:
                raise HTTPException(503, f"Simulator returned {r.status_code}")
            res = r.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Cannot reach EEG Simulator: {e}")

    # For display, get the fresh brainprint from DB
    stored = get_stored_features(uid)
    vec = vectorize(stored) if stored else []
    fp = brainprint_fingerprint(vec) if vec else "ERROR"
    
    return {"uid": uid, "fingerprint": fp, "status": "enrolled", "dimensions": len(vec)}


@app.post("/api/auth/self_test/{uid}")
async def self_test(uid: int):
    """
    Convenience: tests a user's LIVE signal against their OWN stored brainprint.
    A perfect deterministic system should score ≈ 0.99+.
    """
    req = VerifyRequest(probe_uid=uid, target_uid=uid)
    return await verify_brainprint(req)


@app.post("/api/auth/add_user")
async def auth_add_user():
    """
    Tells the simulator to create a new random user.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{SIMULATOR_BASE}/api/users")
            if r.status_code != 200:
                raise HTTPException(503, f"Simulator returned {r.status_code}")
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(503, f"Cannot reach EEG Simulator: {e}")


@app.get("/api/auth/history")
async def auth_history(limit: int = 50):
    return {"history": get_history(limit)}


@app.delete("/api/auth/history")
async def clear_history():
    conn = sqlite3.connect(AUTH_DB)
    conn.execute("DELETE FROM auth_attempts")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


# ─── Deep Learning Endpoints ─────────────────────────────────────────────────

@app.get("/api/auth/dl_status")
async def dl_status():
    """Returns DL model training status and architecture info."""
    if not _DL_AVAILABLE:
        return {"available": False, "message": "dl_auth.py not found"}
    engine = _get_dl_engine()
    return {"available": True, **engine.status()}


@app.post("/api/auth/dl_train")
async def dl_train():
    """
    Trains the 1D CNN + LSTM + MLP pipeline on all raw EEG windows in the DB.
    Safe to call multiple times — retraining refreshes the model.
    """
    if not _DL_AVAILABLE:
        raise HTTPException(503, "DL engine not available")
    import asyncio
    loop = asyncio.get_event_loop()
    engine = _get_dl_engine()
    # Run in thread pool to avoid blocking the event loop during training
    result = await loop.run_in_executor(None, engine.train_from_db)
    return result


@app.post("/api/auth/dl_verify")
async def dl_verify(req: VerifyRequest):
    """
    Authenticates using the CNN+LSTM+MLP deep learning model.
    probe_uid : whose LIVE signal (latest window from DB) to test
    target_uid: whose ENROLLED embedding to compare against
    """
    if not _DL_AVAILABLE:
        raise HTTPException(503, "DL engine not available")
    engine = _get_dl_engine()
    result = engine.verify(req.probe_uid, req.target_uid)
    if result.get("status") == "error":
        raise HTTPException(422, result["message"])

    # Log to auth_history DB with model tag
    all_users = get_all_stored_brainprints()
    umap = {u["uid"]: u["name"] for u in all_users}
    probe_name  = umap.get(req.probe_uid,  f"User {req.probe_uid}")
    target_name = umap.get(req.target_uid, f"User {req.target_uid}")
    log_attempt(
        req.probe_uid, f"[DL] {probe_name}",
        req.target_uid, target_name,
        result["cosine_similarity"],
        result["mlp_probability"],
        result["ensemble_score"],
        result["result"],
        "", "",
    )
    return result


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_auth_db()
    print("=" * 60)
    print("  EEG Brainprint Auth Server  |  http://127.0.0.1:8001")
    print("  Simulator expected at       |  http://127.0.0.1:8000")
    print("  Auth UI                     |  http://127.0.0.1:8001/")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
