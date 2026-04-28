import asyncio
import time
import csv
import io
import math
import sqlite3
import json
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import os

from simulator_core import (
    UserProfile, EEGUserSession, FS, ELECTRODES, SCALP_ELECTRODES, MONTAGES, FEATURE_BANDS,
    highpass, lowpass, band_powers, peak_frequency, rms, variance, line_length,
    zero_crossing_rate, hjorth, spectral_entropy, safe_log_ratio, goertzel_power
)
import numpy as np
from scipy import signal as sig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Multi-User
sessions: Dict[int, EEGUserSession] = {}
active_session_id: int = 1
user_counter = 0

artifacts_config = {
    "blink": True,
    "muscle": True,
    "line": True,
    "pop": True,
}
line_freq = 50.0
active_websockets = set()
is_running = True

DB_FILE = "eeg_auth.db"

def generate_seed_from_traits(name: str, age: int, pathology: str, medication: str) -> int:
    s = f"{name.lower().strip()}|{age}|{pathology}|{medication}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# Authentication Feature Extraction
def extract_auth_features(session: EEGUserSession, montage_name: str = "Longitudinal bipolar") -> Dict[str, Any]:
    window_samples = 5 * FS
    times = list(session.times)
    if len(times) < window_samples:
        return {}
    
    data = {e: list(session.buffers[e])[-window_samples:] for e in ELECTRODES}
    
    montage = MONTAGES.get(montage_name, MONTAGES["Longitudinal bipolar"])
    channel_label, pos, neg = montage[0]
    
    average = None
    if neg == "AVG":
        average = []
        for i in range(window_samples):
            average.append(sum(data[e][i] for e in SCALP_ELECTRODES) / len(SCALP_ELECTRODES))
    
    first = data[pos]
    if neg == "AVG":
        signal = [first[i] - average[i] for i in range(len(first))]
    elif neg == "A1A2":
        a1, a2 = data["A1"], data["A2"]
        signal = [first[i] - 0.5 * (a1[i] + a2[i]) for i in range(len(first))]
    elif neg == "NONE":
        signal = list(first)
    else:
        second = data[neg]
        signal = [first[i] - second[i] for i in range(len(first))]
        
    window = highpass(signal, 0.5, FS)
    window = lowpass(window, 45.0, FS)

    powers = band_powers(window, FS)
    total_power = sum(powers.values()) or 1.0
    activity, mobility, complexity = hjorth(window)
    alpha_power = powers.get("alpha", 1.0) or 1.0
    theta_power = powers.get("theta", 1.0) or 1.0
    gamma_power = powers.get("gamma", 1.0) or 1.0
    
    profile = session.profile
    
    def electrode_band(e, band):
        if e not in data: return 0.0
        w = highpass(data[e], 0.5, FS)
        w = lowpass(w, 45.0, FS)
        low, high = FEATURE_BANDS[band]
        return sum(goertzel_power(w, freq, FS) for freq in range(low, high))

    o1_alpha = electrode_band("O1", "alpha")
    o2_alpha = electrode_band("O2", "alpha")
    f3_beta = electrode_band("F3", "beta")
    f4_beta = electrode_band("F4", "beta")

    c3 = highpass(data["C3"], 0.5, FS)
    c4 = highpass(data["C4"], 0.5, FS)
    mean_c3 = sum(c3) / len(c3)
    mean_c4 = sum(c4) / len(c4)
    num = sum((c3[i] - mean_c3) * (c4[i] - mean_c4) for i in range(len(c3)))
    den = math.sqrt(sum((x - mean_c3)**2 for x in c3) * sum((y - mean_c4)**2 for y in c4))
    coherence_est = num / den if den > 0 else 0.0

    # ── Advanced Alpha Analysis (PSD + WPS Approx) ───────────────────────
    # We focus on the alpha band (8-13 Hz) as requested
    alpha_window = np.array(window)
    # 1. PSD via Welch
    f, psd = sig.welch(alpha_window, FS, nperseg=min(len(alpha_window), 256))
    alpha_idx = np.where((f >= 8) & (f <= 13))[0]
    
    if len(alpha_idx) > 0:
        f_alpha = f[alpha_idx]
        psd_alpha = psd[alpha_idx]
        
        # Alpha Peak Magnitude
        peak_idx = np.argmax(psd_alpha)
        alpha_psd_peak = psd_alpha[peak_idx]
        
        # Center of Gravity (Weighted average frequency)
        alpha_psd_cog = np.sum(f_alpha * psd_alpha) / np.sum(psd_alpha)
        
        # Band Entropy (specifically within alpha)
        p_alpha = psd_alpha / (np.sum(psd_alpha) + 1e-9)
        alpha_psd_entropy = -np.sum(p_alpha * np.log2(p_alpha + 1e-9)) / np.log2(len(p_alpha))
    else:
        alpha_psd_peak = 0.0
        alpha_psd_cog = 10.5
        alpha_psd_entropy = 0.0

    # 2. Sub-band energy (Wavelet Packet approximation)
    # Split 8-13 Hz into 4 sub-bands
    sub_bands = [(8.0, 9.25), (9.25, 10.5), (10.5, 11.75), (11.75, 13.0)]
    sb_powers = []
    for low_b, high_b in sub_bands:
        sb_idx = np.where((f >= low_b) & (f <= high_b))[0]
        sb_powers.append(np.sum(psd[sb_idx]) if len(sb_idx) > 0 else 0.0)
    
    sb_total = sum(sb_powers) + 1e-9
    
    features = {
        "user_id": profile.user_id,
        "user_name": profile.name,
        "seed": profile.seed,
        "sim_time_s": round(session.sim_time, 3),
        "channel": channel_label,
        "age": profile.age,
        "alertness": round(profile.alertness, 3),
        "state": profile.eeg_state,
        "pathology": profile.pathology,
        "medication": profile.medication,
        "alpha_peak_personal_hz": round(profile.alpha_peak, 3),
        "alpha_peak_est_hz": peak_frequency(window, 7, 14, FS),
        "rms_uv": round(rms(window), 5),
        "variance_uv2": round(variance(window), 5),
        "line_length": round(line_length(window), 5),
        "zero_cross_rate": round(zero_crossing_rate(window), 5),
        "hjorth_activity": round(activity, 5),
        "hjorth_mobility": round(mobility, 5),
        "hjorth_complexity": round(complexity, 5),
        "spectral_entropy": round(spectral_entropy(window, 1, 45, FS), 5),
        
        # Advanced Alpha Metrics
        "alpha_psd_peak": round(float(alpha_psd_peak), 5),
        "alpha_psd_cog": round(float(alpha_psd_cog), 5),
        "alpha_psd_entropy": round(float(alpha_psd_entropy), 5),
        "alpha_sb1_rel": round(float(sb_powers[0]/sb_total), 5),
        "alpha_sb2_rel": round(float(sb_powers[1]/sb_total), 5),
        "alpha_sb3_rel": round(float(sb_powers[2]/sb_total), 5),
        "alpha_sb4_rel": round(float(sb_powers[3]/sb_total), 5),
        
        "theta_alpha_ratio": round(theta_power / alpha_power, 5),
        "beta_alpha_ratio": round(powers.get("beta", 0) / alpha_power, 5),
        "gamma_theta_ratio": round(gamma_power / theta_power, 5),
        "delta_gamma_ratio": round(powers.get("delta", 0) / gamma_power, 5),
        "total_spectral_power": round(total_power, 5),
        "estimated_interhemispheric_coherence": round(coherence_est, 5),
        "posterior_alpha_asym": round(safe_log_ratio(o2_alpha, o1_alpha), 5),
        "frontal_beta_asym": round(safe_log_ratio(f4_beta, f3_beta), 5),
    }
    for band, power in powers.items():
        features[f"{band}_power"] = round(power, 5)
        features[f"{band}_rel"] = round(power / total_power, 5)
    
    return features

def extract_raw_data(session: EEGUserSession, window_seconds: int = 10) -> Dict[str, Any]:
    window_samples = window_seconds * FS
    times = list(session.times)
    if len(times) < window_samples:
        return {}
    
    raw_data = {
        "times": list(session.times)[-window_samples:],
        "channels": {e: list(session.buffers[e])[-window_samples:] for e in ELECTRODES}
    }
    return raw_data


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (uid INTEGER PRIMARY KEY, name TEXT, age INTEGER, state TEXT, 
                  sleep_stage TEXT, pathology TEXT, medication TEXT, alertness REAL, seed INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS brainprints
                 (user_id INTEGER PRIMARY KEY, features_json TEXT, raw_data_json TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(uid) ON DELETE CASCADE)''')
    
    try:
        c.execute("ALTER TABLE brainprints ADD COLUMN raw_data_json TEXT")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    
    # Load existing
    c.execute("SELECT * FROM users")
    rows = c.fetchall()
    global user_counter, active_session_id
    if rows:
        for r in rows:
            uid, name, age, state, sleep_stage, pathology, medication, alertness, seed = r
            profile = UserProfile.random_profile(uid, name=name, seed=seed)
            profile.age = age
            profile.eeg_state = state
            profile.sleep_stage = sleep_stage
            profile.pathology = pathology
            profile.medication = medication
            profile.alertness = alertness
            sess = EEGUserSession(profile)
            sess.prefill_history(10.0, artifacts_config, line_freq)
            sessions[uid] = sess
            user_counter = max(user_counter, uid)
        active_session_id = next(iter(sessions.keys()))
    conn.close()

def save_user_to_db(uid: int):
    sess = sessions[uid]
    p = sess.profile
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO users 
                 (uid, name, age, state, sleep_stage, pathology, medication, alertness, seed) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (uid, p.name, p.age, p.eeg_state, p.sleep_stage, p.pathology, p.medication, p.alertness, p.seed))
    
    feat = extract_auth_features(sess)
    raw_data = extract_raw_data(sess)
    
    if feat:
        c.execute('''INSERT OR REPLACE INTO brainprints (user_id, features_json, raw_data_json) VALUES (?, ?, ?)''',
                  (uid, json.dumps(feat), json.dumps(raw_data) if raw_data else None))
    conn.commit()
    conn.close()

def remove_user_from_db(uid: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM brainprints WHERE user_id=?", (uid,))
    c.execute("DELETE FROM users WHERE uid=?", (uid,))
    conn.commit()
    conn.close()

init_db()

def create_user(name: str = None):
    global user_counter, active_session_id
    user_counter += 1
    profile = UserProfile.random_profile(user_id=user_counter, name=name)
    session = EEGUserSession(profile)
    session.prefill_history(10.0, artifacts_config, line_freq)
    sessions[user_counter] = session
    if len(sessions) == 1:
        active_session_id = user_counter
    save_user_to_db(user_counter)
    return user_counter

if not sessions:
    create_user("Synthetic Patient")

class EventModel(BaseModel):
    label: str

class ProfileUpdateModel(BaseModel):
    name: str = None
    age: int = None
    state: str = None
    sleep_stage: str = None
    pathology: str = None
    medication: str = None
    alertness: float = None

class AddUserModel(BaseModel):
    count: int = 1

class CustomUserModel(BaseModel):
    name: str
    age: int
    pathology: str
    medication: str

class ControlModel(BaseModel):
    running: bool


@app.get("/api/db/stats")
async def get_db_stats():
    user_count = len(sessions)
    try:
        size_bytes = os.path.getsize(DB_FILE)
        size_mb = size_bytes / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    return {"user_count": user_count, "db_size_mb": round(size_mb, 2)}

@app.get("/api/config")
async def get_config():
    if active_session_id not in sessions:
        return {}
    profile = sessions[active_session_id].profile
    return {
        "active_user_id": active_session_id,
        "profile": {
            "name": profile.name,
            "age": profile.age,
            "state": profile.eeg_state,
            "sleep_stage": profile.sleep_stage,
            "alertness": profile.alertness,
            "pathology": profile.pathology,
            "medication": profile.medication,
        },
        "artifacts": artifacts_config,
        "line_freq": line_freq,
        "electrodes": ELECTRODES,
        "fs": FS,
        "is_running": is_running,
    }

@app.get("/api/users")
async def list_users():
    res = []
    for uid, sess in sessions.items():
        res.append({
            "id": uid,
            "name": sess.profile.name,
            "age": sess.profile.age,
            "state": sess.profile.eeg_state,
            "signature": sess.profile.signature_summary()
        })
    return {"users": res, "active": active_session_id}

@app.post("/api/users")
async def add_user(data: AddUserModel = None):
    count = data.count if data else 1
    last_uid = None
    for _ in range(count):
        last_uid = create_user(f"User {user_counter + 1}")
    global active_session_id
    if last_uid is not None:
        active_session_id = last_uid
    return {"id": last_uid, "added": count}

@app.post("/api/users/custom")
async def add_custom_user(data: CustomUserModel):
    global user_counter, active_session_id
    user_counter += 1
    seed = generate_seed_from_traits(data.name, data.age, data.pathology, data.medication)
    profile = UserProfile.random_profile(user_counter, name=data.name, seed=seed)
    profile.age = data.age
    profile.pathology = data.pathology
    profile.medication = data.medication
    # Force state to be awake for custom generation baseline
    profile.eeg_state = "Awake - eyes closed"
    
    session = EEGUserSession(profile)
    session.prefill_history(10.0, artifacts_config, line_freq)
    sessions[user_counter] = session
    active_session_id = user_counter
    save_user_to_db(user_counter)
    return {"id": user_counter}

@app.post("/api/control")
async def set_control(data: ControlModel):
    global is_running
    is_running = data.running
    return {"is_running": is_running}


@app.delete("/api/users/{uid}")
async def remove_user(uid: int):
    global active_session_id
    if uid in sessions and len(sessions) > 1:
        del sessions[uid]
        remove_user_from_db(uid)
        if active_session_id == uid:
            active_session_id = next(iter(sessions.keys()))
        return {"status": "ok"}
    return {"status": "error", "message": "Cannot remove last user"}

@app.put("/api/users/active")
async def set_active_user(data: Dict[str, int]):
    global active_session_id
    uid = data.get("id")
    if uid in sessions:
        active_session_id = uid
    return {"active": active_session_id}

@app.put("/api/users/{uid}/profile")
async def update_profile(uid: int, data: ProfileUpdateModel):
    if uid in sessions:
        profile = sessions[uid].profile
        if data.name is not None: profile.name = data.name
        if data.age is not None: profile.age = data.age
        if data.state is not None: profile.eeg_state = data.state
        if data.sleep_stage is not None: profile.sleep_stage = data.sleep_stage
        if data.pathology is not None: profile.pathology = data.pathology
        if data.medication is not None: profile.medication = data.medication
        if data.alertness is not None: profile.alertness = data.alertness
        sessions[uid].generator.set_profile(profile)
        save_user_to_db(uid)
    return {"status": "ok"}

@app.post("/api/users/{uid}/randomize")
async def randomize_user(uid: int):
    if uid in sessions:
        sess = sessions[uid]
        old = sess.profile
        new_seed = 20260426 + old.user_id * 7919 + int(time.time() * 1000) % 900000
        profile = UserProfile.random_profile(old.user_id, name=old.name, seed=new_seed)
        profile.eeg_state = old.eeg_state
        profile.sleep_stage = old.sleep_stage
        profile.pathology = old.pathology
        profile.medication = old.medication
        profile.alertness = old.alertness
        sess.replace_profile(profile, artifacts_config, line_freq)
        save_user_to_db(uid)
    return {"status": "ok"}

@app.post("/api/event")
async def add_event(data: EventModel):
    if active_session_id in sessions:
        sess = sessions[active_session_id]
        sess.events.append((sess.sim_time, data.label))
    return {"status": "ok"}

# ── READ-ONLY: used by auth_server.py — does NOT touch simulation state ────────
@app.get("/api/users/{uid}/features/live")
async def get_live_features(uid: int):
    if uid not in sessions:
        from fastapi import HTTPException
        raise HTTPException(404, f"User {uid} not found")
    feat = extract_auth_features(sessions[uid])
    return feat

@app.post("/api/users/{uid}/enroll")
async def enroll_user_simulator(uid: int):
    """
    Snapshots both features and raw data into the DB for a specific user.
    """
    if uid not in sessions:
        from fastapi import HTTPException
        raise HTTPException(404, f"User {uid} not found")
    save_user_to_db(uid)
    return {"status": "ok", "uid": uid}

@app.get("/api/export/raw")
async def export_raw(montage: str = "Longitudinal bipolar"):
    if active_session_id not in sessions:
        return {"status": "error"}
    sess = sessions[active_session_id]
    
    times = list(sess.times)[-10 * FS:]
    data = {e: list(sess.buffers[e])[-10 * FS:] for e in ELECTRODES}
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    m_info = MONTAGES.get(montage, MONTAGES["Longitudinal bipolar"])
    average = None
    if any(ch[2] == "AVG" for ch in m_info):
        average = []
        for i in range(len(times)):
            average.append(sum(data[e][i] for e in SCALP_ELECTRODES) / len(SCALP_ELECTRODES))
    
    signals = []
    for label, pos, neg in m_info:
        first = data[pos]
        if neg == "AVG":
            signal = [first[i] - average[i] for i in range(len(first))]
        elif neg == "A1A2":
            a1, a2 = data["A1"], data["A2"]
            signal = [first[i] - 0.5 * (a1[i] + a2[i]) for i in range(len(first))]
        elif neg == "NONE":
            signal = list(first)
        else:
            second = data[neg]
            signal = [first[i] - second[i] for i in range(len(first))]
        signals.append((label, signal))
        
    writer.writerow(["time_s"] + [lbl for lbl, _ in signals])
    for i, t in enumerate(times):
        writer.writerow([f"{t:.4f}"] + [f"{sig[i]:.3f}" for _, sig in signals])
        
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=raw_eeg_user_{active_session_id}.csv"})

@app.get("/api/export/features")
async def export_features():
    # Export from SQLite Database directly
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT features_json FROM brainprints")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {"status": "no data"}
        
    output = io.StringIO()
    writer = None
    
    for r in rows:
        feat = json.loads(r[0])
        if not writer:
            writer = csv.DictWriter(output, fieldnames=list(feat.keys()))
            writer.writeheader()
        writer.writerow(feat)
        
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=eeg_auth_features_db.csv"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

async def simulation_loop():
    # Update every 100ms (10Hz) to reduce browser load
    chunk_size = FS // 10
    chunk_interval = chunk_size / FS
    
    while True:
        start_time = time.perf_counter()
        
        if is_running:
            for sess in sessions.values():
                for _ in range(chunk_size):
                    sess.step(artifacts_config, line_freq)
        
        if active_websockets and active_session_id in sessions and is_running:
            sess = sessions[active_session_id]
            chunk_data = {"times": [], "samples": {e: [] for e in ELECTRODES}}
            
            times_list = list(sess.times)[-chunk_size:]
            chunk_data["times"] = times_list
            for e in ELECTRODES:
                chunk_data["samples"][e] = list(sess.buffers[e])[-chunk_size:]
                
            message = {
                "type": "eeg_chunk",
                "data": chunk_data,
                "events": [{"time": t, "label": l} for t, l in sess.events[-5:]]
            }
            dead_sockets = set()
            for ws in active_websockets:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_sockets.add(ws)
            active_websockets.difference_update(dead_sockets)
            
        elapsed = time.perf_counter() - start_time
        sleep_time = chunk_interval - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            await asyncio.sleep(0)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
