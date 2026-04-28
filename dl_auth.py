"""
EEG Deep Learning Authentication Engine
========================================
Pure numpy 1D CNN + LSTM feature extractor, scikit-learn MLP classifier.
No PyTorch/TensorFlow required.

Architecture
------------
Input : raw EEG window  (21 channels × 256 samples = 2 s at 256 Hz)
  ↓  Conv1D  (32 filters, kernel=8, stride=1) → ReLU
  ↓  MaxPool1D (size=4)
  ↓  Conv1D  (64 filters, kernel=4, stride=1) → ReLU
  ↓  MaxPool1D (size=4)
  ↓  LSTM cell over time steps (hidden=64)
  ↓  L2-normalised output vector  (64-dim embedding)
  → sklearn MLPClassifier maps embedding → user class

Authentication (Siamese style)
-------------------------------
  stored_embedding   ← trained from enrolled windows
  live_embedding     ← extracted from current 2s live window
  score              ← cosine similarity in embedding space
  verdict            ← score ≥ threshold

Usage
-----
  engine = DLAuthEngine()
  engine.train_from_db()          # builds dataset, trains, saves model
  engine.verify(uid_probe, uid_target)  → dict
"""

import json
import math
import os
import pickle
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match simulator_core.py)
# ──────────────────────────────────────────────────────────────────────────────
FS           = 256
N_CHANNELS   = 19               # scalp electrodes (excludes A1, A2)
WINDOW_SEC   = 2                # seconds per training window
WINDOW_SAMP  = WINDOW_SEC * FS  # 512 samples
STEP_SAMP    = WINDOW_SAMP // 2 # 50% overlap → more training samples
EMBED_DIM    = 64               # LSTM hidden size / embedding dimension
DB_FILE      = "eeg_auth.db"
MODEL_FILE   = "dl_auth_model.pkl"

SCALP_ELEC = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
               "T3","C3","Cz","C4","T4",
               "T5","P3","Pz","P4","T6","O1","O2"]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Signal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

# Exact same biometric keys used by the simulator
BIOMETRIC_KEYS = [
    "alpha_peak_est_hz", "rms_uv", "variance_uv2", "line_length", "zero_cross_rate",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "spectral_entropy",
    "alpha_psd_peak", "alpha_psd_cog", "alpha_psd_entropy",
    "alpha_sb1_rel", "alpha_sb2_rel", "alpha_sb3_rel", "alpha_sb4_rel",
    "theta_alpha_ratio", "beta_alpha_ratio", "gamma_theta_ratio", "delta_gamma_ratio",
    "total_spectral_power", "estimated_interhemispheric_coherence",
    "posterior_alpha_asym", "frontal_beta_asym",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "delta_rel", "theta_rel", "alpha_rel", "beta_rel", "gamma_rel"
]

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ──────────────────────────────────────────────────────────────────────────────
# 2.  1-D Convolutional Layer (fast vectorised via stride tricks)
# ──────────────────────────────────────────────────────────────────────────────

class Conv1D:
    """
    Input  : (C_in, T)
    Weights: (C_out, C_in, kernel)
    Output : (C_out, T - kernel + 1)
    """
    def __init__(self, c_in: int, c_out: int, kernel: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (c_in * kernel))          # He init
        self.W = rng.standard_normal((c_out, c_in, kernel)).astype(np.float32) * scale
        self.b = np.zeros(c_out, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (C_in, T)
        C_in, T = x.shape
        C_out, _, K = self.W.shape
        T_out = T - K + 1

        # sliding_window_view: (C_in, T_out, K)
        windows = sliding_window_view(x, K, axis=1)        # (C_in, T_out, K)
        windows = windows.transpose(1, 0, 2)               # (T_out, C_in, K)
        flat    = windows.reshape(T_out, -1)               # (T_out, C_in*K)
        W_flat  = self.W.reshape(C_out, -1)                # (C_out, C_in*K)
        out     = (flat @ W_flat.T) + self.b               # (T_out, C_out)
        return out.T                                       # (C_out, T_out)


def maxpool1d(x: np.ndarray, size: int) -> np.ndarray:
    """Max-pool along last axis; drops tail samples if needed."""
    C, T = x.shape
    T_crop = (T // size) * size
    return x[:, :T_crop].reshape(C, T_crop // size, size).max(axis=2)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  LSTM Cell (numpy)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMEncoder:
    """
    Processes a sequence (T_steps, input_dim) and returns the final hidden state.
    """
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 1):
        rng   = np.random.default_rng(seed)
        scale = math.sqrt(1.0 / hidden_dim)
        D, H  = input_dim, hidden_dim

        def _W():
            return rng.standard_normal((H, D + H)).astype(np.float32) * scale
        def _b():
            return np.zeros(H, dtype=np.float32)

        # forget, input, gate, output
        self.Wf, self.bf = _W(), _b()
        self.Wi, self.bi = _W(), _b()
        self.Wg, self.bg = _W(), _b()
        self.Wo, self.bo = _W(), _b()
        self.hidden_dim  = H

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """x_seq: (T, input_dim) → returns h: (hidden_dim,)"""
        T, D = x_seq.shape
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        c = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T):
            xh = np.concatenate([x_seq[t], h])            # (D+H,)
            f  = _sigmoid(self.Wf @ xh + self.bf)
            i  = _sigmoid(self.Wi @ xh + self.bi)
            g  = _tanh   (self.Wg @ xh + self.bg)
            o  = _sigmoid(self.Wo @ xh + self.bo)
            c  = f * c + i * g
            h  = o * _tanh(c)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Full CNN + LSTM Encoder
# ──────────────────────────────────────────────────────────────────────────────

class EEGEncoder:
    """
    CNN + LSTM encoder that maps a raw EEG window to a 64-dim embedding.

    Input shape : (N_CHANNELS, WINDOW_SAMP)  →  (19, 512)
    CNN block 1 : Conv1D(19→32, k=8) + ReLU + MaxPool(4)   → (32, 126)
    CNN block 2 : Conv1D(32→64, k=4) + ReLU + MaxPool(4)   → (64, 30)
    LSTM        : input_dim=64, hidden=64, seq_len=30       → (64,)
    L2 norm     : embedding in unit hypersphere             → (64,)
    """
    def __init__(self):
        self.conv1  = Conv1D(N_CHANNELS, 32, kernel=8, seed=42)
        self.conv2  = Conv1D(32,         64, kernel=4, seed=43)
        self.lstm   = LSTMEncoder(input_dim=64, hidden_dim=EMBED_DIM, seed=44)

    def encode(self, window: np.ndarray) -> np.ndarray:
        """window: (N_CHANNELS, WINDOW_SAMP) → embedding: (EMBED_DIM,)"""
        x = window.astype(np.float32)
        # Normalise each channel to zero-mean, unit-std
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - m) / s

        # CNN
        x = _relu(self.conv1.forward(x))   # (32, T1)
        x = maxpool1d(x, 4)               # (32, T1//4)
        x = _relu(self.conv2.forward(x))   # (64, T2)
        x = maxpool1d(x, 4)               # (64, T2//4)

        # LSTM: treat (64, T) → (T, 64) sequence
        x_seq = x.T                        # (T, 64)
        h     = self.lstm.forward(x_seq)   # (64,)
        return _l2_norm(h)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Data pipeline: load raw EEG from DB, slice into windows
# ──────────────────────────────────────────────────────────────────────────────

def load_hybrid_data_from_db() -> Dict[int, Dict[str, Any]]:
    """
    Returns {uid: {"windows": np.ndarray, "handcrafted": np.ndarray}}
    """
    if not os.path.exists(DB_FILE):
        return {}

    conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    c = conn.cursor()
    c.execute("SELECT user_id, raw_data_json, features_json FROM brainprints")
    rows = c.fetchall()
    conn.close()

    result = {}
    for uid, raw_json, feat_json in rows:
        if not raw_json or not feat_json:
            continue
            
        raw = json.loads(raw_json)
        feat = json.loads(feat_json)
        
        # 1. Raw Windows
        channels_data = raw.get("channels", {})
        arrays = []
        for elec in SCALP_ELEC:
            if elec in channels_data:
                arrays.append(channels_data[elec])
        if len(arrays) < N_CHANNELS:
            continue

        mat = np.array(arrays, dtype=np.float32)
        total = mat.shape[1]
        if total < WINDOW_SAMP:
            continue

        windows = []
        start = 0
        while start + WINDOW_SAMP <= total:
            windows.append(mat[:, start: start + WINDOW_SAMP])
            start += STEP_SAMP
        
        # 2. Handcrafted Vector
        h_vec = []
        for k in BIOMETRIC_KEYS:
            h_vec.append(float(feat.get(k, 0.0)))
            
        result[uid] = {
            "windows": np.stack(windows),
            "handcrafted": np.array(h_vec, dtype=np.float32)
        }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6.  DL Authentication Engine
# ──────────────────────────────────────────────────────────────────────────────

class DLAuthEngine:
    """
    Full pipeline:
      train_from_db()  →  encodes all windows, trains sklearn MLP, saves
      verify(probe_uid, target_uid)  →  authentication result dict
      enroll(uid, raw_window)  →  updates stored embedding
    """

    def __init__(self):
        self.encoder         : EEGEncoder             = EEGEncoder()
        self.classifier      : Optional[Any]          = None   # sklearn MLP
        self.uid_to_label    : Dict[int, int]         = {}
        self.label_to_uid    : Dict[int, int]         = {}
        self.stored_embeddings: Dict[int, np.ndarray] = {}     # uid → mean embedding
        self.trained         : bool                   = False
        self.train_meta      : Dict[str, Any]         = {}
        self._try_load()

    def _try_load(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    state = pickle.load(f)
                self.classifier        = state["classifier"]
                self.uid_to_label      = state["uid_to_label"]
                self.label_to_uid      = state["label_to_uid"]
                self.stored_embeddings = state["stored_embeddings"]
                self.train_meta        = state.get("train_meta", {})
                self.trained           = True
                print(f"[DL Auth] Loaded model for {len(self.uid_to_label)} users")
            except Exception as e:
                print(f"[DL Auth] Could not load model: {e}")

    def _save(self):
        state = {
            "classifier":        self.classifier,
            "uid_to_label":      self.uid_to_label,
            "label_to_uid":      self.label_to_uid,
            "stored_embeddings": self.stored_embeddings,
            "train_meta":        self.train_meta,
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(state, f)

    def train_from_db(self) -> Dict[str, Any]:
        """
        Hybrid training pipeline:
         1. Load raw windows AND handcrafted features from DB.
         2. Encode windows via CNN+LSTM → Embeddings (64-dim).
         3. Concatenate handcrafted features (34-dim) → Hybrid Features (98-dim).
         4. Train scikit-learn MLP on the hybrid vectors.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        t0 = time.time()
        all_data = load_hybrid_data_from_db()

        if len(all_data) < 2:
            return {
                "status": "error",
                "message": f"Need at least 2 users with data. Found {len(all_data)}."
            }

        uids = sorted(all_data.keys())
        self.uid_to_label = {uid: i for i, uid in enumerate(uids)}
        self.label_to_uid = {i: uid for uid, i in self.uid_to_label.items()}

        X_list, y_list = [], []
        self.stored_embeddings = {}
        window_counts = {}
        print(f"[DL Auth] Encoding windows for {len(uids)} users (Hybrid Mode)...")

        for uid, data in all_data.items():
            label = self.uid_to_label[uid]
            windows = data["windows"]
            h_feat = data["handcrafted"]
            
            user_embeddings = []
            for w in windows:
                emb = self.encoder.encode(w)
                # Concatenate DL embedding + Handcrafted features
                hybrid_vec = np.concatenate([emb, h_feat])
                X_list.append(hybrid_vec)
                y_list.append(label)
                user_embeddings.append(emb)
            
            self.stored_embeddings[uid] = _l2_norm(np.mean(user_embeddings, axis=0))
            window_counts[uid] = len(windows)
            print(f"  UID {uid}: {len(windows)} windows -> hybrid_vec {hybrid_vec.shape}")

        X = np.stack(X_list)
        y = np.array(y_list)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
        )
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp",    mlp),
        ])
        pipe.fit(X, y)
        self.classifier = pipe
        self.trained    = True

        elapsed = time.time() - t0
        n_iter  = mlp.n_iter_
        score   = pipe.score(X, y)

        self.train_meta = {
            "users":           uids,
            "window_counts":   window_counts,
            "total_windows":   len(X_list),
            "embed_dim":       EMBED_DIM,
            "mlp_layers":      (128, 64),
            "train_accuracy":  round(float(score), 4),
            "n_iterations":    int(n_iter),
            "elapsed_sec":     round(elapsed, 2),
            "trained_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_file":      MODEL_FILE,
        }
        self._save()
        print(f"[DL Auth] Training complete. Accuracy={score:.3f} in {elapsed:.1f}s")
        return {"status": "ok", **self.train_meta}

    def _get_live_hybrid(self, uid: int) -> Optional[np.ndarray]:
        if not os.path.exists(DB_FILE):
            return None
        conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        c = conn.cursor()
        c.execute("SELECT raw_data_json, features_json FROM brainprints WHERE user_id=?", (uid,))
        row = c.fetchone()
        conn.close()
        if not row or not row[0] or not row[1]:
            return None

        raw = json.loads(row[0])
        feat = json.loads(row[1])
        
        # 1. DL Embedding
        channels_data = raw.get("channels", {})
        arrays = []
        for elec in SCALP_ELEC:
            if elec in channels_data:
                arrays.append(channels_data[elec])
        if len(arrays) < N_CHANNELS:
            return None
        mat = np.array(arrays, dtype=np.float32)
        if mat.shape[1] < WINDOW_SAMP:
            return None
        window = mat[:, -WINDOW_SAMP:]
        emb = self.encoder.encode(window)
        
        # 2. Handcrafted vector
        h_vec = []
        for k in BIOMETRIC_KEYS:
            h_vec.append(float(feat.get(k, 0.0)))
            
        return np.concatenate([emb, np.array(h_vec, dtype=np.float32)])

    def verify(self, probe_uid: int, target_uid: int) -> Dict[str, Any]:
        """
        Hybrid Authentication:
         - uses CNN+LSTM embedding for Siamese scoring
         - uses Concatenated (DL + Handcrafted) vector for MLP scoring
        """
        if not self.trained or self.classifier is None:
            return {
                "status": "error",
                "message": "Model not trained. Call /api/auth/dl_train first."
            }

        if target_uid not in self.stored_embeddings:
            return {
                "status": "error",
                "message": f"No enrolled embedding for target UID {target_uid}."
            }
        target_emb = self.stored_embeddings[target_uid]

        # Get probe hybrid features
        probe_hybrid = self._get_live_hybrid(probe_uid)
        if probe_hybrid is None:
            return {
                "status": "error",
                "message": f"Cannot extract hybrid features for probe UID {probe_uid}."
            }
        
        # Extract individual components for ensemble
        probe_emb = probe_hybrid[:EMBED_DIM]

        # Cosine similarity in embedding space
        cos_score = _cosine_sim(probe_emb, target_emb)
        cos_score_01 = float((cos_score + 1.0) / 2.0)

        # MLP probability on hybrid features
        mlp_prob = 0.0
        if target_uid in self.uid_to_label:
            label = self.uid_to_label[target_uid]
            probs = self.classifier.predict_proba([probe_hybrid])[0]
            if label < len(probs):
                mlp_prob = float(probs[label])

        # Ensemble: 60% cosine (invariant shape) + 40% MLP probability
        ensemble = round(0.60 * cos_score_01 + 0.40 * mlp_prob, 6)

        # Threshold: 0.65 (tuned for deterministic simulator signals)
        threshold = 0.65
        matched   = ensemble >= threshold

        return {
            "status":            "ok",
            "probe_uid":         probe_uid,
            "target_uid":        target_uid,
            "cosine_similarity": round(cos_score_01, 6),
            "mlp_probability":   round(mlp_prob, 6),
            "ensemble_score":    ensemble,
            "threshold":         threshold,
            "match":             matched,
            "result":            "GRANTED" if matched else "DENIED",
            "embed_dim":         EMBED_DIM,
            "model":             "CNN+LSTM+MLP",
        }

    def status(self) -> Dict[str, Any]:
        return {
            "trained":           self.trained,
            "n_users_enrolled":  len(self.stored_embeddings),
            "enrolled_uids":     list(self.stored_embeddings.keys()),
            "feature_dim":       EMBED_DIM + len(BIOMETRIC_KEYS),
            "architecture":      "CNN+LSTM(64) + Handcrafted(34) -> MLP(128,64)",
            **self.train_meta,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Singleton instance (imported by auth_server.py)
# ──────────────────────────────────────────────────────────────────────────────

_engine: Optional[DLAuthEngine] = None

def get_engine() -> DLAuthEngine:
    global _engine
    if _engine is None:
        _engine = DLAuthEngine()
    return _engine


# ──────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EEG Deep Learning Auth Engine — Standalone Test")
    print("=" * 60)

    engine = DLAuthEngine()

    print("\n[1] Training...")
    result = engine.train_from_db()
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "ok":
        uids = result["users"]
        print(f"\n[2] Self-test: UID {uids[0]} vs itself")
        r = engine.verify(uids[0], uids[0])
        print(json.dumps(r, indent=2))

        if len(uids) > 1:
            print(f"\n[3] Imposter test: UID {uids[0]} vs UID {uids[1]}")
            r2 = engine.verify(uids[0], uids[1])
            print(json.dumps(r2, indent=2))
