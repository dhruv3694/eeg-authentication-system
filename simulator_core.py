"""Multi-user real-time clinical EEG simulator.

This is an educational and research-prototyping simulator, not a medical
device. It imitates the screen layout and signal conventions used by clinical
EEG systems while also producing user-specific synthetic signals and live
feature rows that can later feed classification or authentication experiments.
"""

from __future__ import annotations

import bisect
import csv
import math
import random
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk


FS = 256
MAX_HISTORY_SECONDS = 45
FEATURE_HISTORY_ROWS = 7200


ELECTRODE_POS = {
    "Fp1": (-0.42, -0.94),
    "Fp2": (0.42, -0.94),
    "F7": (-0.88, -0.56),
    "F3": (-0.42, -0.50),
    "Fz": (0.00, -0.52),
    "F4": (0.42, -0.50),
    "F8": (0.88, -0.56),
    "T3": (-0.98, 0.00),
    "C3": (-0.45, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.45, 0.00),
    "T4": (0.98, 0.00),
    "T5": (-0.84, 0.58),
    "P3": (-0.42, 0.50),
    "Pz": (0.00, 0.54),
    "P4": (0.42, 0.50),
    "T6": (0.84, 0.58),
    "O1": (-0.35, 0.92),
    "O2": (0.35, 0.92),
    "A1": (-1.18, 0.05),
    "A2": (1.18, 0.05),
}

ELECTRODES = list(ELECTRODE_POS.keys())
SCALP_ELECTRODES = [name for name in ELECTRODES if name not in {"A1", "A2"}]


MONTAGES = {
    "Longitudinal bipolar": [
        ("Fp1-F7", "Fp1", "F7"),
        ("F7-T3", "F7", "T3"),
        ("T3-T5", "T3", "T5"),
        ("T5-O1", "T5", "O1"),
        ("Fp1-F3", "Fp1", "F3"),
        ("F3-C3", "F3", "C3"),
        ("C3-P3", "C3", "P3"),
        ("P3-O1", "P3", "O1"),
        ("Fp2-F8", "Fp2", "F8"),
        ("F8-T4", "F8", "T4"),
        ("T4-T6", "T4", "T6"),
        ("T6-O2", "T6", "O2"),
        ("Fp2-F4", "Fp2", "F4"),
        ("F4-C4", "F4", "C4"),
        ("C4-P4", "C4", "P4"),
        ("P4-O2", "P4", "O2"),
        ("Fz-Cz", "Fz", "Cz"),
        ("Cz-Pz", "Cz", "Pz"),
    ],
    "Transverse bipolar": [
        ("Fp1-Fp2", "Fp1", "Fp2"),
        ("F7-F3", "F7", "F3"),
        ("F3-Fz", "F3", "Fz"),
        ("Fz-F4", "Fz", "F4"),
        ("F4-F8", "F4", "F8"),
        ("T3-C3", "T3", "C3"),
        ("C3-Cz", "C3", "Cz"),
        ("Cz-C4", "Cz", "C4"),
        ("C4-T4", "C4", "T4"),
        ("T5-P3", "T5", "P3"),
        ("P3-Pz", "P3", "Pz"),
        ("Pz-P4", "Pz", "P4"),
        ("P4-T6", "P4", "T6"),
        ("O1-O2", "O1", "O2"),
    ],
    "Average reference": [(f"{e}-AVG", e, "AVG") for e in SCALP_ELECTRODES],
    "Linked ears reference": [(f"{e}-A1/A2", e, "A1A2") for e in SCALP_ELECTRODES],
    "Raw Electrodes": [(e, e, "NONE") for e in SCALP_ELECTRODES],
}


STATE_PROFILES = {
    "Awake - eyes closed": {
        "alpha": 42.0,
        "beta": 7.5,
        "theta": 5.5,
        "delta": 4.0,
        "gamma": 2.2,
        "noise": 1.8,
        "spindle": 0.0,
        "spike_wave": 0.0,
    },
    "Awake - eyes open": {
        "alpha": 14.0,
        "beta": 12.0,
        "theta": 5.0,
        "delta": 3.5,
        "gamma": 3.0,
        "noise": 2.0,
        "spindle": 0.0,
        "spike_wave": 0.0,
    },
    "Drowsy": {
        "alpha": 20.0,
        "beta": 5.0,
        "theta": 18.0,
        "delta": 9.0,
        "gamma": 1.8,
        "noise": 2.2,
        "spindle": 0.0,
        "spike_wave": 0.0,
    },
    "N2 sleep": {
        "alpha": 8.0,
        "beta": 3.5,
        "theta": 16.0,
        "delta": 24.0,
        "gamma": 1.2,
        "noise": 2.4,
        "spindle": 32.0,
        "spike_wave": 0.0,
    },
    "Generalized 3 Hz spike-wave": {
        "alpha": 6.0,
        "beta": 3.0,
        "theta": 7.0,
        "delta": 10.0,
        "gamma": 1.5,
        "noise": 2.4,
        "spindle": 0.0,
        "spike_wave": 88.0,
    },
}


SLEEP_STAGES = ["Wake", "Drowsy", "N1", "N2", "N3", "REM"]
PATHOLOGY_TENDENCIES = [
    "None",
    "Frontal slowing tendency",
    "Temporal sharp tendency",
    "Generalized spike-wave tendency",
    "Low voltage fast tendency",
]
MEDICATION_EFFECTS = ["None", "Sedative effect", "Stimulant effect", "Antiseizure effect"]

FEATURE_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp(t, 0.0, 1.0)
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def voltage_color(value: float, span: float = 120.0) -> str:
    t = clamp((value + span) / (2.0 * span), 0.0, 1.0)
    blue = (37, 99, 235)
    white = (248, 250, 252)
    red = (220, 38, 38)
    if t < 0.5:
        return rgb_to_hex(blend(blue, white, t * 2.0))
    return rgb_to_hex(blend(white, red, (t - 0.5) * 2.0))


def highpass(values: list[float], cutoff: float, fs: float) -> list[float]:
    if not values or cutoff <= 0:
        return values
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out: list[float] = []
    prev_y = 0.0
    prev_x = values[0]
    for x in values:
        y = alpha * (prev_y + x - prev_x)
        out.append(y)
        prev_y = y
        prev_x = x
    return out


def lowpass(values: list[float], cutoff: float, fs: float) -> list[float]:
    if not values or cutoff <= 0:
        return values
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / fs
    alpha = dt / (rc + dt)
    out: list[float] = []
    y = values[0]
    for x in values:
        y += alpha * (x - y)
        out.append(y)
    return out


def notch(values: list[float], freq: float, fs: float, q: float = 30.0) -> list[float]:
    if len(values) < 3 or freq <= 0 or freq >= fs / 2:
        return values
    w0 = 2.0 * math.pi * freq / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    out: list[float] = []
    x1 = x2 = y1 = y2 = 0.0
    for x0 in values:
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out.append(y0)
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def line_length(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[i] - values[i - 1]) for i in range(1, len(values))) / (len(values) - 1)


def zero_crossing_rate(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    crossings = 0
    prev = values[0] - avg
    for value in values[1:]:
        current = value - avg
        if (prev <= 0 < current) or (prev >= 0 > current):
            crossings += 1
        prev = current
    return crossings / (len(values) - 1)


def hjorth(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 4:
        return 0.0, 0.0, 0.0
    activity = variance(values)
    first = [values[i] - values[i - 1] for i in range(1, len(values))]
    second = [first[i] - first[i - 1] for i in range(1, len(first))]
    var_first = variance(first)
    var_second = variance(second)
    mobility = math.sqrt(var_first / activity) if activity > 0 else 0.0
    mobility_first = math.sqrt(var_second / var_first) if var_first > 0 else 0.0
    complexity = mobility_first / mobility if mobility > 0 else 0.0
    return activity, mobility, complexity


def goertzel_power(values: list[float], freq: float, fs: float) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    coeff = 2.0 * math.cos(2.0 * math.pi * freq / fs)
    s_prev = 0.0
    s_prev2 = 0.0
    for value in values:
        s = (value - avg) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2


def band_powers(values: list[float], fs: int = FS) -> dict[str, float]:
    powers: dict[str, float] = {}
    for band, (low, high) in FEATURE_BANDS.items():
        powers[band] = sum(goertzel_power(values, freq, fs) for freq in range(low, high))
    return powers


def peak_frequency(values: list[float], low: int, high: int, fs: int = FS) -> float:
    if not values:
        return 0.0
    best_freq = float(low)
    best_power = -1.0
    for freq in range(low, high + 1):
        power = goertzel_power(values, freq, fs)
        if power > best_power:
            best_power = power
            best_freq = float(freq)
    return best_freq


def spectral_entropy(values: list[float], low: int = 1, high: int = 45, fs: int = FS) -> float:
    powers = [goertzel_power(values, freq, fs) for freq in range(low, high + 1)]
    total = sum(powers)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for power in powers:
        probability = power / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(powers))


def safe_log_ratio(right: float, left: float) -> float:
    return math.log(right + 1.0) - math.log(left + 1.0)


@dataclass
class UserProfile:
    user_id: int
    name: str
    seed: int
    age: int
    eeg_state: str
    sleep_stage: str
    alertness: float
    pathology: str
    medication: str
    alpha_peak: float
    beta_peak: float
    theta_peak: float
    delta_peak: float
    gamma_peak: float
    alpha_scale: float
    beta_scale: float
    theta_scale: float
    delta_scale: float
    gamma_scale: float
    noise_scale: float
    left_right_bias: float
    frontal_posterior_bias: float
    coherence: float
    blink_rate_scale: float
    muscle_rate_scale: float
    pop_rate_scale: float
    line_noise_uv: float
    electrode_impedance: dict[str, float] = field(default_factory=dict)

    @classmethod
    def random_profile(cls, user_id: int, name: str | None = None, seed: int | None = None) -> UserProfile:
        seed = seed if seed is not None else 20260426 + user_id * 7919 + random.randint(0, 250000)
        rng = random.Random(seed)
        age = rng.randint(18, 72)
        alpha_peak = clamp(rng.gauss(10.0, 0.9) - max(age - 55, 0) * 0.015, 8.0, 12.5)
        profile = cls(
            user_id=user_id,
            name=name or f"User {user_id}",
            seed=seed,
            age=age,
            eeg_state="Awake - eyes closed",
            sleep_stage="Wake",
            alertness=round(rng.uniform(0.42, 0.88), 2),
            pathology=rng.choices(
                PATHOLOGY_TENDENCIES,
                weights=[0.62, 0.12, 0.10, 0.07, 0.09],
                k=1,
            )[0],
            medication=rng.choices(MEDICATION_EFFECTS, weights=[0.70, 0.10, 0.12, 0.08], k=1)[0],
            alpha_peak=alpha_peak,
            beta_peak=clamp(rng.gauss(19.0, 2.4), 14.0, 26.0),
            theta_peak=clamp(rng.gauss(5.6, 0.55), 4.4, 7.3),
            delta_peak=clamp(rng.gauss(1.35, 0.22), 0.8, 2.2),
            gamma_peak=clamp(rng.gauss(38.0, 4.0), 31.0, 45.0),
            alpha_scale=clamp(rng.lognormvariate(0.0, 0.22), 0.60, 1.55),
            beta_scale=clamp(rng.lognormvariate(0.0, 0.20), 0.65, 1.50),
            theta_scale=clamp(rng.lognormvariate(0.0, 0.24), 0.55, 1.70),
            delta_scale=clamp(rng.lognormvariate(0.0, 0.25), 0.55, 1.80),
            gamma_scale=clamp(rng.lognormvariate(-0.08, 0.24), 0.45, 1.40),
            noise_scale=clamp(rng.lognormvariate(0.0, 0.20), 0.65, 1.65),
            left_right_bias=clamp(rng.gauss(0.0, 0.11), -0.28, 0.28),
            frontal_posterior_bias=clamp(rng.gauss(0.0, 0.13), -0.30, 0.30),
            coherence=clamp(rng.uniform(0.35, 0.78), 0.20, 0.90),
            blink_rate_scale=clamp(rng.lognormvariate(0.0, 0.30), 0.45, 2.20),
            muscle_rate_scale=clamp(rng.lognormvariate(0.0, 0.28), 0.45, 2.10),
            pop_rate_scale=clamp(rng.lognormvariate(-0.10, 0.35), 0.30, 2.20),
            line_noise_uv=clamp(rng.gauss(3.0, 1.0), 0.5, 7.0),
        )
        profile.electrode_impedance = {
            electrode: clamp(rng.lognormvariate(0.0, 0.22), 0.65, 1.85)
            for electrode in ELECTRODES
        }
        if rng.random() < 0.25:
            weak_electrode = rng.choice(SCALP_ELECTRODES)
            profile.electrode_impedance[weak_electrode] = rng.uniform(1.8, 2.6)
        return profile

    def signature_summary(self) -> str:
        return (
            f"seed {self.seed} | alpha {self.alpha_peak:.1f} Hz | "
            f"LR {self.left_right_bias:+.2f} | noise x{self.noise_scale:.2f}"
        )


class EEGSignalGenerator:
    def __init__(self, profile: UserProfile, fs: int = FS) -> None:
        self.fs = fs
        self.profile = profile
        self.rng = random.Random(profile.seed)
        self.phase = {
            name: {
                "alpha": self.rng.random() * math.tau,
                "beta": self.rng.random() * math.tau,
                "theta": self.rng.random() * math.tau,
                "delta": self.rng.random() * math.tau,
                "gamma": self.rng.random() * math.tau,
                "line": self.rng.random() * math.tau,
                "drift": self.rng.random() * math.tau,
            }
            for name in ELECTRODES
        }
        self.common_phase = {
            band: self.rng.random() * math.tau
            for band in ["alpha", "beta", "theta", "delta", "gamma"]
        }
        self.spatial_gain = {
            electrode: {
                band: clamp(self.rng.lognormvariate(0.0, 0.14), 0.70, 1.38)
                for band in ["alpha", "beta", "theta", "delta", "gamma"]
            }
            for electrode in ELECTRODES
        }
        self.drift_amp = {name: self.rng.uniform(1.5, 6.0) for name in ELECTRODES}
        self.reset(-MAX_HISTORY_SECONDS)

    def set_profile(self, profile: UserProfile) -> None:
        self.profile = profile

    def reset(self, start_time: float = 0.0) -> None:
        self.last_blink = start_time - 10.0
        self.next_blink = start_time + self.rng.uniform(1.0, 4.0)
        self.next_muscle = start_time + self.rng.uniform(4.0, 9.0)
        self.muscle_start = start_time - 10.0
        self.muscle_until = start_time - 9.0
        self.next_pop = start_time + self.rng.uniform(8.0, 18.0)
        self.pop_start = start_time - 10.0
        self.pop_electrode = "F7"
        self.pop_amp = 0.0
        self.next_spindle = start_time + self.rng.uniform(3.0, 7.0)
        self.spindle_start = start_time - 10.0
        self.spindle_until = start_time - 9.0
        self.next_temporal_sharp = start_time + self.rng.uniform(5.0, 12.0)
        self.temporal_sharp_start = start_time - 10.0
        self.temporal_sharp_side = self.rng.choice(["left", "right"])
        self.temporal_sharp_amp = 0.0
        self.event_counts = {"blink": 0, "muscle": 0, "pop": 0, "spindle": 0, "sharp": 0}

    def _blink_interval(self) -> float:
        profile = self.profile
        alertness_factor = 1.25 - 0.45 * profile.alertness
        return self.rng.uniform(2.8, 6.5) * alertness_factor / max(profile.blink_rate_scale, 0.20)

    def _muscle_interval(self) -> float:
        profile = self.profile
        alertness_factor = 1.12 - 0.18 * profile.alertness
        return self.rng.uniform(7.0, 15.0) * alertness_factor / max(profile.muscle_rate_scale, 0.20)

    def _update_events(self, t: float, artifacts: dict[str, bool]) -> None:
        profile = self.profile
        if artifacts.get("blink", True) and t >= self.next_blink:
            self.last_blink = t
            self.next_blink = t + self._blink_interval()
            self.event_counts["blink"] += 1

        if artifacts.get("muscle", True) and t >= self.next_muscle:
            self.muscle_start = t
            self.muscle_until = t + self.rng.uniform(0.8, 2.2)
            self.next_muscle = t + self._muscle_interval()
            self.event_counts["muscle"] += 1

        if artifacts.get("pop", True) and t >= self.next_pop:
            self.pop_start = t
            self.pop_electrode = self.rng.choice(SCALP_ELECTRODES)
            self.pop_amp = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(95.0, 240.0)
            self.next_pop = t + self.rng.uniform(16.0, 32.0) / max(profile.pop_rate_scale, 0.20)
            self.event_counts["pop"] += 1

        if (profile.eeg_state == "N2 sleep" or profile.sleep_stage == "N2") and t >= self.next_spindle:
            self.spindle_start = t
            self.spindle_until = t + self.rng.uniform(0.8, 1.8)
            self.next_spindle = t + self.rng.uniform(5.0, 11.0)
            self.event_counts["spindle"] += 1

        if profile.pathology == "Temporal sharp tendency" and t >= self.next_temporal_sharp:
            self.temporal_sharp_start = t
            self.temporal_sharp_side = self.rng.choice(["left", "right"])
            self.temporal_sharp_amp = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(70.0, 160.0)
            self.next_temporal_sharp = t + self.rng.uniform(4.5, 12.0)
            self.event_counts["sharp"] += 1

    def _modulated_amplitudes(self) -> dict[str, float]:
        profile = self.profile
        state = STATE_PROFILES[profile.eeg_state]
        alpha = state["alpha"] * profile.alpha_scale
        beta = state["beta"] * profile.beta_scale
        theta = state["theta"] * profile.theta_scale
        delta = state["delta"] * profile.delta_scale
        gamma = state["gamma"] * profile.gamma_scale
        noise = state["noise"] * profile.noise_scale
        spindle = state["spindle"]
        spike_wave = state["spike_wave"]

        alpha *= 0.72 + 0.48 * profile.alertness
        beta *= 0.72 + 0.70 * profile.alertness
        theta *= 1.48 - 0.60 * profile.alertness
        delta *= 1.24 - 0.34 * profile.alertness

        if profile.age > 60:
            age_factor = min(profile.age - 60, 25) / 25.0
            alpha *= 1.0 - 0.16 * age_factor
            theta *= 1.0 + 0.18 * age_factor
            delta *= 1.0 + 0.14 * age_factor
        elif profile.age < 22:
            youth_factor = (22 - profile.age) / 6.0
            theta *= 1.0 + 0.16 * youth_factor

        if profile.sleep_stage == "Drowsy":
            alpha *= 0.85
            theta *= 1.35
            delta *= 1.12
        elif profile.sleep_stage == "N1":
            alpha *= 0.45
            theta *= 1.65
            delta *= 1.25
        elif profile.sleep_stage == "N2":
            alpha *= 0.35
            theta *= 1.45
            delta *= 1.55
            spindle = max(spindle, 32.0)
        elif profile.sleep_stage == "N3":
            alpha *= 0.20
            beta *= 0.45
            theta *= 1.10
            delta *= 2.85
        elif profile.sleep_stage == "REM":
            alpha *= 0.35
            beta *= 1.18
            theta *= 1.30
            delta *= 0.70

        if profile.medication == "Sedative effect":
            beta *= 1.22
            theta *= 1.22
            delta *= 1.12
            noise *= 0.92
        elif profile.medication == "Stimulant effect":
            beta *= 1.35
            gamma *= 1.20
            theta *= 0.82
            delta *= 0.84
        elif profile.medication == "Antiseizure effect":
            spike_wave *= 0.45
            noise *= 0.90
            beta *= 1.08

        if profile.pathology == "Frontal slowing tendency":
            theta *= 1.18
            delta *= 1.25
        elif profile.pathology == "Generalized spike-wave tendency":
            spike_wave = max(spike_wave, 38.0)
        elif profile.pathology == "Low voltage fast tendency":
            alpha *= 0.45
            theta *= 0.70
            delta *= 0.65
            beta *= 1.55
            gamma *= 1.35

        return {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma,
            "noise": noise,
            "spindle": spindle,
            "spike_wave": spike_wave,
        }

    def _mixed_wave(self, electrode: str, band: str, freq: float, t: float, coord_shift: float = 0.0) -> float:
        profile = self.profile
        local = math.sin(math.tau * freq * t + self.phase[electrode][band] + coord_shift)
        common = math.sin(math.tau * freq * t + self.common_phase[band])
        return profile.coherence * common + (1.0 - profile.coherence) * local

    def sample(self, t: float, artifacts: dict[str, bool], line_freq: float) -> dict[str, float]:
        self._update_events(t, artifacts)
        profile = self.profile
        amp = self._modulated_amplitudes()
        alpha_env = 0.72 + 0.28 * math.sin(math.tau * 0.18 * t + profile.seed % 17)
        theta_env = 0.75 + 0.25 * math.sin(math.tau * 0.11 * t + 1.2)
        delta_env = 0.78 + 0.22 * math.sin(math.tau * 0.07 * t + 0.4)
        beta_env = 0.85 + 0.15 * math.sin(math.tau * 0.24 * t + 0.8)
        values: dict[str, float] = {}

        spike_wave = 0.0
        if amp["spike_wave"] > 0:
            phase = (t * 3.0) % 1.0
            sharp = math.exp(-((phase - 0.035) / 0.018) ** 2)
            slow = math.exp(-((phase - 0.33) / 0.15) ** 2)
            spike_wave = amp["spike_wave"] * (1.35 * sharp - 0.72 * slow)

        spindle_value = 0.0
        if self.spindle_start <= t <= self.spindle_until:
            age = t - self.spindle_start
            duration = max(self.spindle_until - self.spindle_start, 0.001)
            envelope = math.sin(math.pi * age / duration) ** 2
            spindle_value = amp["spindle"] * envelope * math.sin(math.tau * 13.5 * t)

        blink_value = 0.0
        blink_age = t - self.last_blink
        if artifacts.get("blink", True) and 0.0 <= blink_age <= 0.72:
            blink_value = 175.0 * (math.sin(math.pi * blink_age / 0.72) ** 1.35)

        muscle_env = 0.0
        if artifacts.get("muscle", True) and self.muscle_start <= t <= self.muscle_until:
            duration = max(self.muscle_until - self.muscle_start, 0.001)
            age = t - self.muscle_start
            muscle_env = math.sin(math.pi * age / duration) ** 2

        pop_age = t - self.pop_start
        pop_active = artifacts.get("pop", True) and 0.0 <= pop_age <= 2.2
        sharp_age = t - self.temporal_sharp_start
        sharp_active = profile.pathology == "Temporal sharp tendency" and 0.0 <= sharp_age <= 0.42

        for name in ELECTRODES:
            x, y = ELECTRODE_POS[name]
            posterior = clamp((y + 0.18) / 1.10, 0.0, 1.0)
            anterior = clamp((-y + 0.20) / 1.20, 0.0, 1.0)
            temporal = clamp((abs(x) - 0.42) / 0.58, 0.0, 1.0)
            central = clamp(1.0 - math.hypot(x * 0.85, y * 0.85), 0.0, 1.0)
            hemisphere = 1.0 + profile.left_right_bias * (1.0 if x > 0 else -1.0 if x < 0 else 0.0)
            front_back = 1.0 + profile.frontal_posterior_bias * (anterior - posterior)
            impedance = profile.electrode_impedance.get(name, 1.0)

            alpha = (
                amp["alpha"]
                * (0.20 + 1.05 * posterior)
                * alpha_env
                * hemisphere
                * self.spatial_gain[name]["alpha"]
                * self._mixed_wave(name, "alpha", profile.alpha_peak, t, 0.25 * x)
            )
            alpha += (
                0.22
                * amp["alpha"]
                * posterior
                * math.sin(math.tau * (profile.alpha_peak - 0.8) * t + self.phase[name]["alpha"] * 0.7)
            )

            beta = (
                amp["beta"]
                * (0.30 + 0.75 * anterior + 0.20 * temporal)
                * beta_env
                * front_back
                * self.spatial_gain[name]["beta"]
                * (
                    0.65 * self._mixed_wave(name, "beta", profile.beta_peak, t)
                    + 0.35 * math.sin(math.tau * (profile.beta_peak + 4.8) * t + 0.7 * self.phase[name]["beta"])
                )
            )

            theta = (
                amp["theta"]
                * (0.35 + 0.45 * central + 0.35 * anterior)
                * theta_env
                * front_back
                * self.spatial_gain[name]["theta"]
                * self._mixed_wave(name, "theta", profile.theta_peak, t)
            )
            delta = (
                amp["delta"]
                * (0.35 + 0.55 * anterior + 0.30 * central)
                * delta_env
                * front_back
                * self.spatial_gain[name]["delta"]
                * self._mixed_wave(name, "delta", profile.delta_peak, t)
            )
            gamma = (
                amp["gamma"]
                * (0.22 + 0.45 * anterior + 0.35 * temporal)
                * self.spatial_gain[name]["gamma"]
                * math.sin(math.tau * profile.gamma_peak * t + self.phase[name]["gamma"])
            )

            if profile.pathology == "Frontal slowing tendency":
                delta += amp["delta"] * 0.42 * anterior * math.sin(math.tau * 1.1 * t + self.phase[name]["delta"])

            slow_drift = self.drift_amp[name] * math.sin(math.tau * 0.045 * t + self.phase[name]["drift"])
            measurement_noise = self.rng.gauss(0.0, amp["noise"] * (0.65 + 0.65 * impedance))
            value = alpha + beta + theta + delta + gamma + slow_drift + measurement_noise

            if amp["spike_wave"] > 0:
                value += spike_wave * (0.65 + 0.35 * central + 0.20 * anterior)

            if spindle_value:
                value += spindle_value * (0.45 + 0.80 * central)

            if blink_value:
                blink_weight = 0.10 + 1.30 * anterior
                if name in {"Fp1", "Fp2"}:
                    blink_weight *= 1.35
                value += blink_value * blink_weight

            if muscle_env:
                fast = (
                    20.0 * math.sin(math.tau * 43.0 * t + self.phase[name]["beta"])
                    + 13.0 * math.sin(math.tau * 71.0 * t + self.phase[name]["line"])
                    + self.rng.gauss(0.0, 14.0)
                )
                value += muscle_env * temporal * fast

            if artifacts.get("line", True):
                value += profile.line_noise_uv * impedance * math.sin(
                    math.tau * line_freq * t + self.phase[name]["line"]
                )

            if pop_active and name == self.pop_electrode:
                value += self.pop_amp * math.exp(-pop_age / 0.34)

            if sharp_active:
                side_match = (self.temporal_sharp_side == "left" and x < -0.35) or (
                    self.temporal_sharp_side == "right" and x > 0.35
                )
                if side_match and temporal > 0.2:
                    sharp = math.exp(-((sharp_age - 0.055) / 0.025) ** 2)
                    slow = math.exp(-((sharp_age - 0.21) / 0.10) ** 2)
                    value += temporal * self.temporal_sharp_amp * (sharp - 0.45 * slow)

            values[name] = value

        return values


class EEGUserSession:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile
        self.generator = EEGSignalGenerator(profile)
        self.times: deque[float] = deque(maxlen=MAX_HISTORY_SECONDS * FS)
        self.buffers = {
            electrode: deque(maxlen=MAX_HISTORY_SECONDS * FS) for electrode in ELECTRODES
        }
        self.events: list[tuple[float, str]] = []
        self.sim_time = 0.0
        self.feature_history: deque[dict[str, object]] = deque(maxlen=FEATURE_HISTORY_ROWS)
        self.feature_cache_time = -999.0
        self.feature_cache_montage = ""
        self.feature_cache: dict[str, object] = {}
        self.band_cache_time = -999.0
        self.band_cache: dict[str, float] = {}
        self.last_feature_row_time = -999.0

    def clear_buffers(self) -> None:
        self.times.clear()
        for buffer in self.buffers.values():
            buffer.clear()
        self.events.clear()
        self.feature_history.clear()
        self.feature_cache_time = -999.0
        self.band_cache_time = -999.0
        self.last_feature_row_time = -999.0

    def prefill_history(self, seconds: float, artifacts: dict[str, bool], line_freq: float) -> None:
        self.clear_buffers()
        start_time = -seconds
        self.generator.reset(start_time)
        total = int(seconds * FS)
        for index in range(total):
            t = start_time + index / FS
            self.append_sample_at(t, artifacts, line_freq)
        self.sim_time = 0.0

    def append_sample_at(self, t: float, artifacts: dict[str, bool], line_freq: float) -> None:
        sample = self.generator.sample(t, artifacts, line_freq)
        self.times.append(t)
        for electrode in ELECTRODES:
            self.buffers[electrode].append(sample[electrode])

    def step(self, artifacts: dict[str, bool], line_freq: float) -> None:
        self.sim_time += 1.0 / FS
        self.append_sample_at(self.sim_time, artifacts, line_freq)

    def replace_profile(self, profile: UserProfile, artifacts: dict[str, bool], line_freq: float) -> None:
        self.profile = profile
        self.generator = EEGSignalGenerator(profile)
        self.prefill_history(min(30.0, MAX_HISTORY_SECONDS - 1), artifacts, line_freq)


