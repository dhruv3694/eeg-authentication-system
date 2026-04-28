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


class ClinicalEEGSimulator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Multi-User Real-Time Clinical EEG Simulator")
        self.geometry("1500x930")
        self.minsize(1050, 680)

        self.sessions: list[EEGUserSession] = []
        self.active_index = 0
        self.user_counter = 0
        self.running = True
        self.last_wall = time.perf_counter()
        self.last_draw = 0.0
        self.last_feature_update = 0.0
        self.loading_profile = False

        self.montage_var = tk.StringVar(value="Longitudinal bipolar")
        self.sweep_var = tk.StringVar(value="10")
        self.sensitivity_var = tk.StringVar(value="7")
        self.lff_var = tk.StringVar(value="1.0")
        self.hff_var = tk.StringVar(value="70")
        self.line_freq_var = tk.StringVar(value="50")
        self.notch_var = tk.BooleanVar(value=True)
        self.blink_var = tk.BooleanVar(value=True)
        self.muscle_var = tk.BooleanVar(value=True)
        self.line_var = tk.BooleanVar(value=True)
        self.pop_var = tk.BooleanVar(value=True)

        self.profile_name_var = tk.StringVar()
        self.profile_age_var = tk.StringVar()
        self.state_var = tk.StringVar()
        self.sleep_stage_var = tk.StringVar()
        self.alertness_var = tk.DoubleVar(value=0.70)
        self.alertness_text_var = tk.StringVar(value="0.70")
        self.pathology_var = tk.StringVar()
        self.medication_var = tk.StringVar()

        self._build_style()
        self._build_ui()
        self._add_user(initial=True)
        self.after(20, self._tick)

    @property
    def active_session(self) -> EEGUserSession:
        return self.sessions[self.active_index]

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#eef2f7")
        style.configure("Toolbar.TFrame", background="#dbe3ec")
        style.configure("UserBar.TFrame", background="#cbd5e1")
        style.configure("TLabel", background="#eef2f7", foreground="#111827")
        style.configure("Toolbar.TLabel", background="#dbe3ec", foreground="#111827")
        style.configure("TButton", padding=(8, 4))
        style.configure("TCheckbutton", background="#dbe3ec", foreground="#111827")

    def _build_ui(self) -> None:
        self.user_bar = ttk.Frame(self, style="UserBar.TFrame", padding=(8, 5))
        self.user_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(self.user_bar, text="Users", background="#cbd5e1").pack(side=tk.LEFT, padx=(0, 8))
        self.user_buttons_frame = ttk.Frame(self.user_bar, style="UserBar.TFrame")
        self.user_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.user_bar, text="+ User", command=self._add_user).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(self.user_bar, text="Remove", command=self._remove_active_user).pack(side=tk.RIGHT, padx=(6, 0))

        profile = ttk.Frame(self, style="Toolbar.TFrame", padding=(8, 5))
        profile.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(profile, text="Profile", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.name_entry = ttk.Entry(profile, textvariable=self.profile_name_var, width=13)
        self.name_entry.pack(side=tk.LEFT, padx=(0, 6))
        self.name_entry.bind("<Return>", lambda _event: self._apply_profile_from_ui())

        ttk.Label(profile, text="Age", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(4, 3))
        age = ttk.Spinbox(profile, from_=5, to=95, width=4, textvariable=self.profile_age_var)
        age.pack(side=tk.LEFT, padx=(0, 6))

        self._combo(profile, "State", self.state_var, list(STATE_PROFILES.keys()), 24, self._apply_profile_from_ui)
        self._combo(profile, "Sleep", self.sleep_stage_var, SLEEP_STAGES, 8, self._apply_profile_from_ui)
        self._combo(profile, "Pathology", self.pathology_var, PATHOLOGY_TENDENCIES, 27, self._apply_profile_from_ui)
        self._combo(profile, "Medication", self.medication_var, MEDICATION_EFFECTS, 18, self._apply_profile_from_ui)

        ttk.Label(profile, text="Alert", style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(8, 2))
        alert = ttk.Scale(
            profile,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.alertness_var,
            command=self._on_alertness_change,
            length=88,
        )
        alert.pack(side=tk.LEFT)
        ttk.Label(profile, textvariable=self.alertness_text_var, style="Toolbar.TLabel", width=4).pack(side=tk.LEFT)

        ttk.Button(profile, text="Apply Profile", command=self._apply_profile_from_ui).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(profile, text="Randomize Signature", command=self._randomize_active_signature).pack(side=tk.LEFT, padx=2)

        toolbar = ttk.Frame(self, style="Toolbar.TFrame", padding=(8, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.run_button = ttk.Button(toolbar, text="Pause", width=8, command=self._toggle_run)
        self.run_button.pack(side=tk.LEFT, padx=(0, 8))

        self._combo(toolbar, "Montage", self.montage_var, list(MONTAGES.keys()), 21, self._draw)
        self._combo(toolbar, "Seconds", self.sweep_var, ["5", "10", "15", "20", "30", "40"], 5, self._draw)
        self._combo(toolbar, "uV/mm", self.sensitivity_var, ["3", "5", "7", "10", "15", "20", "30", "50"], 5, self._draw)
        self._combo(toolbar, "LFF", self.lff_var, ["0.1", "0.5", "1.0", "2.0"], 5, self._draw)
        self._combo(toolbar, "HFF", self.hff_var, ["15", "35", "70"], 5, self._draw)
        self._combo(toolbar, "Hz", self.line_freq_var, ["50", "60"], 4, self._draw)

        ttk.Checkbutton(toolbar, text="Notch", variable=self.notch_var).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Checkbutton(toolbar, text="Blink", variable=self.blink_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="EMG", variable=self.muscle_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="Line", variable=self.line_var).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="Pop", variable=self.pop_var).pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="Event", command=self._add_marker).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(toolbar, text="Export Raw CSV", command=self._export_raw_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export Features", command=self._export_feature_csv).pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self, background="#f8fafc", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(8, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        width: int,
        callback: object | None = None,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label, style="Toolbar.TLabel").pack(side=tk.LEFT, padx=(8, 3))
        box = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=width,
            state="readonly",
        )
        box.pack(side=tk.LEFT)
        if callback is not None:
            box.bind("<<ComboboxSelected>>", lambda _event: callback())
        return box

    def _artifact_flags(self) -> dict[str, bool]:
        return {
            "blink": self.blink_var.get(),
            "muscle": self.muscle_var.get(),
            "line": self.line_var.get(),
            "pop": self.pop_var.get(),
        }

    def _add_user(self, initial: bool = False) -> None:
        self.user_counter += 1
        profile = UserProfile.random_profile(self.user_counter)
        session = EEGUserSession(profile)
        session.prefill_history(min(30.0, MAX_HISTORY_SECONDS - 1), self._artifact_flags(), float(self.line_freq_var.get()))
        self.sessions.append(session)
        self.active_index = len(self.sessions) - 1
        self._refresh_user_tabs()
        self._load_profile_to_ui(profile)
        self._draw()
        if not initial:
            self.status_var.set(f"Added {profile.name}: {profile.signature_summary()}")

    def _remove_active_user(self) -> None:
        if len(self.sessions) <= 1:
            messagebox.showinfo("Users", "Keep at least one synthetic user in the simulator.")
            return
        removed = self.sessions.pop(self.active_index)
        self.active_index = min(self.active_index, len(self.sessions) - 1)
        self._refresh_user_tabs()
        self._load_profile_to_ui(self.active_session.profile)
        self._draw()
        self.status_var.set(f"Removed {removed.profile.name}.")

    def _refresh_user_tabs(self) -> None:
        for child in self.user_buttons_frame.winfo_children():
            child.destroy()
        for index, session in enumerate(self.sessions):
            selected = index == self.active_index
            button = tk.Button(
                self.user_buttons_frame,
                text=session.profile.name,
                padx=10,
                pady=3,
                relief=tk.SUNKEN if selected else tk.RAISED,
                bg="#f8fafc" if selected else "#e2e8f0",
                fg="#0f172a",
                activebackground="#ffffff",
                command=lambda idx=index: self._select_user(idx),
            )
            button.pack(side=tk.LEFT, padx=(0, 5))

    def _select_user(self, index: int) -> None:
        if index == self.active_index:
            return
        self._apply_profile_from_ui(redraw=False)
        self.active_index = index
        self._refresh_user_tabs()
        self._load_profile_to_ui(self.active_session.profile)
        self._draw()

    def _load_profile_to_ui(self, profile: UserProfile) -> None:
        self.loading_profile = True
        self.profile_name_var.set(profile.name)
        self.profile_age_var.set(str(profile.age))
        self.state_var.set(profile.eeg_state)
        self.sleep_stage_var.set(profile.sleep_stage)
        self.alertness_var.set(profile.alertness)
        self.alertness_text_var.set(f"{profile.alertness:.2f}")
        self.pathology_var.set(profile.pathology)
        self.medication_var.set(profile.medication)
        self.loading_profile = False

    def _on_alertness_change(self, value: str) -> None:
        alertness = float(value)
        self.alertness_text_var.set(f"{alertness:.2f}")
        if not self.loading_profile and self.sessions:
            self.active_session.profile.alertness = alertness

    def _apply_profile_from_ui(self, redraw: bool = True) -> None:
        if self.loading_profile or not self.sessions:
            return
        session = self.active_session
        profile = session.profile
        name = self.profile_name_var.get().strip() or f"User {profile.user_id}"
        try:
            age = int(float(self.profile_age_var.get()))
        except ValueError:
            age = profile.age
        profile.name = name
        profile.age = int(clamp(age, 5, 95))
        profile.eeg_state = self.state_var.get() if self.state_var.get() in STATE_PROFILES else profile.eeg_state
        profile.sleep_stage = self.sleep_stage_var.get() if self.sleep_stage_var.get() in SLEEP_STAGES else profile.sleep_stage
        profile.alertness = clamp(float(self.alertness_var.get()), 0.0, 1.0)
        profile.pathology = self.pathology_var.get() if self.pathology_var.get() in PATHOLOGY_TENDENCIES else profile.pathology
        profile.medication = self.medication_var.get() if self.medication_var.get() in MEDICATION_EFFECTS else profile.medication
        session.generator.set_profile(profile)
        self._refresh_user_tabs()
        self._load_profile_to_ui(profile)
        if redraw:
            self._draw()

    def _randomize_active_signature(self) -> None:
        session = self.active_session
        old = session.profile
        seed = 20260426 + old.user_id * 7919 + random.randint(0, 900000)
        profile = UserProfile.random_profile(old.user_id, name=old.name, seed=seed)
        profile.eeg_state = old.eeg_state
        profile.sleep_stage = old.sleep_stage
        profile.pathology = old.pathology
        profile.medication = old.medication
        profile.alertness = old.alertness
        session.replace_profile(profile, self._artifact_flags(), float(self.line_freq_var.get()))
        self._load_profile_to_ui(profile)
        self._draw()

    def _tick(self) -> None:
        now = time.perf_counter()
        if self.running:
            elapsed = now - self.last_wall
            samples = int(elapsed * FS)
            samples = min(samples, FS // 2)
            if samples > 0:
                artifacts = self._artifact_flags()
                line_freq = float(self.line_freq_var.get())
                for _ in range(samples):
                    for session in self.sessions:
                        session.step(artifacts, line_freq)
                self.last_wall += samples / FS
        else:
            self.last_wall = now

        if now - self.last_feature_update > 0.6:
            self._update_feature_history()
            self.last_feature_update = now

        if now - self.last_draw > 0.055:
            self._draw()
            self.last_draw = now
        self.after(15, self._tick)

    def _toggle_run(self) -> None:
        self.running = not self.running
        self.run_button.configure(text="Pause" if self.running else "Run")
        self.last_wall = time.perf_counter()

    def _add_marker(self) -> None:
        session = self.active_session
        session.events.append((session.sim_time, f"E{len(session.events) + 1}"))

    def _get_window(
        self,
        seconds: float | None = None,
        session: EEGUserSession | None = None,
    ) -> tuple[list[float], dict[str, list[float]]]:
        session = session or self.active_session
        page_seconds = seconds if seconds is not None else float(self.sweep_var.get())
        start = session.sim_time - page_seconds
        times = list(session.times)
        start_index = bisect.bisect_left(times, start)
        window_times = times[start_index:]
        data = {electrode: list(buffer)[start_index:] for electrode, buffer in session.buffers.items()}
        return window_times, data

    def _reference_average(self, data: dict[str, list[float]]) -> list[float]:
        length = len(next(iter(data.values()))) if data else 0
        averages: list[float] = []
        for index in range(length):
            averages.append(sum(data[e][index] for e in SCALP_ELECTRODES) / len(SCALP_ELECTRODES))
        return averages

    def _montage_signal(
        self,
        pos: str,
        neg: str,
        data: dict[str, list[float]],
        average: list[float] | None,
    ) -> list[float]:
        first = data[pos]
        if neg == "AVG":
            if average is None:
                average = self._reference_average(data)
            return [first[i] - average[i] for i in range(len(first))]
        if neg == "A1A2":
            a1 = data["A1"]
            a2 = data["A2"]
            return [first[i] - 0.5 * (a1[i] + a2[i]) for i in range(len(first))]
        second = data[neg]
        return [first[i] - second[i] for i in range(len(first))]

    def _display_filter(self, values: list[float]) -> list[float]:
        if not values:
            return values
        filtered = values
        filtered = highpass(filtered, float(self.lff_var.get()), FS)
        if self.notch_var.get():
            filtered = notch(filtered, float(self.line_freq_var.get()), FS)
        filtered = lowpass(filtered, float(self.hff_var.get()), FS)
        return filtered

    def _export_raw_csv(self) -> None:
        self._apply_profile_from_ui(redraw=False)
        session = self.active_session
        times, data = self._get_window(session=session)
        if not times:
            messagebox.showinfo("Export Raw CSV", "No samples are available yet.")
            return

        initial = f"{session.profile.name.lower().replace(' ', '_')}_eeg_page.csv"
        path = filedialog.asksaveasfilename(
            title="Export current EEG page",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=initial,
        )
        if not path:
            return

        montage = MONTAGES[self.montage_var.get()]
        average = self._reference_average(data) if any(ch[2] == "AVG" for ch in montage) else None
        signals = [
            (label, self._montage_signal(pos, neg, data, average))
            for label, pos, neg in montage
        ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "user_id",
                        "user_name",
                        "state",
                        "sleep_stage",
                        "pathology",
                        "medication",
                        "time_s",
                        *[label for label, _signal in signals],
                    ]
                )
                for index, t in enumerate(times):
                    writer.writerow(
                        [
                            session.profile.user_id,
                            session.profile.name,
                            session.profile.eeg_state,
                            session.profile.sleep_stage,
                            session.profile.pathology,
                            session.profile.medication,
                            f"{t:.4f}",
                            *[f"{signal[index]:.3f}" for _label, signal in signals],
                        ]
                    )
        except OSError as exc:
            messagebox.showerror("Export Raw CSV", f"Could not write file:\n{exc}")
            return

        messagebox.showinfo("Export Raw CSV", f"Exported {len(times)} samples for {session.profile.name}.")

    def _export_feature_csv(self) -> None:
        self._apply_profile_from_ui(redraw=False)
        self._update_feature_history(force=True)
        rows: list[dict[str, object]] = []
        for session in self.sessions:
            rows.extend(session.feature_history)
        if not rows:
            messagebox.showinfo("Export Features", "No feature rows are available yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Export synthetic EEG feature rows",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="synthetic_eeg_features.csv",
        )
        if not path:
            return

        preferred = [
            "user_id",
            "user_name",
            "seed",
            "sim_time_s",
            "channel",
            "age",
            "alertness",
            "state",
            "sleep_stage",
            "pathology",
            "medication",
            "alpha_peak_personal_hz",
            "alpha_peak_est_hz",
            "rms_uv",
            "variance_uv2",
            "line_length",
            "zero_cross_rate",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
            "spectral_entropy",
            "theta_alpha_ratio",
            "beta_alpha_ratio",
            "posterior_alpha_asym",
            "frontal_beta_asym",
            "blink_count",
            "muscle_burst_count",
            "electrode_pop_count",
            "temporal_sharp_count",
        ]
        for band in FEATURE_BANDS:
            preferred.extend([f"{band}_power", f"{band}_rel"])
        keys = preferred + sorted({key for row in rows for key in row if key not in preferred})

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in keys})
        except OSError as exc:
            messagebox.showerror("Export Features", f"Could not write file:\n{exc}")
            return

        messagebox.showinfo("Export Features", f"Exported {len(rows)} feature rows across {len(self.sessions)} users.")

    def _update_feature_history(self, force: bool = False) -> None:
        montage_name = self.montage_var.get()
        for session in self.sessions:
            if not force and session.sim_time - session.last_feature_row_time < 0.95:
                continue
            _times, data = self._get_window(seconds=5.0, session=session)
            features = self._extract_features(session, data, montage_name)
            if features:
                session.feature_history.append(features.copy())
                session.last_feature_row_time = session.sim_time

    def _extract_features(
        self,
        session: EEGUserSession,
        data: dict[str, list[float]],
        montage_name: str,
    ) -> dict[str, object]:
        if session.sim_time - session.feature_cache_time < 0.45 and session.feature_cache_montage == montage_name:
            return session.feature_cache

        if not data or not next(iter(data.values()), []):
            return {}
        montage = MONTAGES[montage_name]
        channel_label, pos, neg = montage[0]
        average = self._reference_average(data) if neg == "AVG" else None
        signal = self._montage_signal(pos, neg, data, average)
        window = signal[-min(len(signal), 5 * FS):]
        if len(window) < FS:
            return {}
        window = highpass(window, 0.5, FS)
        window = lowpass(window, 45.0, FS)

        powers = band_powers(window, FS)
        total_power = sum(powers.values()) or 1.0
        activity, mobility, complexity = hjorth(window)
        alpha_power = powers["alpha"] or 1.0
        profile = session.profile

        o1_alpha = self._electrode_band_power(data, "O1", "alpha")
        o2_alpha = self._electrode_band_power(data, "O2", "alpha")
        f3_beta = self._electrode_band_power(data, "F3", "beta")
        f4_beta = self._electrode_band_power(data, "F4", "beta")
        features: dict[str, object] = {
            "user_id": profile.user_id,
            "user_name": profile.name,
            "seed": profile.seed,
            "sim_time_s": round(session.sim_time, 3),
            "channel": channel_label,
            "age": profile.age,
            "alertness": round(profile.alertness, 3),
            "state": profile.eeg_state,
            "sleep_stage": profile.sleep_stage,
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
            "theta_alpha_ratio": round(powers["theta"] / alpha_power, 5),
            "beta_alpha_ratio": round(powers["beta"] / alpha_power, 5),
            "posterior_alpha_asym": round(safe_log_ratio(o2_alpha, o1_alpha), 5),
            "frontal_beta_asym": round(safe_log_ratio(f4_beta, f3_beta), 5),
            "blink_count": session.generator.event_counts["blink"],
            "muscle_burst_count": session.generator.event_counts["muscle"],
            "electrode_pop_count": session.generator.event_counts["pop"],
            "temporal_sharp_count": session.generator.event_counts["sharp"],
        }
        for band, power in powers.items():
            features[f"{band}_power"] = round(power, 5)
            features[f"{band}_rel"] = round(power / total_power, 5)

        session.feature_cache_time = session.sim_time
        session.feature_cache_montage = montage_name
        session.feature_cache = features
        return features

    def _electrode_band_power(self, data: dict[str, list[float]], electrode: str, band: str) -> float:
        if electrode not in data:
            return 0.0
        values = data[electrode][-min(len(data[electrode]), 5 * FS):]
        if len(values) < FS:
            return 0.0
        filtered = highpass(values, 0.5, FS)
        filtered = lowpass(filtered, 45.0, FS)
        low, high = FEATURE_BANDS[band]
        return sum(goertzel_power(filtered, freq, FS) for freq in range(low, high))

    def _draw(self) -> None:
        if not self.sessions:
            return
        canvas = self.canvas
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)
        canvas.delete("all")

        session = self.active_session
        page_seconds = float(self.sweep_var.get())
        sensitivity = float(self.sensitivity_var.get())
        montage = MONTAGES[self.montage_var.get()]
        times, data = self._get_window(page_seconds, session)

        side_width = 310 if width >= 1220 else 0
        left = 106
        right = width - side_width - 16
        top = 44
        bottom = height - 38
        plot_width = max(80, right - left)
        plot_height = max(80, bottom - top)

        self._draw_grid(canvas, left, top, right, bottom, page_seconds)

        if not times:
            canvas.create_text(
                width / 2,
                height / 2,
                text="waiting for samples",
                fill="#475569",
                font=("Segoe UI", 14),
            )
            return

        average = self._reference_average(data) if any(ch[2] == "AVG" for ch in montage) else None
        channel_count = len(montage)
        spacing = plot_height / max(channel_count, 1)
        pixels_per_mm = clamp(spacing / 11.5, 2.2, 5.6)
        pixels_per_uv = pixels_per_mm / sensitivity
        visible_start = session.sim_time - page_seconds

        self._draw_headers(
            canvas,
            left,
            top,
            right,
            page_seconds,
            sensitivity,
            pixels_per_mm,
            session,
        )

        for index, (label, pos, neg) in enumerate(montage):
            baseline = top + spacing * (index + 0.5)
            canvas.create_text(
                left - 10,
                baseline,
                text=label,
                anchor=tk.E,
                fill="#0f172a",
                font=("Consolas", 9),
            )
            canvas.create_line(left, baseline, right, baseline, fill="#d2dae5")
            raw_signal = self._montage_signal(pos, neg, data, average)
            signal = self._display_filter(raw_signal)
            if len(signal) < 2:
                continue
            step = max(1, len(signal) // max(plot_width, 1))
            points: list[float] = []
            for sample_index in range(0, len(signal), step):
                t = times[sample_index]
                x = left + ((t - visible_start) / page_seconds) * plot_width
                y = baseline + signal[sample_index] * pixels_per_uv
                points.extend((x, y))
            if len(points) >= 4:
                canvas.create_line(*points, fill="#111827", width=1, tags=("trace",))

        self._draw_event_markers(canvas, left, top, right, bottom, visible_start, page_seconds, session)
        self._draw_calibration(canvas, right, top, pixels_per_uv, page_seconds, plot_width)

        if side_width:
            self._draw_side_panel(canvas, width - side_width + 8, top, width - 10, bottom, data, session)

        self.status_var.set(
            f"{session.profile.name} | FS {FS} Hz | {self.montage_var.get()} | "
            f"{session.profile.eeg_state} | {sensitivity:g} uV/mm | {page_seconds:g} s/page | "
            f"LFF {self.lff_var.get()} Hz  HFF {self.hff_var.get()} Hz  "
            f"{'notch ' + self.line_freq_var.get() + ' Hz' if self.notch_var.get() else 'notch off'} | "
            f"negative-up polarity | features {len(session.feature_history)} rows | t={session.sim_time:0.1f}s"
        )

    def _draw_grid(
        self,
        canvas: tk.Canvas,
        left: float,
        top: float,
        right: float,
        bottom: float,
        page_seconds: float,
    ) -> None:
        canvas.create_rectangle(left, top, right, bottom, fill="#fbfdff", outline="#cbd5e1")
        plot_width = right - left
        minor_step = 0.2
        for tick in range(int(page_seconds / minor_step) + 1):
            seconds = tick * minor_step
            x = left + seconds / page_seconds * plot_width
            major = tick % 5 == 0
            canvas.create_line(
                x,
                top,
                x,
                bottom,
                fill="#cfd8e3" if major else "#edf2f7",
            )
            if major:
                canvas.create_text(
                    x,
                    bottom + 16,
                    text=f"{seconds:g}",
                    fill="#64748b",
                    font=("Segoe UI", 8),
                )

        y = top
        while y <= bottom:
            canvas.create_line(left, y, right, y, fill="#eef3f8")
            y += 10

    def _draw_headers(
        self,
        canvas: tk.Canvas,
        left: float,
        top: float,
        right: float,
        page_seconds: float,
        sensitivity: float,
        pixels_per_mm: float,
        session: EEGUserSession,
    ) -> None:
        rec_fill = "#dc2626" if self.running else "#64748b"
        canvas.create_oval(left, 14, left + 12, 26, fill=rec_fill, outline="")
        canvas.create_text(
            left + 20,
            20,
            text=f"{'LIVE' if self.running else 'PAUSED'}  {session.profile.name}",
            anchor=tk.W,
            fill="#111827",
            font=("Segoe UI", 10, "bold"),
        )
        canvas.create_text(
            right,
            20,
            text=f"{page_seconds:g} s/page   {sensitivity:g} uV/mm   {pixels_per_mm:.1f} px/mm",
            anchor=tk.E,
            fill="#334155",
            font=("Segoe UI", 9),
        )
        canvas.create_text(left - 10, top - 18, text="Channel", anchor=tk.E, fill="#475569", font=("Segoe UI", 9))

    def _draw_calibration(
        self,
        canvas: tk.Canvas,
        right: float,
        top: float,
        pixels_per_uv: float,
        page_seconds: float,
        plot_width: float,
    ) -> None:
        cal_uv = 100.0
        cal_seconds = 1.0
        cal_width = cal_seconds / page_seconds * plot_width
        cal_height = cal_uv * pixels_per_uv
        x0 = right - cal_width - 18
        y0 = top + 26
        y1 = y0 - cal_height
        points = [x0, y0, x0, y1, x0 + cal_width, y1, x0 + cal_width, y0]
        canvas.create_line(*points, fill="#2563eb", width=2)
        canvas.create_text(
            x0 + cal_width / 2,
            max(top + 8, y1 - 10),
            text="100 uV / 1 s",
            fill="#1d4ed8",
            font=("Segoe UI", 8),
        )

    def _draw_event_markers(
        self,
        canvas: tk.Canvas,
        left: float,
        top: float,
        right: float,
        bottom: float,
        visible_start: float,
        page_seconds: float,
        session: EEGUserSession,
    ) -> None:
        plot_width = right - left
        for event_time, label in session.events:
            if visible_start <= event_time <= session.sim_time:
                x = left + ((event_time - visible_start) / page_seconds) * plot_width
                canvas.create_line(x, top, x, bottom, fill="#e11d48", width=1)
                canvas.create_text(
                    x + 4,
                    top + 8,
                    text=label,
                    anchor=tk.W,
                    fill="#be123c",
                    font=("Segoe UI", 8, "bold"),
                )

    def _draw_side_panel(
        self,
        canvas: tk.Canvas,
        left: float,
        top: float,
        right: float,
        bottom: float,
        data: dict[str, list[float]],
        session: EEGUserSession,
    ) -> None:
        canvas.create_rectangle(left, top, right, bottom, fill="#f8fafc", outline="#cbd5e1")
        canvas.create_text(left + 8, top + 16, text="Scalp uV", anchor=tk.W, fill="#111827", font=("Segoe UI", 10, "bold"))

        cx = (left + right) / 2
        cy = top + 112
        radius = min(72, (right - left) * 0.34)
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#334155", width=2)
        canvas.create_arc(cx - 14, cy - radius - 10, cx + 14, cy - radius + 12, start=0, extent=180, outline="#334155", width=2)

        latest = {electrode: (values[-1] if values else 0.0) for electrode, values in data.items()}
        for electrode in SCALP_ELECTRODES:
            ex, ey = ELECTRODE_POS[electrode]
            px = cx + ex * radius * 0.92
            py = cy + ey * radius * 0.92
            color = voltage_color(latest[electrode])
            canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill=color, outline="#0f172a")
            canvas.create_text(px, py + 13, text=electrode, fill="#334155", font=("Segoe UI", 7))

        scale_y = cy + radius + 22
        for i, value in enumerate([-120, -60, 0, 60, 120]):
            x = left + 24 + i * 48
            canvas.create_rectangle(x, scale_y, x + 42, scale_y + 9, fill=voltage_color(value), outline="")
        canvas.create_text(left + 8, scale_y + 22, text="-120", anchor=tk.W, fill="#64748b", font=("Segoe UI", 7))
        canvas.create_text(right - 8, scale_y + 22, text="+120 uV", anchor=tk.E, fill="#64748b", font=("Segoe UI", 7))

        spectrum_top = scale_y + 48
        canvas.create_text(left + 8, spectrum_top, text="Band power", anchor=tk.W, fill="#111827", font=("Segoe UI", 10, "bold"))
        bands = self._band_power_for_panel(data, session)
        if bands:
            max_power = max(bands.values()) or 1.0
            colors = {
                "Delta": "#64748b",
                "Theta": "#0f766e",
                "Alpha": "#2563eb",
                "Beta": "#7c3aed",
                "Gamma": "#db2777",
            }
            bar_left = left + 62
            bar_right = right - 12
            y = spectrum_top + 24
            for band, power in bands.items():
                normalized = math.log10(power + 1.0) / math.log10(max_power + 1.0)
                canvas.create_text(left + 10, y + 6, text=band, anchor=tk.W, fill="#334155", font=("Segoe UI", 8))
                canvas.create_rectangle(bar_left, y, bar_right, y + 12, fill="#e2e8f0", outline="")
                canvas.create_rectangle(
                    bar_left,
                    y,
                    bar_left + (bar_right - bar_left) * normalized,
                    y + 12,
                    fill=colors[band],
                    outline="",
                )
                y += 23
        else:
            y = spectrum_top + 24

        profile = session.profile
        signature_top = y + 14
        canvas.create_text(left + 8, signature_top, text="Synthetic signature", anchor=tk.W, fill="#111827", font=("Segoe UI", 10, "bold"))
        lines = [
            f"Age {profile.age}  alert {profile.alertness:.2f}",
            f"Alpha {profile.alpha_peak:.1f} Hz  beta {profile.beta_peak:.1f} Hz",
            f"LR bias {profile.left_right_bias:+.2f}  coherence {profile.coherence:.2f}",
            f"Noise x{profile.noise_scale:.2f}  line {profile.line_noise_uv:.1f} uV",
            profile.pathology,
            profile.medication,
        ]
        text_y = signature_top + 22
        for line in lines:
            canvas.create_text(left + 10, text_y, text=line, anchor=tk.W, fill="#334155", font=("Segoe UI", 8))
            text_y += 17

        features = self._extract_features(session, data, self.montage_var.get())
        feature_top = text_y + 12
        canvas.create_text(left + 8, feature_top, text="Live features", anchor=tk.W, fill="#111827", font=("Segoe UI", 10, "bold"))
        feature_lines = [
            f"RMS {features.get('rms_uv', 0)} uV",
            f"Alpha est {features.get('alpha_peak_est_hz', 0)} Hz",
            f"Theta/alpha {features.get('theta_alpha_ratio', 0)}",
            f"Beta/alpha {features.get('beta_alpha_ratio', 0)}",
            f"Entropy {features.get('spectral_entropy', 0)}",
            f"Post alpha asym {features.get('posterior_alpha_asym', 0)}",
        ]
        text_y = feature_top + 22
        for line in feature_lines:
            if text_y < bottom - 8:
                canvas.create_text(left + 10, text_y, text=line, anchor=tk.W, fill="#334155", font=("Segoe UI", 8))
            text_y += 17

    def _band_power_for_panel(self, data: dict[str, list[float]], session: EEGUserSession) -> dict[str, float]:
        if session.sim_time - session.band_cache_time < 0.4:
            return session.band_cache

        channel_label, pos, neg = MONTAGES[self.montage_var.get()][0]
        average = self._reference_average(data) if neg == "AVG" else None
        signal = self._montage_signal(pos, neg, data, average)
        window = signal[-min(len(signal), 3 * FS):]
        if len(window) < FS:
            return {}
        window = highpass(window, 0.5, FS)
        window = lowpass(window, 45.0, FS)

        raw = band_powers(window, FS)
        bands = {band.title(): raw[band] for band in FEATURE_BANDS}
        session.band_cache_time = session.sim_time
        session.band_cache = bands
        return bands


def main() -> None:
    app = ClinicalEEGSimulator()
    app.mainloop()


if __name__ == "__main__":
    main()
