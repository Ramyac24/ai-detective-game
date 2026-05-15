"""
data_generator.py
Generates synthetic alien signal datasets for the ML Mystery Game.
Creates realistic signal data with embedded mystery patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducible seed ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)


# ── Signal class definitions ───────────────────────────────────────────────────
SIGNAL_CLASSES = ["Beacon", "Warning", "DataTransmission", "Anomaly", "Interference"]

MODULATION_TYPES = {
    "Beacon":          ["PSK", "FSK"],
    "Warning":         ["AM", "FM"],
    "DataTransmission":["QAM", "PSK", "OFDM"],
    "Anomaly":         ["Unknown", "QAM"],
    "Interference":    ["AM", "FM", "Noise"],
}

SECTOR_CENTERS = {
    "Beacon":          (120, 340),
    "Warning":         (280, 180),
    "DataTransmission":(450, 420),
    "Anomaly":         (200, 250),   # Anomalies cluster tightly → the hidden pattern
    "Interference":    (rng.integers(50, 500), rng.integers(50, 500)),
}

MYSTERY_COORDS = [(195, 242), (198, 255), (203, 248),   # The hidden cluster
                  (207, 260), (201, 245), (210, 252)]


def _signal_row(signal_id: int, signal_class: str, base_time: datetime) -> dict:
    cx, cy = SECTOR_CENTERS[signal_class]
    spread = 15 if signal_class == "Anomaly" else 60

    sector_x = float(np.clip(rng.normal(cx, spread), 0, 500))
    sector_y = float(np.clip(rng.normal(cy, spread), 0, 500))

    # Frequency ranges per class
    freq_ranges = {
        "Beacon":          (200,  800),
        "Warning":         (800, 1500),
        "DataTransmission":(1500, 4000),
        "Anomaly":         (400,  900),
        "Interference":    (50,  5000),
    }
    fmin, fmax = freq_ranges[signal_class]
    frequency = float(rng.uniform(fmin, fmax))

    amplitude = float(rng.uniform(-115, -45))
    duration  = float(np.clip(rng.exponential(30), 0.5, 300))
    pulse_rate = float(rng.uniform(0, 50))
    noise_ratio = float(np.clip(rng.beta(2, 5), 0, 1))

    modulation = random.choice(MODULATION_TYPES[signal_class])

    # Embed clue fragments in Anomaly signal metadata
    clue_fragment = ""
    if signal_class == "Anomaly":
        fragments = [
            "HEADER::SYS-7749::ORIGIN_UNKNOWN",
            "PAYLOAD::COORD_ENCODED::THETA=0.87",
            "SIGNATURE::NON-TERRESTRIAL::CONFIRMED",
            "FREQ_PATTERN::FIBONACCI_SERIES::DETECTED",
            "TIMESTAMP_DELTA::PRIME_SEQUENCE::ACTIVE",
            "ENCRYPTION::UNKNOWN_CIPHER::LAYER_3",
        ]
        clue_fragment = random.choice(fragments)

    offset_hours = rng.integers(0, 720)
    timestamp = base_time + timedelta(hours=int(offset_hours),
                                      minutes=int(rng.integers(0, 60)))

    return {
        "signal_id":      f"SIG-{signal_id:05d}",
        "timestamp":      timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "frequency_mhz":  round(frequency, 3),
        "amplitude_db":   round(amplitude, 2),
        "duration_sec":   round(duration, 2),
        "pulse_rate_hz":  round(pulse_rate, 3),
        "noise_ratio":    round(noise_ratio, 4),
        "modulation":     modulation,
        "sector_x":       round(sector_x, 2),
        "sector_y":       round(sector_y, 2),
        "signal_class":   signal_class,
        "clue_fragment":  clue_fragment,
    }


def generate_dataset(n_samples: int = 500, output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic alien signal dataset.

    Parameters
    ----------
    n_samples : int
        Total number of signals to generate.
    output_path : str, optional
        If provided, saves the CSV to this path.

    Returns
    -------
    pd.DataFrame
    """
    base_time = datetime(2031, 3, 14, 0, 0, 0)

    # Class distribution (Anomaly intentionally rare → harder to detect)
    class_weights = {
        "Beacon":          0.30,
        "Warning":         0.20,
        "DataTransmission":0.25,
        "Anomaly":         0.12,
        "Interference":    0.13,
    }
    classes = list(class_weights.keys())
    weights = list(class_weights.values())

    records = []
    for i in range(n_samples):
        sc = random.choices(classes, weights=weights, k=1)[0]
        records.append(_signal_row(i + 1, sc, base_time))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Saved {len(df)} signals → {output_path}")

    return df


def load_or_generate(data_dir: str = "data") -> pd.DataFrame:
    """Load cached CSV or generate a fresh dataset."""
    csv_path = os.path.join(data_dir, "signals_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df
    return generate_dataset(n_samples=500, output_path=csv_path)


# ── Feature engineering helpers ────────────────────────────────────────────────
FEATURE_COLS = [
    "frequency_mhz", "amplitude_db", "duration_sec",
    "pulse_rate_hz", "noise_ratio", "sector_x", "sector_y",
]

MODULATION_MAP = {m: i for i, m in enumerate(
    ["AM", "FM", "PSK", "FSK", "QAM", "OFDM", "Unknown", "Noise"]
)}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with modulation one-hot encoded for ML use."""
    df2 = df.copy()
    df2["modulation_enc"] = df2["modulation"].map(MODULATION_MAP).fillna(0).astype(int)
    return df2


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """Return (X, y) for ML training."""
    df2 = encode_features(df)
    feature_cols = FEATURE_COLS + ["modulation_enc"]
    X = df2[feature_cols].values
    y = df2["signal_class"].values
    return X, y, feature_cols


if __name__ == "__main__":
    df = generate_dataset(n_samples=500, output_path="data/signals_dataset.csv")
    print(df["signal_class"].value_counts())
    print(df.head())
