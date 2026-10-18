"""Shared dataset builders for the paper experiments.

Every builder reproduces the corresponding cell of ``train_models.ipynb`` exactly
(same seeds, same order of random draws), so the models saved in ``models/`` can
be re-evaluated on precisely the validation data they were scored on.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LEAF_CLASSES = ["Tomato___Healthy", "Tomato___Late_Blight"]
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TS_FEATURES = ["temperature", "humidity", "co2", "temp_lag_1", "temp_lag_2", "temp_lag_6", "hum_lag_1", "co2_lag_1"]


# --------------------------------------------------------------------------- leaf images (cells 3-4)
@dataclass
class LeafData:
    train_files: list
    val_files: list
    workdir: str


def build_leaf_dataset() -> LeafData:
    workdir = tempfile.mkdtemp(prefix="verdant_leaf_")
    raw = os.path.join(workdir, "plantvillage_raw")
    for cls in LEAF_CLASSES:
        os.makedirs(os.path.join(raw, cls), exist_ok=True)
    size, per_class = 128, 320
    rng = np.random.default_rng(42)

    def synthesize(is_diseased: bool) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.float32)
        if not is_diseased:
            g, r, b = rng.uniform(110, 200), rng.uniform(40, 115), rng.uniform(20, 70)
        else:
            g, r, b = rng.uniform(90, 185), rng.uniform(60, 145), rng.uniform(15, 60)
        img[:, :, 1], img[:, :, 0], img[:, :, 2] = g, r, b
        if is_diseased and rng.random() < 0.65:
            for _ in range(rng.integers(1, 6)):
                cx, cy = rng.integers(10, size - 10, size=2)
                rad = rng.integers(3, 9)
                yy, xx = np.ogrid[-rad:rad, -rad:rad]
                mask = xx**2 + yy**2 <= rad**2
                sub = img[cx - rad:cx + rad, cy - rad:cy + rad, :]
                if sub.shape[:2] == mask.shape:
                    sub[mask, 0] = rng.uniform(40, 90)
                    sub[mask, 1] = rng.uniform(20, 60)
                    sub[mask, 2] = rng.uniform(5, 30)
        elif (not is_diseased) and rng.random() < 0.30:
            for _ in range(rng.integers(1, 3)):
                cx, cy = rng.integers(10, size - 10, size=2)
                rad = rng.integers(2, 5)
                yy, xx = np.ogrid[-rad:rad, -rad:rad]
                mask = xx**2 + yy**2 <= rad**2
                sub = img[cx - rad:cx + rad, cy - rad:cy + rad, :]
                if sub.shape[:2] == mask.shape:
                    sub[mask, 0] += rng.uniform(10, 30)
                    sub[mask, 1] -= rng.uniform(10, 20)
        noise = rng.normal(0, 34, size=img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    for cls in LEAF_CLASSES:
        is_dis = cls == "Tomato___Late_Blight"
        for idx in range(per_class):
            path = os.path.join(raw, cls, f"img_{idx}.jpg")
            if idx in (10, 25, 87) and is_dis:
                open(path, "wb").close()  # corrupted zero-byte file
                continue
            Image.fromarray(synthesize(is_dis)).save(path)

    cleaned = []
    for cls_idx, cls in enumerate(LEAF_CLASSES):
        cls_path = os.path.join(raw, cls)
        for name in os.listdir(cls_path):
            full = os.path.join(cls_path, name)
            if os.path.getsize(full) == 0:
                os.remove(full)
            else:
                cleaned.append((full, cls_idx))

    np.random.seed(7)
    np.random.shuffle(cleaned)
    split = int(0.8 * len(cleaned))
    return LeafData(cleaned[:split], cleaned[split:], workdir)


def load_leaf_arrays(files) -> tuple[np.ndarray, np.ndarray]:
    xs = np.stack([np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p, _ in files])
    ys = np.array([lbl for _, lbl in files])
    return xs, ys


# --------------------------------------------------------------------------- crop recommendation (cells 10-13)
def build_crop_dataset():
    np.random.seed(42)
    labels = ["rice", "maize", "chickpea", "cotton", "coffee"]
    ranges = {
        "rice":     [(60, 105), (25, 60), (25, 60), (20, 30), (65, 90), (5.5, 7.5), (140, 270)],
        "maize":    [(48, 98), (28, 62), (8, 32), (18, 29), (50, 82), (5.3, 7.2), (65, 140)],
        "chickpea": [(18, 48), (45, 82), (62, 98), (13, 25), (10, 32), (6.5, 8.3), (48, 92)],
        "cotton":   [(76, 120), (26, 54), (8, 34), (22, 32), (65, 90), (6.0, 7.6), (65, 112)],
        "coffee":   [(76, 120), (14, 38), (16, 44), (19, 28), (42, 68), (5.5, 7.0), (105, 185)],
    }
    rows = []
    for _ in range(1200):
        target = np.random.choice(labels)
        rows.append([np.random.uniform(lo, hi) for lo, hi in ranges[target]] + [target])
    df = pd.DataFrame(rows, columns=CROP_FEATURES + ["label"])
    sigma = {"N": 9, "P": 8, "K": 8, "temperature": 1.8, "humidity": 6, "ph": 0.35, "rainfall": 18}
    for col, s in sigma.items():
        df[col] = df[col] + np.random.normal(0, s, size=len(df))
    for col in ["N", "temperature", "humidity"]:
        mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
        df.loc[mask, col] = np.nan
    raw = df.copy()
    for col in ["N", "temperature", "humidity"]:
        df[col] = df.groupby("label")[col].transform(lambda x: x.fillna(x.median()))
    for col in CROP_FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    X_tr, X_te, y_tr, y_te = train_test_split(df[CROP_FEATURES], df["label"], test_size=0.2,
                                              stratify=df["label"], random_state=42)
    return raw, df, X_tr, X_te, y_tr, y_te


# --------------------------------------------------------------------------- lettuce growth (cells 17-18)
def build_lettuce_dataset():
    np.random.seed(42)
    n = 600
    dli = np.random.uniform(11.0, 19.0, size=n)
    temp = np.random.uniform(18.0, 24.0, size=n)
    ec = np.random.uniform(0.9, 2.1, size=n)
    ph = np.random.uniform(5.5, 6.5, size=n)
    w_temp = np.random.uniform(17.0, 22.0, size=n)
    days = 45.0 - 0.9 * dli - 0.5 * temp - 1.6 * ec + np.random.normal(0, 1.05, size=n)
    df = pd.DataFrame({"dli": dli, "air_temp": temp, "ec": ec, "ph": ph, "water_temp": w_temp, "growth_days": days})
    X, y = df.drop(columns=["growth_days"]), df["growth_days"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return df, X_tr, X_te, y_tr, y_te


# --------------------------------------------------------------------------- greenhouse sensors (cells 21-22)
def build_sensor_dataset():
    np.random.seed(42)
    ticks = pd.date_range(start="2026-07-01 00:00:00", end="2026-07-11 00:00:00", freq="10min")
    n = len(ticks)
    hr = ticks.hour + ticks.minute / 60.0
    temp = 22.0 + 4.0 * np.sin((hr - 8.0) * (2 * np.pi / 24.0)) + np.random.normal(0, 0.9, size=n)
    hum = 65.0 - 15.0 * np.sin((hr - 8.0) * (2 * np.pi / 24.0)) + np.random.normal(0, 3.2, size=n)
    co2 = 780.0 - 220.0 * np.sin((hr - 6.0) * (2 * np.pi / 24.0)) + np.random.normal(0, 25.0, size=n)
    sensors = pd.DataFrame({"timestamp": ticks, "temperature": temp, "humidity": hum, "co2": co2}).set_index("timestamp")
    d = sensors.copy()
    d["temp_lag_1"] = d["temperature"].shift(1)
    d["temp_lag_2"] = d["temperature"].shift(2)
    d["temp_lag_6"] = d["temperature"].shift(6)
    d["hum_lag_1"] = d["humidity"].shift(1)
    d["co2_lag_1"] = d["co2"].shift(1)
    d["target_temperature"] = d["temperature"].shift(-6)
    d = d.dropna()
    X, y = d[TS_FEATURES], d["target_temperature"]
    split = int(0.8 * len(X))
    return sensors, d, X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def scale(X_tr, X_te):
    s = StandardScaler()
    return s, s.fit_transform(X_tr), s.transform(X_te)
