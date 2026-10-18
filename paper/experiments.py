"""Experiments reported in the VERDANT research paper.

    python paper/experiments.py verify      # re-score the saved models in models/
    python paper/experiments.py baselines   # baseline comparison for the tabular tasks
    python paper/experiments.py ablation    # CNN ablation study (retrains, ~15 runs)
    python paper/experiments.py all

Results are written to paper/results/*.json and consumed by figures.py.
Saved models are only read, never overwritten.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def save(name, obj):
    with open(os.path.join(C.RESULTS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  -> results/{name}.json")


def load_pkl(name):
    with open(os.path.join(C.MODEL_DIR, name), "rb") as f:
        return pickle.load(f)


def latency_ms(fn, n=200):
    for _ in range(10):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def reg_metrics(y, p):
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": r2_score(y, p)}


# --------------------------------------------------------------------------- CNN definition (cell 5)
class FoliarDiseaseCNN(nn.Module):
    def __init__(self, batchnorm=True, dropout=0.4):
        super().__init__()
        layers, ch = [], 3
        for out in (16, 32, 64, 128):
            layers += [nn.Conv2d(ch, out, 3, padding=1)]
            if batchnorm:
                layers += [nn.BatchNorm2d(out)]
            layers += [nn.ReLU(), nn.MaxPool2d(2, 2)]
            ch = out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def leaf_tensors():
    leaf = C.build_leaf_dataset()
    Xtr, ytr = C.load_leaf_arrays(leaf.train_files)
    Xva, yva = C.load_leaf_arrays(leaf.val_files)
    return Xtr, ytr, Xva, yva


# --------------------------------------------------------------------------- 1. verify saved models
def verify():
    print("== verify saved models")
    out = {}

    # CNN
    Xtr, ytr, Xva, yva = leaf_tensors()
    model = FoliarDiseaseCNN()
    model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, "disease_classifier.pth"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva).permute(0, 3, 1, 2))
        prob = torch.softmax(logits, 1)[:, 1].numpy()
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yva, prob)
    p, r, f, s = precision_recall_fscore_support(yva, pred)
    x1 = torch.tensor(Xva[:1]).permute(0, 3, 1, 2)
    with torch.no_grad():
        lat = latency_ms(lambda: model(x1))
    out["cnn"] = {"acc": accuracy_score(yva, pred), "cm": confusion_matrix(yva, pred).tolist(),
                  "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist(),
                  "f1_weighted": f1_score(yva, pred, average="weighted"),
                  "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(), "auc": auc(fpr, tpr),
                  "params": sum(q.numel() for q in model.parameters()), "latency_ms": lat,
                  "size_mb": os.path.getsize(os.path.join(C.MODEL_DIR, "disease_classifier.pth")) / 1e6,
                  "n_train": len(ytr), "n_val": len(yva)}
    print(f"  CNN acc={out['cnn']['acc']:.4f} auc={out['cnn']['auc']:.4f} cm={out['cnn']['cm']}")

    # crop recommender
    raw, df, X_tr, X_te, y_tr, y_te = C.build_crop_dataset()
    b = load_pkl("crop_recommender.pkl")
    m, sc = b["model"], b["scaler"]
    pv, pt = m.predict(sc.transform(X_te)), m.predict(sc.transform(X_tr))
    labels = list(m.classes_)
    p, r, f, s = precision_recall_fscore_support(y_te, pv, labels=labels)
    xs = sc.transform(X_te.iloc[:1])
    out["crop"] = {"train_acc": accuracy_score(y_tr, pt), "acc": accuracy_score(y_te, pv),
                   "f1_weighted": f1_score(y_te, pv, average="weighted"), "f1_macro": f1_score(y_te, pv, average="macro"),
                   "labels": labels, "cm": confusion_matrix(y_te, pv, labels=labels).tolist(),
                   "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist(),
                   "importance": dict(zip(C.CROP_FEATURES, m.feature_importances_.tolist())),
                   "corr": df[C.CROP_FEATURES].corr().values.tolist(),
                   "missing": raw[C.CROP_FEATURES].isnull().sum().to_dict(),
                   "nodes": int(sum(t.tree_.node_count for t in m.estimators_)),
                   "latency_ms": latency_ms(lambda: m.predict(xs), n=100),
                   "size_mb": os.path.getsize(os.path.join(C.MODEL_DIR, "crop_recommender.pkl")) / 1e6}
    print(f"  crop train={out['crop']['train_acc']:.4f} val={out['crop']['acc']:.4f}")

    # lettuce
    dfl, X_tr, X_te, y_tr, y_te = C.build_lettuce_dataset()
    b = load_pkl("lettuce_predictor.pkl")
    m, sc = b["model"], b["scaler"]
    pv, pt = m.predict(sc.transform(X_te)), m.predict(sc.transform(X_tr))
    xs = sc.transform(X_te.iloc[:1])
    out["lettuce"] = {**reg_metrics(y_te, pv), "train_r2": r2_score(y_tr, pt),
                      "y_true": y_te.tolist(), "y_pred": pv.tolist(), "all_days": dfl["growth_days"].tolist(),
                      "coef": dict(zip(X_tr.columns, m.coef_.tolist())), "intercept": float(m.intercept_),
                      "latency_ms": latency_ms(lambda: m.predict(xs)),
                      "size_mb": os.path.getsize(os.path.join(C.MODEL_DIR, "lettuce_predictor.pkl")) / 1e6}
    print(f"  lettuce r2={out['lettuce']['r2']:.4f} mae={out['lettuce']['mae']:.4f}")

    # forecaster
    sensors, d, X_tr, X_te, y_tr, y_te = C.build_sensor_dataset()
    b = load_pkl("greenhouse_forecaster.pkl")
    m, sc = b["model"], b["scaler"]
    pv, pt = m.predict(sc.transform(X_te)), m.predict(sc.transform(X_tr))
    xs = sc.transform(X_te.iloc[:1])
    out["forecast"] = {**reg_metrics(y_te, pv), "train_r2": r2_score(y_tr, pt),
                       "index": [str(t) for t in y_te.index], "y_true": y_te.tolist(), "y_pred": pv.tolist(),
                       "importance": dict(zip(C.TS_FEATURES, m.feature_importances_.tolist())),
                       "sensors": {"index": [str(t) for t in sensors.index], **{c: sensors[c].tolist() for c in sensors.columns}},
                       "nodes": int(sum(t.tree_.node_count for t in m.estimators_)),
                       "latency_ms": latency_ms(lambda: m.predict(xs), n=100),
                       "size_mb": os.path.getsize(os.path.join(C.MODEL_DIR, "greenhouse_forecaster.pkl")) / 1e6}
    print(f"  forecast r2={out['forecast']['r2']:.4f} rmse={out['forecast']['rmse']:.4f}")
    save("verify", out)


# --------------------------------------------------------------------------- 2. baselines
def colour_hist(X):
    feats = []
    for img in X:
        h = [np.histogram(img[:, :, c], bins=16, range=(0, 1))[0] / img[:, :, c].size for c in range(3)]
        feats.append(np.concatenate(h + [img.mean((0, 1)), img.std((0, 1))]))
    return np.array(feats)


def baselines():
    print("== baselines")
    out = {}

    # leaf: classical colour-histogram pipelines vs CNN
    Xtr, ytr, Xva, yva = leaf_tensors()
    Ftr, Fva = colour_hist(Xtr), colour_hist(Xva)
    leaf = {}
    for name, est in [("Colour hist. + LogReg", LogisticRegression(max_iter=2000)),
                      ("Colour hist. + SVM (RBF)", SVC()),
                      ("Colour hist. + RF", RandomForestClassifier(n_estimators=200, random_state=42))]:
        pipe = make_pipeline(StandardScaler(), est).fit(Ftr, ytr)
        pv = pipe.predict(Fva)
        leaf[name] = {"acc": accuracy_score(yva, pv), "f1": f1_score(yva, pv, average="weighted")}
        print(f"  leaf {name}: {leaf[name]['acc']:.4f}")
    out["leaf"] = leaf

    # crop recommendation
    raw, df, X_tr, X_te, y_tr, y_te = C.build_crop_dataset()
    sc, Xtr_s, Xte_s = C.scale(X_tr, X_te)
    crop = {}
    cands = [("Logistic Regression", LogisticRegression(max_iter=2000)),
             ("k-NN (k = 5)", KNeighborsClassifier(5)),
             ("SVM (RBF)", SVC()),
             ("Decision Tree (depth 7)", DecisionTreeClassifier(max_depth=7, random_state=42)),
             ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
             ("Random Forest (unconstrained)", RandomForestClassifier(n_estimators=150, random_state=42)),
             ("Random Forest (ours)", RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=3, random_state=42))]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, est in cands:
        est.fit(Xtr_s, y_tr)
        cvs = cross_val_score(make_pipeline(StandardScaler(), est), X_tr, y_tr, cv=cv)
        pv = est.predict(Xte_s)
        crop[name] = {"train_acc": accuracy_score(y_tr, est.predict(Xtr_s)), "acc": accuracy_score(y_te, pv),
                      "f1_macro": f1_score(y_te, pv, average="macro"), "cv_mean": cvs.mean(), "cv_std": cvs.std()}
        print(f"  crop {name}: train={crop[name]['train_acc']:.4f} val={crop[name]['acc']:.4f} cv={cvs.mean():.4f}±{cvs.std():.4f}")
    out["crop"] = crop

    # depth sweep (bias-variance)
    sweep = []
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]:
        est = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=3 if depth else 1, random_state=42)
        est.fit(Xtr_s, y_tr)
        sweep.append({"depth": depth or 0, "train": accuracy_score(y_tr, est.predict(Xtr_s)), "val": accuracy_score(y_te, est.predict(Xte_s))})
    out["crop_depth_sweep"] = sweep

    # lettuce growth
    dfl, X_tr, X_te, y_tr, y_te = C.build_lettuce_dataset()
    sc, Xtr_s, Xte_s = C.scale(X_tr, X_te)
    let = {}
    kcv = KFold(5, shuffle=True, random_state=42)
    for name, est in [("Mean predictor", DummyRegressor()),
                      ("Linear Regression (OLS)", LinearRegression()),
                      ("Lasso (alpha = 0.05)", Lasso(alpha=0.05)),
                      ("k-NN (k = 5)", KNeighborsRegressor(5)),
                      ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42)),
                      ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
                      ("Ridge (ours)", Ridge(alpha=1.5))]:
        est.fit(Xtr_s, y_tr)
        cvs = cross_val_score(make_pipeline(StandardScaler(), est), X_tr, y_tr, cv=kcv, scoring="r2")
        let[name] = {**reg_metrics(y_te, est.predict(Xte_s)), "cv_r2_mean": cvs.mean(), "cv_r2_std": cvs.std()}
        print(f"  lettuce {name}: r2={let[name]['r2']:.4f} mae={let[name]['mae']:.4f}")
    out["lettuce"] = let
    alphas = np.logspace(-3, 3, 25)
    out["ridge_alpha_sweep"] = [{"alpha": a, "r2": r2_score(y_te, Ridge(alpha=a).fit(Xtr_s, y_tr).predict(Xte_s))} for a in alphas]

    # greenhouse forecaster
    sensors, d, X_tr, X_te, y_tr, y_te = C.build_sensor_dataset()
    sc, Xtr_s, Xte_s = C.scale(X_tr, X_te)
    fc = {}
    fc["Persistence (T_t)"] = reg_metrics(y_te, X_te["temperature"].values)
    seasonal = sensors["temperature"].shift(138).reindex(y_te.index)  # same clock time one day earlier
    fc["Seasonal naive (24 h)"] = reg_metrics(y_te, seasonal.values)
    no_co2 = [c for c in C.TS_FEATURES if "co2" not in c]
    sc2 = StandardScaler().fit(X_tr[no_co2])
    for name, est, cols in [("Linear Regression", LinearRegression(), None),
                            ("Gradient Boosting", GradientBoostingRegressor(random_state=42), None),
                            ("Random Forest w/o CO2 lags", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42), no_co2),
                            ("Random Forest (ours)", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42), None)]:
        if cols:
            est.fit(sc2.transform(X_tr[cols]), y_tr)
            pv = est.predict(sc2.transform(X_te[cols]))
        else:
            est.fit(Xtr_s, y_tr)
            pv = est.predict(Xte_s)
        fc[name] = reg_metrics(y_te, pv)
    for k, v in fc.items():
        print(f"  forecast {k}: rmse={v['rmse']:.4f} r2={v['r2']:.4f}")
    out["forecast"] = fc
    save("baselines", out)


# --------------------------------------------------------------------------- 3. CNN ablation
def train_cnn(Xtr, ytr, Xva, yva, seed, augment=True, batchnorm=True, dropout=0.4, epochs=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FoliarDiseaseCNN(batchnorm=batchnorm, dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xva_t = torch.tensor(Xva).permute(0, 3, 1, 2)
    hist = []
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(Xtr))
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            xb = Xtr[idx].copy()
            if augment:
                for j in range(len(xb)):
                    if np.random.rand() < 0.5:
                        xb[j] = xb[j][:, ::-1, :]
                    if np.random.rand() < 0.5:
                        xb[j] = xb[j][::-1, :, :]
                    xb[j] = np.clip(xb[j] * np.random.uniform(0.85, 1.15), 0, 1)
            xb = torch.tensor(np.ascontiguousarray(xb)).permute(0, 3, 1, 2)
            opt.zero_grad()
            loss = crit(model(xb), torch.tensor(ytr[idx]))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist.append(float((model(Xva_t).argmax(1).numpy() == yva).mean()))
        model.train()
    model.eval()
    with torch.no_grad():
        tr_acc = float((model(torch.tensor(Xtr).permute(0, 3, 1, 2)).argmax(1).numpy() == ytr).mean())
    return {"final_val": hist[-1], "best_val": max(hist), "train": tr_acc, "curve": hist}


def ablation():
    print("== CNN ablation")
    Xtr, ytr, Xva, yva = leaf_tensors()
    variants = {"Full model (ours)": {}, "w/o augmentation": {"augment": False}, "w/o dropout": {"dropout": 0.0},
                "w/o batch normalisation": {"batchnorm": False}, "w/o augmentation & dropout": {"augment": False, "dropout": 0.0}}
    out = {}
    for name, kw in variants.items():
        runs = []
        for seed in (0, 1, 2):
            t0 = time.time()
            runs.append(train_cnn(Xtr, ytr, Xva, yva, seed, **kw))
            print(f"  {name} seed {seed}: val={runs[-1]['final_val']:.4f} train={runs[-1]['train']:.4f} ({time.time()-t0:.0f}s)")
        fv = [r["final_val"] for r in runs]
        tr = [r["train"] for r in runs]
        out[name] = {"val_mean": float(np.mean(fv)), "val_std": float(np.std(fv)), "train_mean": float(np.mean(tr)),
                     "train_std": float(np.std(tr)), "best_mean": float(np.mean([r["best_val"] for r in runs])), "runs": runs}
        save("ablation", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("verify", "all"):
        verify()
    if what in ("baselines", "all"):
        baselines()
    if what in ("ablation", "all"):
        ablation()
