from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
EPS = 1e-12
MAX_POINTS = 3000
INLINE_RE = re.compile(r"Subject(?P<subject>\d+)_(?P<peep>[048])cmH2O_(?P<label>normal|apnea1|apnea2)\.csv$")
PULSE_RE = re.compile(r"Subject(?P<subject>\d+)_(?P<peep>[048])cmH2O_(?P<label>normal|apnea1|apnea2)_pulse\.csv$")

WINDOWS: List[Tuple[str, float, float]] = [
    ("w00_10", 0.0, 10.0),
    ("w10_25", 10.0, 25.0),
    ("w25_40", 25.0, 40.0),
    ("w40_60", 40.0, 60.0),
    ("w00_30", 0.0, 30.0),
    ("full", 0.0, 60.0),
]


@dataclass(frozen=True)
class TrialKey:
    subject: int
    peep: int
    label: str


def parse_key(filename: str, sensor: str) -> Optional[TrialKey]:
    base = os.path.basename(filename)
    pat = INLINE_RE if sensor == "inline" else PULSE_RE
    m = pat.match(base)
    if not m:
        return None
    return TrialKey(int(m.group("subject")), int(m.group("peep")), m.group("label"))


def list_csv_members(zip_path: str, sensor: str) -> Dict[TrialKey, str]:
    out: Dict[TrialKey, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith(".csv"):
                continue
            key = parse_key(n, sensor)
            if key is not None:
                out[key] = n
    return out


def read_member(z: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with z.open(member) as fh:
        return pd.read_csv(fh)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    time_col = df.columns[0]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[time_col])
    if len(df) < 2:
        return df
    t = df[time_col].to_numpy(float)
    jumps = np.where(np.diff(t) < -0.05)[0]
    if len(jumps):
        df = df.iloc[: jumps[0] + 1].copy()
    df = df[df[time_col] >= 0].copy()
    df = df.drop_duplicates(subset=[time_col], keep="first")
    if len(df):
        df[time_col] = df[time_col] - float(df[time_col].iloc[0])
    return df.reset_index(drop=True)


def fs_from_time(t: np.ndarray) -> float:
    if t.size < 3:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else np.nan


def thin(t: np.ndarray, x: np.ndarray, max_points: int = MAX_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        return t[idx], x[idx]
    return t, x


def robust_stats(prefix: str, t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    t, x = thin(t, x)
    f: Dict[str, float] = {f"{prefix}__n": float(x.size)}
    if x.size < 3:
        for k in ["mean", "std", "median", "p05", "p95", "iqr", "range", "rms", "mad", "diff_abs_mean", "diff_std", "slope"]:
            f[f"{prefix}__{k}"] = np.nan
        return f
    q05, q25, q50, q75, q95 = np.nanquantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    dx = np.diff(x)
    f.update({
        f"{prefix}__mean": float(np.nanmean(x)),
        f"{prefix}__std": float(np.nanstd(x)),
        f"{prefix}__median": float(q50),
        f"{prefix}__p05": float(q05),
        f"{prefix}__p95": float(q95),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__range": float(q95 - q05),
        f"{prefix}__rms": float(np.sqrt(np.nanmean(x * x))),
        f"{prefix}__mad": float(np.nanmedian(np.abs(x - q50))),
        f"{prefix}__diff_abs_mean": float(np.nanmean(np.abs(dx))) if dx.size else np.nan,
        f"{prefix}__diff_std": float(np.nanstd(dx)) if dx.size else np.nan,
    })
    tt = t - t[0]
    if np.nanstd(tt) > EPS and np.nanstd(x) > EPS:
        try:
            f[f"{prefix}__slope"] = float(np.polyfit(tt, x, 1)[0])
        except Exception:
            f[f"{prefix}__slope"] = np.nan
    else:
        f[f"{prefix}__slope"] = 0.0
    return f


def fft_features(prefix: str, t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    t, x = thin(t, x, max_points=2048)
    out = {f"{prefix}__fft_dom_freq": np.nan, f"{prefix}__fft_resp_frac": np.nan, f"{prefix}__fft_heart_frac": np.nan}
    if x.size < 64 or np.nanstd(x) < EPS:
        return {k: 0.0 for k in out}
    fs = fs_from_time(t)
    if not np.isfinite(fs) or fs <= 0:
        return out
    x = x - np.nanmean(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    valid = (freqs >= 0.05) & (freqs <= min(8.0, fs / 2.0))
    if not valid.any() or np.nansum(spec[valid]) <= EPS:
        return {k: 0.0 for k in out}
    dom_idx = np.argmax(spec[valid])
    total = float(np.nansum(spec[valid]))
    resp = (freqs >= 0.05) & (freqs < 0.70)
    heart = (freqs >= 0.70) & (freqs < 3.00)
    out[f"{prefix}__fft_dom_freq"] = float(freqs[valid][dom_idx])
    out[f"{prefix}__fft_resp_frac"] = float(np.nansum(spec[resp]) / (total + EPS))
    out[f"{prefix}__fft_heart_frac"] = float(np.nansum(spec[heart]) / (total + EPS))
    return out


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.nanstd(a[m]) < EPS or np.nanstd(b[m]) < EPS:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def sensor_features(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    df = clean_df(df)
    feats: Dict[str, float] = {}
    if df.empty or df.shape[1] < 2:
        return feats
    time_col = df.columns[0]
    t_all = df[time_col].to_numpy(float)
    feats[f"{prefix}__duration"] = float(np.nanmax(t_all) - np.nanmin(t_all))
    feats[f"{prefix}__fs"] = fs_from_time(t_all)
    feats[f"{prefix}__samples"] = float(len(df))
    channels = [c for c in df.columns if c != time_col]

    for win, start, end in WINDOWS:
        mask = (t_all >= start) & (t_all < end)
        t = t_all[mask]
        feats[f"{prefix}__{win}__samples"] = float(mask.sum())
        for ch in channels:
            x = df.loc[mask, ch].to_numpy(float)
            safe = re.sub(r"[^0-9A-Za-z]+", "_", ch).strip("_")
            base = f"{prefix}__{win}__{safe}"
            feats.update(robust_stats(base, t, x))
            if win in {"w00_30", "full"}:
                feats.update(fft_features(base, t, x))

        if prefix == "inline" and {"Gauge Pressure [cmH2O]", "Inspiratory differential pressure [cmH2O]"}.issubset(df.columns):
            gp = df.loc[mask, "Gauge Pressure [cmH2O]"].to_numpy(float)
            dp = df.loc[mask, "Inspiratory differential pressure [cmH2O]"].to_numpy(float)
            feats[f"inline__{win}__gp_dp_corr"] = corr(gp, dp)
            feats[f"inline__{win}__diffp_abs_mean"] = float(np.nanmean(np.abs(dp))) if dp.size else np.nan
            feats[f"inline__{win}__diffp_quiet_frac"] = float(np.nanmean(np.abs(dp) < 0.05)) if dp.size else np.nan

        if prefix == "pulse":
            red_cols = [c for c in ["PD1", "PD2", "PD3", "PD4"] if c in df.columns]
            ir_cols = [c for c in ["PD1_9", "PD2_9", "PD3_9", "PD4_9"] if c in df.columns]
            if red_cols and ir_cols:
                red = df.loc[mask, red_cols].mean(axis=1).to_numpy(float)
                ir = df.loc[mask, ir_cols].mean(axis=1).to_numpy(float)
                ratio = red / (ir + EPS)
                feats.update(robust_stats(f"pulse__{win}__red_mean", t, red))
                feats.update(robust_stats(f"pulse__{win}__ir_mean", t, ir))
                feats.update(robust_stats(f"pulse__{win}__red_ir_ratio", t, ratio))
                feats[f"pulse__{win}__red_ir_corr"] = corr(red, ir)

    for ch in channels:
        safe = re.sub(r"[^0-9A-Za-z]+", "_", ch).strip("_")
        early = df.loc[(t_all >= 10) & (t_all < 25), ch].to_numpy(float)
        late = df.loc[(t_all >= 40) & (t_all < 60), ch].to_numpy(float)
        e_std = float(np.nanstd(early)) if early.size else np.nan
        l_std = float(np.nanstd(late)) if late.size else np.nan
        e_abs = float(np.nanmean(np.abs(early))) if early.size else np.nan
        l_abs = float(np.nanmean(np.abs(late))) if late.size else np.nan
        feats[f"{prefix}__contrast__{safe}__std_early_late_ratio"] = e_std / (l_std + EPS)
        feats[f"{prefix}__contrast__{safe}__abs_early_late_ratio"] = e_abs / (l_abs + EPS)
    return feats


def build_features(inline_zip: str, pulse_zip: str, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv)
    inline_members = list_csv_members(inline_zip, "inline")
    pulse_members = list_csv_members(pulse_zip, "pulse")
    keys = sorted(set(inline_members) & set(pulse_members), key=lambda k: (k.subject, k.peep, k.label))
    if not keys:
        raise RuntimeError("No matched inline/pulse files found.")
    rows: List[Dict[str, object]] = []
    with zipfile.ZipFile(inline_zip) as zi, zipfile.ZipFile(pulse_zip) as zp:
        for i, key in enumerate(keys, 1):
            row: Dict[str, object] = {
                "subject": key.subject,
                "peep_cmH2O": key.peep,
                "label_multiclass": key.label,
                "label_binary": "normal" if key.label == "normal" else "apnea",
            }
            row.update(sensor_features(read_member(zi, inline_members[key]), "inline"))
            row.update(sensor_features(read_member(zp, pulse_members[key]), "pulse"))
            rows.append(row)
            if i % 20 == 0 or i == len(keys):
                print(f"Extracted features for {i}/{len(keys)} trials", flush=True)
    df = pd.DataFrame(rows).sort_values(["subject", "peep_cmH2O", "label_multiclass"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def get_feature_cols(df: pd.DataFrame, include_peep: bool) -> List[str]:
    excluded = {"subject", "label_binary", "label_multiclass"}
    if not include_peep:
        excluded.add("peep_cmH2O")
    cols: List[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 0 and v.nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def make_model(n_estimators: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def aligned_proba(model: Pipeline, X: pd.DataFrame, labels: List[str]) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_)
    out = np.zeros((len(X), len(labels)))
    for j, lab in enumerate(classes):
        if lab in labels:
            out[:, labels.index(lab)] = proba[:, j]
    return out


def loso_eval(df: pd.DataFrame, target: str, name: str, include_peep: bool, model: Pipeline, out_dir: Path) -> Dict[str, object]:
    X_cols = get_feature_cols(df, include_peep)
    X = df[X_cols]
    y = df[target].astype(str).to_numpy()
    groups = df["subject"].to_numpy()
    labels = sorted(np.unique(y).tolist())
    logo = LeaveOneGroupOut()
    rows, probs = [], []
    y_true, y_pred, subj = [], [], []
    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        clf = clone(model)
        clf.fit(X.iloc[tr], y[tr])
        pred = clf.predict(X.iloc[te])
        y_true.extend(y[te].tolist()); y_pred.extend(pred.tolist()); subj.extend(groups[te].tolist())
        probs.append(aligned_proba(clf, X.iloc[te], labels))
        rows.append({
            "fold": fold,
            "held_out_subject": int(groups[te][0]),
            "accuracy": accuracy_score(y[te], pred),
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
            "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
        })
    proba = np.vstack(probs)
    pred_df = pd.DataFrame({"subject": subj, "y_true": y_true, "y_pred": y_pred})
    for j, lab in enumerate(labels):
        pred_df[f"proba_{lab}"] = proba[:, j]
    pred_df.to_csv(out_dir / f"predictions_{name}.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / f"fold_metrics_{name}.csv", index=False)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(out_dir / f"confusion_matrix_{name}.csv")
    metrics: Dict[str, object] = {
        "task": name,
        "target": target,
        "n_trials": int(len(df)),
        "n_subjects": int(len(np.unique(groups))),
        "labels": labels,
        "n_features": int(len(X_cols)),
        "include_peep": bool(include_peep),
        "model": "ExtraTreesClassifier",
        "evaluation": "Leave-One-Subject-Out CV",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }
    try:
        if len(labels) == 2:
            pos = "apnea" if "apnea" in labels else labels[1]
            metrics["roc_auc"] = float(roc_auc_score(np.asarray(y_true) == pos, proba[:, labels.index(pos)]))
        else:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, labels=labels, multi_class="ovr", average="macro"))
    except Exception as e:
        metrics["roc_auc_error"] = str(e)
    with open(out_dir / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    final = clone(model)
    final.fit(X, y)
    joblib.dump({"model": final, "feature_columns": X_cols, "labels": labels, "target": target}, out_dir / f"model_{name}.joblib")
    imp = pd.DataFrame({"feature": X_cols, "importance": final.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / f"feature_importance_{name}.csv", index=False)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inline-zip", required=True)
    ap.add_argument("--pulse-zip", required=True)
    ap.add_argument("--out-dir", default="resp_oximetry_baseline_results")
    ap.add_argument("--include-peep", action="store_true", help="Optionally use PEEP metadata as an input feature")
    ap.add_argument("--n-estimators", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_features(args.inline_zip, args.pulse_zip, out / "features.csv")
    print(f"Feature table: {df.shape}")
    print("Multiclass counts:", df["label_multiclass"].value_counts().to_dict())
    print("Binary counts:", df["label_binary"].value_counts().to_dict())
    model = make_model(args.n_estimators)
    summary: Dict[str, object] = {}
    for target, name in [("label_binary", "binary"), ("label_multiclass", "multiclass")]:
        print(f"Evaluating {name}...", flush=True)
        summary[name] = loso_eval(df, target, name, args.include_peep, model, out)
        small = {k: summary[name][k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"] if k in summary[name]}
        print(name, small)
    with open(out / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
