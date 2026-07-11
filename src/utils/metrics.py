import psutil
import os
import time
import csv
import pandas as pd
from pathlib import Path


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def get_total_ram_mb() -> float:
    return psutil.virtual_memory().total / 1024 ** 2


def get_available_ram_mb() -> float:
    return psutil.virtual_memory().available / 1024 ** 2


def get_model_size_mb(model) -> float:
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1024 ** 2


class ResourceMonitor:
    def __init__(self):
        self.baseline_mb = get_ram_usage_mb()

    def snapshot(self, label: str = "") -> dict:
        current = get_ram_usage_mb()
        return {
            "label": label,
            "ram_used_mb": current,
            "ram_delta_mb": current - self.baseline_mb,
            "ram_available_mb": get_available_ram_mb(),
        }


class ResultsLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def log(self, row: dict):
        self.rows.append(row)
        df = pd.DataFrame(self.rows)
        df.to_csv(self.output_path, index=False)
        print(f"[Logger] Saved {len(self.rows)} rows -> {self.output_path}")

    def summary(self):
        return pd.DataFrame(self.rows)


import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(correctness: list, entropies: list) -> float:
    if len(set(correctness)) < 2:
        return float("nan")
    return roc_auc_score(correctness, [-e for e in entropies])


def compute_rejection_accuracy(correctness, entropies, threshold):
    retained_correct = [c for c, e in zip(correctness, entropies) if e <= threshold]
    n_retained = len(retained_correct)
    n_total = len(correctness)
    if n_retained == 0:
        return {"threshold": threshold, "accuracy_retained": float("nan"),
                "coverage": 0.0, "n_retained": 0, "n_rejected": n_total}
    return {"threshold": threshold,
            "accuracy_retained": sum(retained_correct) / n_retained,
            "coverage": n_retained / n_total,
            "n_retained": n_retained,
            "n_rejected": n_total - n_retained}


def compute_aurac(correctness, entropies, n_thresholds: int = 100):
    min_e = min(entropies); max_e = max(entropies)
    thresholds = np.linspace(min_e, max_e, n_thresholds)
    coverages, accuracies = [], []
    for t in thresholds:
        result = compute_rejection_accuracy(correctness, entropies, threshold=t)
        cov = result["coverage"]; acc = result["accuracy_retained"]
        if not np.isnan(acc):
            coverages.append(cov); accuracies.append(acc)
    if len(coverages) < 2:
        return {"aurac": float("nan"), "coverages": [], "accuracies": [], "best_threshold": None}
    paired = sorted(zip(coverages, accuracies))
    coverages_sorted = [p[0] for p in paired]
    accuracies_sorted = [p[1] for p in paired]
    aurac = float(np.trapezoid(accuracies_sorted, coverages_sorted))
    best_threshold = None
    for t in reversed(thresholds):
        res = compute_rejection_accuracy(correctness, entropies, threshold=t)
        if not np.isnan(res["accuracy_retained"]):
            if (1 - res["accuracy_retained"]) < 0.3:
                best_threshold = t
                break
    return {"aurac": round(aurac, 4),
            "coverages": coverages_sorted,
            "accuracies": accuracies_sorted,
            "best_threshold": round(best_threshold, 4) if best_threshold else None}
