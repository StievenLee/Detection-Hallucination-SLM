import psutil
import os
import time
import csv
import pandas as pd
from pathlib import Path


def get_ram_usage_mb() -> float:
    """RAM usage proses saat ini dalam MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def get_total_ram_mb() -> float:
    """Total RAM sistem dalam MB."""
    return psutil.virtual_memory().total / 1024 ** 2


def get_available_ram_mb() -> float:
    """RAM tersisa dalam MB."""
    return psutil.virtual_memory().available / 1024 ** 2

def get_model_size_mb(model) -> float:
    """
    Ukur ukuran model langsung dari parameter & buffer.
    Akurat dan tidak terpengaruh OS memory management.
    """
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1024 ** 2

class ResourceMonitor:
    """Monitor RAM usage sebelum & sesudah operasi."""

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
    """Simpan hasil eksperimen ke CSV secara incremental."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def log(self, row: dict):
        self.rows.append(row)
        # Tulis langsung ke disk supaya tidak hilang kalau crash
        df = pd.DataFrame(self.rows)
        df.to_csv(self.output_path, index=False)
        print(f"[Logger] Saved {len(self.rows)} rows → {self.output_path}")

    def summary(self):
        return pd.DataFrame(self.rows)
    
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(correctness: list[int], entropies: list[float]) -> float:
    """
    AUROC: kemampuan SE membedakan benar vs salah.
    SE tinggi → prediksi salah → flip ke -SE.
    """
    if len(set(correctness)) < 2:
        return float("nan")
    return roc_auc_score(correctness, [-e for e in entropies])


def compute_rejection_accuracy(
    correctness: list[int],
    entropies: list[float],
    threshold: float
) -> dict:
    """
    Rejection accuracy pada threshold SE tertentu.
    Soal dengan SE > threshold ditolak (tidak dijawab).

    Return:
      - accuracy_retained : akurasi pada soal yang dijawab
      - coverage          : % soal yang dijawab
      - n_retained        : jumlah soal yang dijawab
      - n_rejected        : jumlah soal yang ditolak
    """
    retained_correct = [
        c for c, e in zip(correctness, entropies) if e <= threshold
    ]
    n_retained = len(retained_correct)
    n_total    = len(correctness)

    if n_retained == 0:
        return {
            "threshold":          threshold,
            "accuracy_retained":  float("nan"),
            "coverage":           0.0,
            "n_retained":         0,
            "n_rejected":         n_total,
        }

    return {
        "threshold":         threshold,
        "accuracy_retained": sum(retained_correct) / n_retained,
        "coverage":          n_retained / n_total,
        "n_retained":        n_retained,
        "n_rejected":        n_total - n_retained,
    }


def compute_aurac(
    correctness: list[int],
    entropies: list[float],
    n_thresholds: int = 100
) -> dict:
    """
    AURAC: Area Under Risk-Coverage Curve.

    Risk     = error rate pada soal yang dijawab = 1 - accuracy_retained
    Coverage = proporsi soal yang dijawab

    Kurva: sumbu X = coverage (0→1), sumbu Y = risk (0→1)
    AURAC = luas area di bawah kurva (lebih RENDAH = lebih baik)

    Catatan konvensi: beberapa paper plot Coverage vs Accuracy
    (AURAC lebih tinggi = lebih baik). Kita pakai Risk-Coverage
    agar konsisten dengan paper Kuhn et al.

    Return:
      - aurac         : float, luas area (trapezoid integration)
      - coverages     : list titik coverage
      - risks         : list titik risk
      - best_threshold: threshold SE yang memberi trade-off terbaik
    """
    # Buat threshold dari min ke max entropy
    min_e = min(entropies)
    max_e = max(entropies)
    thresholds = np.linspace(min_e, max_e, n_thresholds)

    coverages = []
    risks     = []

    for t in thresholds:
        result = compute_rejection_accuracy(correctness, entropies, threshold=t)
        cov  = result["coverage"]
        acc  = result["accuracy_retained"]

        if not np.isnan(acc):
            coverages.append(cov)
            risks.append(1.0 - acc)   # risk = error rate

    if len(coverages) < 2:
        return {"aurac": float("nan"), "coverages": [], "risks": []}

    # Sort by coverage untuk trapezoid integration
    paired = sorted(zip(coverages, risks))
    coverages_sorted = [p[0] for p in paired]
    risks_sorted     = [p[1] for p in paired]

    # Trapezoid rule
    aurac = float(np.trapz(risks_sorted, coverages_sorted))

    # Best threshold = coverage tertinggi dengan risk < 0.3
    best_threshold = None
    for t in reversed(thresholds):
        res = compute_rejection_accuracy(correctness, entropies, threshold=t)
        if not np.isnan(res["accuracy_retained"]):
            if (1 - res["accuracy_retained"]) < 0.3:
                best_threshold = t
                break

    return {
        "aurac":          round(aurac, 4),
        "coverages":      coverages_sorted,
        "risks":          risks_sorted,
        "best_threshold": round(best_threshold, 4) if best_threshold else None,
    }