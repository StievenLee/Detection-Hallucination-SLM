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