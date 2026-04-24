"""
Week 1 — Setup & Resource Baseline
===================================
Tujuan:
  - Verifikasi semua model bisa di-load dan generate teks
  - Catat resource usage (RAM, load time, throughput) sebagai angka baseline
  - Simpan ke results/metrics/baseline_metrics.csv

Jalankan:
  python src/baseline.py
"""

import sys
from pathlib import Path
import pandas as pd

# Pastikan src/ ada di path
sys.path.insert(0, str(Path(__file__).parent))

from utils.metrics import ResultsLogger, get_ram_usage_mb, get_total_ram_mb
from utils.model_utils import (
    build_prompt,
    load_model_and_tokenizer,
    unload_model,
    generate_responses,
)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5",
    "Qwen/Qwen1.5-1.8B-Chat",
]

SYSTEM_PROMPT = "You are a helpful assistant."

# Gunakan pertanyaan sederhana di Week 1 untuk verifikasi saja
# Week 2 nanti pakai TriviaQA/BioASQ yang sebenarnya
TEST_QUESTION = "Explain artificial intelligence in simple terms."

M_BASELINE = 10     # Angka resmi paper — JANGAN kurangi untuk baseline
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.9
TOP_P = 0.95
DEVICE = "cpu"

OUTPUT_CSV = "results/metrics/baseline_metrics.csv"
OUTPUT_RESPONSES = "results/outputs/sample_responses.txt"

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    logger = ResultsLogger(OUTPUT_CSV)
    # Path("results").mkdir(parents=True, exist_ok=True)

    print(f"System RAM total : {get_total_ram_mb():.0f} MB")
    print(f"RAM awal proses  : {get_ram_usage_mb():.1f} MB")
    print(f"M (jumlah sampel): {M_BASELINE}")
    print(f"Device           : {DEVICE}")

    all_responses = {}

    for model_name in MODELS:
        # 1. Load model
        model, tokenizer, load_stats = load_model_and_tokenizer(model_name, DEVICE)

        # 2. Build prompt dengan template yang benar per model
        prompt = build_prompt(
            tokenizer=tokenizer,
            model_name=model_name,
            system=SYSTEM_PROMPT,
            user=TEST_QUESTION,
        )
        print(f"\n[Prompt preview]\n{prompt[:200]}...")

        # 3. Generate M sampel
        responses, gen_stats = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            M=M_BASELINE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        all_responses[model_name] = responses

        # 4. Catat semua metric ke CSV
        row = {
            "model": model_name,
            "M": M_BASELINE,
            "device": DEVICE,
            **load_stats,
            **gen_stats,
        }
        logger.log(row)

        # 5. Unload model sebelum load berikutnya — PENTING untuk CPU
        unload_model(model)

    # 6. Simpan semua respons ke file teks untuk inspeksi manual
    with open(OUTPUT_RESPONSES, "w", encoding="utf-8") as f:
        for model_name, responses in all_responses.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Question: {TEST_QUESTION}\n")
            f.write(f"{'='*60}\n")
            for i, resp in enumerate(responses, 1):
                f.write(f"\n--- Sample {i} ---\n")
                f.write(resp + "\n")

    print(f"\n{'='*50}")
    print("SELESAI. Summary:")
    print(logger.summary().to_string(index=False))
    print(f"\nFile tersimpan:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_RESPONSES}")


if __name__ == "__main__":
    main()