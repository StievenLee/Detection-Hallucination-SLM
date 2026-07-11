"""
m_variation.py — Sweep sampling count M ∈ {3,5,7,10} untuk Fig. 3.
==================================================================
Menghasilkan data untuk grafik "Effect of the sampling count M on
detection performance and latency, averaged across models and datasets."

EFISIENSI (penting):
  Kita TIDAK generate ulang untuk tiap M. Sebagai gantinya, generate
  M_MAX=10 sampel SEKALI per soal, lalu untuk tiap M kecil kita ambil
  subset M sampel pertama. Ini:
    - ~2.5x lebih cepat (10 generasi, bukan 3+5+7+10=25),
    - lebih benar secara metodologis: M kecil = subset dari M besar,
      persis seperti mengurangi budget sampling pada distribusi yang sama.

  Latency per-M diestimasi proporsional: karena generasi mendominasi
  latency dan tiap sampel ~sama biayanya, latency(M) = (waktu_gen_total
  untuk M_MAX / M_MAX) * M  +  waktu_clustering(M).
  Waktu clustering diukur nyata untuk tiap M (subset).

Output:
  results/m_variation/m_variation_raw.csv     (per model×dataset×M)
  results/m_variation/m_variation_summary.csv (rata-rata across model×dataset per M)
  results/m_variation/fig_m_variation.pdf     (grafik untuk paper)
  results/m_variation/fig_m_variation.png     (preview)

Jalankan (Colab GPU disarankan biar cepat):
  python m_variation.py
"""

import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from utils.metrics import ResultsLogger, get_ram_usage_mb, compute_auroc, compute_aurac
from utils.model_utils import (
    build_prompt, load_model_and_tokenizer, unload_model,
    generate_responses, generate_best_answer, set_seed,
)
from utils.data_loader import load_dataset_by_name
from utils.scoring import is_correct
from utils.semantic_entropy import SemanticEntropyCalculator

# ── Impor konfigurasi & prompt builder dari se_baseline agar KONSISTEN ──
from se_baseline import (
    MODELS, DATASETS, SIMILARITY_THRESHOLDS, NLI_MODEL,
    MAX_TOKENS, TEMPERATURE, BEST_TEMPERATURE, TOP_P, SEED,
    build_dataset_prompt,
)

# ──────────────────────────────────────────────────
# KONFIGURASI SWEEP
# ──────────────────────────────────────────────────
M_VALUES = [3, 5, 7, 10]      # nilai M yang dievaluasi (Fig. 3)
M_MAX    = max(M_VALUES)      # generate sekali sebanyak ini, sisanya subset
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR      = Path("../results/m_variation")
RAW_CSV      = OUT_DIR / "m_variation_raw.csv"
SUMMARY_CSV  = OUT_DIR / "m_variation_summary.csv"
FIG_PDF      = OUT_DIR / "fig_m_variation.pdf"
FIG_PNG      = OUT_DIR / "fig_m_variation.png"


def run_model_dataset(model_name, dataset_cfg, se_calc, raw_logger):
    """Jalankan satu (model, dataset): generate M_MAX sekali, evaluasi semua M."""
    ds_name = dataset_cfg["name"]
    dataset = load_dataset_by_name(
        name=ds_name, split=dataset_cfg.get("split", "validation"),
        n=dataset_cfg["n"], csv_path=dataset_cfg.get("csv_path"),
    )
    if not dataset:
        print(f"[Skip] {ds_name} kosong"); return []

    language = dataset[0]["language"]
    se_calc.similarity_threshold = SIMILARITY_THRESHOLDS[language]

    print(f"\n{'='*55}\nMODEL: {model_name} | DATASET: {ds_name} ({language}) | {len(dataset)} soal\n{'='*55}")
    model, tokenizer, _ = load_model_and_tokenizer(model_name, DEVICE)

    # Simpan per-soal: 10 sampel + label correct + biaya generasi per sampel
    per_q = []   # list of dict{samples(list[str]), correct(int), gen_time_per_sample(float)}

    for q_idx, sample in enumerate(dataset):
        system_prompt, user_input = build_dataset_prompt(ds_name, sample, language)
        prompt = build_prompt(tokenizer, model_name, system_prompt, user_input)

        # Generate M_MAX sampel SEKALI (T=0.5) — dipakai untuk semua M via subset
        responses, gen_stats = generate_responses(
            model=model, tokenizer=tokenizer, prompt=prompt,
            M=M_MAX, max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, top_p=TOP_P, seed=SEED + q_idx,
        )
        gen_time_per_sample = gen_stats["gen_time_s"] / M_MAX

        # Best answer (T=0.1) SEKALI untuk correctness — sama untuk semua M
        best_answer = generate_best_answer(
            model=model, tokenizer=tokenizer, prompt=prompt,
            max_new_tokens=MAX_TOKENS, temperature=BEST_TEMPERATURE,
            top_p=TOP_P, seed=SEED + 10000 + q_idx,
        )
        correct = int(is_correct(best_answer, sample, dataset_name=ds_name, mode="gold_recall"))

        per_q.append({
            "samples": responses,
            "correct": correct,
            "gen_time_per_sample": gen_time_per_sample,
        })
        if (q_idx + 1) % 20 == 0:
            print(f"  ...{q_idx+1}/{len(dataset)} soal selesai")

    unload_model(model)

    # ── Evaluasi tiap M sebagai subset dari M_MAX sampel ──
    rows = []
    for M in M_VALUES:
        entropies, correctness = [], []
        clustering_time_total = 0.0
        gen_latency_total = 0.0

        for q in per_q:
            subset = q["samples"][:M]           # M sampel pertama
            t0 = time.time()
            se = se_calc.semantic_entropy(subset)
            clustering_time_total += time.time() - t0

            entropies.append(se["entropy"])
            correctness.append(q["correct"])
            # latency per soal = biaya generate M sampel + clustering M sampel
            gen_latency_total += q["gen_time_per_sample"] * M

        auroc = compute_auroc(correctness, entropies)
        aurac = compute_aurac(correctness, entropies)["aurac"]
        accuracy = sum(correctness) / len(correctness)
        n = len(per_q)
        avg_latency = (gen_latency_total + clustering_time_total) / n

        row = {
            "model": model_name, "dataset": ds_name, "language": language,
            "M": M,
            "auroc": round(auroc, 4) if not np.isnan(auroc) else "N/A",
            "aurac": aurac,
            "accuracy": round(accuracy, 4),
            "avg_entropy": round(float(np.mean(entropies)), 4),
            "avg_latency_s": round(avg_latency, 3),
        }
        rows.append(row)
        raw_logger.log(row)
        print(f"  [M={M:2d}] AUROC={row['auroc']} AURAC={aurac} "
              f"acc={accuracy:.2f} lat={avg_latency:.1f}s")

    return rows


def build_summary(all_rows):
    """Rata-ratakan across model×dataset untuk tiap M (untuk Fig. 3)."""
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df["auroc_num"] = pd.to_numeric(df["auroc"], errors="coerce")
    summary = []
    for M in M_VALUES:
        sub = df[df["M"] == M]
        summary.append({
            "M": M,
            "auroc_mean": round(sub["auroc_num"].mean(), 4),
            "auroc_std":  round(sub["auroc_num"].std(), 4),
            "aurac_mean": round(pd.to_numeric(sub["aurac"], errors="coerce").mean(), 4),
            "accuracy_mean": round(sub["accuracy"].mean(), 4),
            "latency_mean_s": round(sub["avg_latency_s"].mean(), 3),
        })
    return pd.DataFrame(summary)


def plot_figure(summary_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = summary_df["M"].values
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Sumbu kiri: metrik deteksi (0..1)
    ax1.plot(M, summary_df["auroc_mean"], "o-", label="AUROC", color="#1f77b4")
    ax1.plot(M, summary_df["aurac_mean"], "s--", label="AURAC", color="#2ca02c")
    ax1.plot(M, summary_df["accuracy_mean"], "^:", label="Accuracy", color="#9467bd")
    ax1.set_xlabel("Sampling count $M$")
    ax1.set_ylabel("Detection metric")
    ax1.set_xticks(M)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Sumbu kanan: latency (detik)
    ax2 = ax1.twinx()
    ax2.plot(M, summary_df["latency_mean_s"], "d-", label="Latency (s)", color="#d62728")
    ax2.set_ylabel("Latency per query (s)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Gabung legenda dua sumbu
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)

    plt.title("Effect of sampling count $M$ (avg. across models & datasets)")
    fig.tight_layout()
    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
    print(f"[Fig] Tersimpan: {FIG_PDF} dan {FIG_PNG}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print("=" * 55)
    print("M-VARIATION SWEEP  (Fig. 3)")
    print("=" * 55)
    print(f"M values : {M_VALUES}  (generate {M_MAX} sekali, sisanya subset)")
    print(f"Models   : {[m.split('/')[-1] for m in MODELS]}")
    print(f"Datasets : {[d['name'] for d in DATASETS]}")
    print(f"Device   : {DEVICE}")

    se_calc    = SemanticEntropyCalculator(model_name=NLI_MODEL,
                                           similarity_threshold=SIMILARITY_THRESHOLDS)
    raw_logger = ResultsLogger(str(RAW_CSV))

    all_rows = []
    for dataset_cfg in DATASETS:
        for model_name in MODELS:
            rows = run_model_dataset(model_name, dataset_cfg, se_calc, raw_logger)
            all_rows.extend(rows)

    se_calc.unload()

    summary_df = build_summary(all_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Summary] Tersimpan: {SUMMARY_CSV}")
    print(summary_df.to_string(index=False))

    plot_figure(summary_df)
    print("\nSELESAI. Masukkan fig_m_variation.pdf ke folder gambar paper Anda.")


if __name__ == "__main__":
    main()
