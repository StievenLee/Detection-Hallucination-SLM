"""
Week 2 — Replikasi Semantic Entropy Baseline
=============================================
Tujuan:
  - Hitung semantic entropy untuk setiap respons model di TriviaQA
  - Ukur AUROC: seberapa baik SE membedakan jawaban benar vs salah
  - Catat resource usage sebagai baseline semantic entropy
  - Simpan ke results/
    
Jalankan:
  python src/se_baseline.py

Estimasi waktu (CPU):
  50 soal × 3 model × M=10 → ~2-3 jam
  Mulai dengan N_QUESTIONS=20 untuk validasi dulu
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.metrics import ResultsLogger, get_ram_usage_mb
from utils.model_utils import (
    build_prompt,
    load_model_and_tokenizer,
    unload_model,
    generate_responses,
)
from utils.data_loader import load_trivia_qa, is_correct
from utils.semantic_entropy import SemanticEntropyCalculator

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5",
    "Qwen/Qwen1.5-1.8B-Chat",
]

NLI_MODEL     = "cross-encoder/nli-MiniLM2-L6-H768"
N_QUESTIONS   = 100        # ← mulai dari 20 untuk validasi, naikkan ke 100+ nanti
M             = 10        # jumlah sampel per pertanyaan (sesuai paper)
MAX_TOKENS    = 100
TEMPERATURE   = 0.9
TOP_P         = 0.95
DEVICE        = "cpu"
NLI_THRESHOLD = 0.5       # threshold entailment

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question concisely in 1-2 sentences."
)

Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/outputs").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

RESULTS_CSV    = "results/metrics/se_results.csv"
AUROC_CSV      = "results/metrics/se_auroc_summary.csv"
RESPONSES_TXT  = "results/outputs/se_responses.txt"

# ──────────────────────────────────────────────
# PIPELINE PER MODEL
# ──────────────────────────────────────────────
def run_model(model_name, dataset, se_calc, logger):
    """Jalankan semantic entropy pipeline untuk satu model."""

    print(f"\n{'='*55}")
    print(f"MODEL: {model_name}")
    print(f"{'='*55}")

    model, tokenizer, load_stats = load_model_and_tokenizer(model_name, DEVICE)

    entropies       = []
    correctness     = []
    total_nli_calls = 0
    start_total     = time.time()

    for q_idx, sample in enumerate(dataset):
        print(f"\n[Q {q_idx+1}/{len(dataset)}] {sample['question'][:70]}...")

        prompt = build_prompt(
            tokenizer=tokenizer,
            model_name=model_name,
            system=SYSTEM_PROMPT,
            user=sample["question"],
        )

        # Generate M respons
        responses, gen_stats = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            M=M,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        # Hitung semantic entropy
        se_start  = time.time()
        se_result = se_calc.semantic_entropy(responses, threshold=NLI_THRESHOLD)
        se_time   = time.time() - se_start

        # NLI calls = M*(M-1) bidirectional
        total_nli_calls += M * (M - 1)

        # Correctness: pakai respons pertama sebagai prediksi utama
        correct = int(is_correct(responses[0], sample))

        entropies.append(se_result["entropy"])
        correctness.append(correct)

        print(f"  ✓ SE={se_result['entropy']:.4f} | "
              f"clusters={se_result['n_clusters']}/{M} | "
              f"correct={correct} | "
              f"SE time={se_time:.1f}s")

        logger.log({
            "model":        model_name,
            "question_idx": q_idx,
            "question":     sample["question"][:100],
            "answer_gt":    sample["answer"],
            "prediction":   responses[0][:100],
            "correct":      correct,
            "se_entropy":   se_result["entropy"],
            "n_clusters":   se_result["n_clusters"],
            "M":            M,
            "se_time_s":    round(se_time, 3),
            **{f"response_{i+1}": r[:80] for i, r in enumerate(responses)},
        })

    total_time = time.time() - start_total

    # ── AUROC
    # SE tinggi → model ragu → prediksi salah → flip ke -SE untuk roc_auc_score
    if len(set(correctness)) < 2:
        auroc = float("nan")
        print("  [Warning] Semua jawaban sama — AUROC tidak bisa dihitung.")
        print("  [Tip] Naikkan N_QUESTIONS agar ada variasi benar/salah.")
    else:
        auroc = roc_auc_score(correctness, [-e for e in entropies])

    accuracy = sum(correctness) / len(correctness)

    print(f"\n{'─'*45}")
    print(f"  AUROC     : {auroc:.4f}" if not np.isnan(auroc) else "  AUROC     : N/A")
    print(f"  Accuracy  : {accuracy:.2%}")
    print(f"  Avg SE    : {np.mean(entropies):.4f}")
    print(f"  Std SE    : {np.std(entropies):.4f}")
    print(f"  Total time: {total_time:.0f}s")

    unload_model(model)

    return {
        "model":            model_name,
        "n_questions":      len(dataset),
        "M":                M,
        "auroc":            round(auroc, 4) if not np.isnan(auroc) else "N/A",
        "accuracy":         round(accuracy, 4),
        "avg_entropy":      round(float(np.mean(entropies)), 4),
        "std_entropy":      round(float(np.std(entropies)), 4),
        "total_time_s":     round(total_time, 1),
        "nli_calls_total":  total_nli_calls,
        **load_stats,
    }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("SE BASELINE — SEMANTIC ENTROPY EVALUATION")
    print("=" * 55)
    print(f"Models     : {len(MODELS)}")
    print(f"Questions  : {N_QUESTIONS}")
    print(f"M (samples): {M}")
    print(f"NLI model  : {NLI_MODEL}")
    print(f"RAM awal   : {get_ram_usage_mb():.0f} MB")

    # Load dataset
    dataset = load_trivia_qa(split="validation", n=N_QUESTIONS)

    print(f"\nContoh soal:")
    for s in dataset[:3]:
        print(f"  Q: {s['question']}")
        print(f"  A: {s['answer']}\n")

    # Load NLI — tetap di memory selama semua model berjalan
    se_calc      = SemanticEntropyCalculator(NLI_MODEL)
    q_logger     = ResultsLogger(RESULTS_CSV)
    auroc_logger = ResultsLogger(AUROC_CSV)

    for model_name in MODELS:
        summary = run_model(model_name, dataset, se_calc, q_logger)
        auroc_logger.log(summary)

    se_calc.unload()

    with open(RESPONSES_TXT, "w", encoding="utf-8") as f:
        df = pd.read_csv(RESULTS_CSV)
        for _, row in df.iterrows():
            f.write(f"\n{'='*55}\n")
            f.write(f"Model   : {row['model']}\n")
            f.write(f"Q       : {row['question']}\n")
            f.write(f"Answer  : {row['answer_gt']}\n")
            f.write(f"Predict : {row['prediction']}\n")
            f.write(f"Correct : {row['correct']} | SE: {row['se_entropy']}\n")

    print(f"\n{'='*55}")
    print("SELESAI. AUROC Summary:")
    print(auroc_logger.summary()[
        ["model", "auroc", "accuracy", "avg_entropy", "total_time_s"]
    ].to_string(index=False))

    print(f"\nFile tersimpan:")
    print(f"  {RESULTS_CSV}")
    print(f"  {AUROC_CSV}")


if __name__ == "__main__":
    main()