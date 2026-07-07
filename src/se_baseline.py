"""
SE Baseline — Multi-dataset & Multi-language
=============================================
Support:
  EN: TriviaQA, BioASQ
  ID: FacQA, WReTE

Jalankan:
  python src/se_baseline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.metrics import ResultsLogger, get_ram_usage_mb
from utils.model_utils import build_prompt, load_model_and_tokenizer, unload_model, generate_responses
from utils.data_loader import load_dataset_by_name, is_correct
from utils.semantic_entropy import SemanticEntropyCalculator

# ──────────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────────
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "microsoft/phi-1_5",
    # "Qwen/Qwen1.5-1.8B-Chat",
]

# Dataset yang dijalankan — comment/uncomment sesuai kebutuhan
DATASETS = [
    # {"name": "trivia_qa", "split": "validation", "n": 100, "csv_path": None},
    # {"name": "bioasq",    "split": "factoid",       "n": 100, "csv_path": None},
    # {"name": "facqa",     "split": None,           "n": 100,
    #  "csv_path": "data/raw/facqa/train_preprocess.csv"},
    {"name": "wrete",     "split": None,           "n": 20,
     "csv_path": "data/raw/wrete/train_preprocess.csv"},
]

# Prompt per bahasa
SYSTEM_PROMPTS = {
    "en": (
        "You are a helpful assistant. "
        "Answer the question concisely in 1-2 sentences."
    ),
    "id": (
        "Anda adalah asisten yang membantu. "
        "Jawab pertanyaan berikut secara singkat dalam 1-2 kalimat dalam Bahasa Indonesia."
    ),
}

NLI_MODEL             = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# cosine similarity, bukan entailment score
SIMILARITY_THRESHOLDS = {
    "en": 0.80,   # English: P1=0.998 ✅, P2=0.745 ✅, P4=0.196 ✅
    "id": 0.62,   # Indonesian: P3=0.639 ✅, P5=0.462 ✅
}
M             = 10
MAX_TOKENS    = 100       
TEMPERATURE   = 0.5       # diturunkan dari 0.9 berdasarkan analisis Week 2
TOP_P         = 0.95
DEVICE        = "cpu"
# NLI_THRESHOLD = 0.5

# Path("results/metrics/qwen").mkdir(parents=True, exist_ok=True)
# Path("results/outputs/qwen").mkdir(parents=True, exist_ok=True)
# Path("results/figures/qwen").mkdir(parents=True, exist_ok=True)

RESULTS_CSV = "results/metrics/tinyllama/se_results.csv"
AUROC_CSV   = "results/metrics/tinyllama/se_auroc_summary.csv"

# ──────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────
def run_experiment(model_name, dataset_cfg, se_calc, q_logger):
    """Jalankan SE pipeline untuk satu model × satu dataset."""

    dataset = load_dataset_by_name(
        name=dataset_cfg["name"],
        split=dataset_cfg.get("split", "validation"),
        n=dataset_cfg["n"],
        csv_path=dataset_cfg.get("csv_path"),
    )
    if not dataset:
        print(f"[Skip] Dataset kosong: {dataset_cfg['name']}")
        return None

    # Deteksi bahasa dari dataset
    language = dataset[0]["language"]
    system_prompt = SYSTEM_PROMPTS[language]
    threshold     = SIMILARITY_THRESHOLDS[language]  # ← ambil threshold per bahasa
    se_calc.similarity_threshold = threshold   

    # WReTE adalah task entailment biner -> instruksikan jawab ya/tidak saja
    if dataset_cfg["name"] == "wrete":
        system_prompt = (
            "Anda adalah asisten yang menentukan apakah sebuah pernyataan benar "
            "berdasarkan teks bacaan. Jawab HANYA dengan satu kata: 'ya' jika "
            "pernyataan benar sesuai bacaan, atau 'tidak' jika tidak sesuai. "
            "Jangan memberikan penjelasan."
        )

    print(f"\n{'='*55}")
    print(f"MODEL  : {model_name}")
    print(f"DATASET: {dataset_cfg['name']} ({language.upper()}) — {len(dataset)} soal")
    print(f"{'='*55}")

    model, tokenizer, load_stats = load_model_and_tokenizer(model_name, DEVICE)

    entropies   = []
    correctness = []
    all_throughput = []   
    start_total = time.time()

    # Peak RAM tracking: RSS awal (sudah termasuk SLM + MiniLM ter-load)
    peak_ram_mb = get_ram_usage_mb()

    for q_idx, sample in enumerate(dataset):
        print(f"\n[Q {q_idx+1}/{len(dataset)}] {sample['question'][:65]}...")

        # Sebelum build_prompt
        if sample.get("passage"):
            user_input = f"Bacaan: {sample['passage']}\n\nPertanyaan: {sample['question']}"
        else:
            user_input = sample["question"]

        # WReTE: paksa format jawaban biner sedekat mungkin dgn pertanyaan
        if dataset_cfg["name"] == "wrete":
            user_input = user_input + "\n\nJawab dengan satu kata saja: ya atau tidak."

        prompt = build_prompt(
            tokenizer=tokenizer,
            model_name=model_name,
            system=system_prompt,
            user=user_input,
        )

        responses, gen_stats= generate_responses(
            model=model, tokenizer=tokenizer, prompt=prompt,
            M=M, max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, top_p=TOP_P,
        )
        all_throughput.append(gen_stats["tokens_per_sec"])

        se_start  = time.time()
        se_result = se_calc.semantic_entropy(responses)
        se_time   = time.time() - se_start

        # Update peak RAM setelah generate + clustering (titik RSS tertinggi per soal)
        current_ram = get_ram_usage_mb()
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram

        correct = int(is_correct(responses[0], sample, dataset_name=dataset_cfg["name"]))

        # Tambahkan ini sementara
        print(f"  GT    : '{sample['answer']}'")
        print(f"  Pred  : '{responses[0][:100]}'")
        print(f"  Correct: {correct}")

        entropies.append(se_result["entropy"])
        correctness.append(correct)

        print(f"  ✓ SE={se_result['entropy']:.4f} | "
              f"clusters={se_result['n_clusters']}/{M} | "
              f"correct={correct} | lang={language}")

        # Simpan hasil dengan kolom language & dataset
        q_logger.log({
            "model":        model_name,
            "dataset":      dataset_cfg["name"],
            "language":     language,
            "question_idx": q_idx,
            "question":     sample["question"][:100],
            "answer_gt":    sample["answer"],
            "prediction":   responses[0][:100],
            "correct":      correct,
            "se_entropy":   se_result["entropy"],
            "n_clusters":   se_result["n_clusters"],
            "M":            M,
            "temperature":  TEMPERATURE,
            "se_time_s":    round(se_time, 3),
            "gen_time_s":   gen_stats["gen_time_s"],        # ← tambah
            "latency_s":    gen_stats["gen_time_s"] + se_time,  # ← total latency per query
            "throughput_tokens_per_sec": gen_stats["tokens_per_sec"],  # ← tambah
        })

    total_time = time.time() - start_total

    from utils.metrics import compute_auroc, compute_aurac, compute_rejection_accuracy
    # AUROC
    auroc = compute_auroc(correctness, entropies)
    # Rejection accuracy di beberapa threshold
    p25 = float(np.percentile(entropies, 25))   # buang 75% teratas
    p50 = float(np.percentile(entropies, 50))   # buang 50% teratas
    p75 = float(np.percentile(entropies, 75))   # buang 25% teratas
    p90 = float(np.percentile(entropies, 90))   # buang 10% teratas

    rej_05 = compute_rejection_accuracy(correctness, entropies, threshold=p25)
    rej_10 = compute_rejection_accuracy(correctness, entropies, threshold=p50)
    rej_15 = compute_rejection_accuracy(correctness, entropies, threshold=p75)
    rej_20 = compute_rejection_accuracy(correctness, entropies, threshold=p90)

    # AURAC
    aurac_result = compute_aurac(correctness, entropies)

    accuracy = sum(correctness) / len(correctness)

    print(f"\n  AUROC    : {auroc:.4f}" if not np.isnan(auroc) else "\n  AUROC    : N/A")
    print(f"  AURAC    : {aurac_result['aurac']:.4f}" if aurac_result['aurac'] is not None and not np.isnan(aurac_result['aurac']) else "  AURAC    : N/A")
    print(f"  Accuracy : {accuracy:.2%}")
    print(f"  Avg SE   : {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")


    if aurac_result.get("best_threshold") is not None:
        print(f"\n  Best threshold: SE ≤ {aurac_result['best_threshold']:.4f} "
              f"(risk < 30%)")

    unload_model(model)

    return {
        "model":           model_name,
        "dataset":         dataset_cfg["name"],
        "language":        language,
        "n_questions":     len(dataset),
        "M":               M,
        "temperature":     TEMPERATURE,
        "auroc":           round(auroc, 4) if not np.isnan(auroc) else "N/A",
        "aurac":           aurac_result["aurac"],         
        "best_threshold":  aurac_result["best_threshold"],           
        "accuracy":        round(accuracy, 4),
        "avg_entropy":     round(float(np.mean(entropies)), 4),
        "std_entropy":     round(float(np.std(entropies)), 4),
        "total_time_s":    round(total_time, 1),
        "similarity_calls_total": (M * (M - 1) // 2) * len(dataset),
        "avg_latency_s":    round(total_time / len(dataset), 2),
        "throughput_tok_s": round(float(np.mean(all_throughput)), 2),
        "peak_ram_mb":      round(peak_ram_mb, 1),
        **load_stats,
    }


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("SE BASELINE — MULTI-DATASET & MULTI-LANGUAGE")
    print("=" * 55)
    print(f"Models  : {len(MODELS)}")
    print(f"Datasets: {[d['name'] for d in DATASETS]}")
    print(f"M       : {M}")
    print(f"Temp    : {TEMPERATURE}")

    se_calc = SemanticEntropyCalculator(model_name=NLI_MODEL, similarity_threshold=SIMILARITY_THRESHOLDS)
    q_logger     = ResultsLogger(RESULTS_CSV)
    auroc_logger = ResultsLogger(AUROC_CSV)

    for dataset_cfg in DATASETS:
        for model_name in MODELS:
            result = run_experiment(model_name, dataset_cfg, se_calc, q_logger)
            if result:
                auroc_logger.log(result)

    se_calc.unload()

    print(f"\n{'='*55}")
    print("SELESAI. AUROC Summary:")
    cols = ["model", "dataset", "language", "auroc", "aurac", "accuracy", "avg_entropy", "peak_ram_mb", "avg_latency_s", "throughput_tok_s"]
    print(auroc_logger.summary()[cols].to_string(index=False))

    print(f"\nFile tersimpan:")
    print(f"  {RESULTS_CSV}")
    print(f"  {AUROC_CSV}")


if __name__ == "__main__":
    main()