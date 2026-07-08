"""
SE Baseline — Multi-dataset & Multi-language  (OPSI A: short-phrase)
====================================================================
Support:
  EN: TriviaQA, BioASQ
  ID: FacQA, WReTE

Perubahan Opsi A (mengikuti protokol short-phrase Farquhar et al., Nature 2024):
  - Prompt meminta jawaban SESINGKAT MUNGKIN (bukan 1-2 kalimat).
  - MAX_TOKENS diturunkan (30) agar jawaban ringkas.
  - Correctness factoid dinilai F1 SQuAD > 0.5 (BUKAN substring matching).
  - WReTE tetap polaritas ya/tidak; macro-F1 dilaporkan terpisah.

Jalankan:
  python src/se_baseline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.metrics_old import ResultsLogger, get_ram_usage_mb
from utils.model_utils import build_prompt, load_model_and_tokenizer, unload_model, generate_responses
from utils.data_loader import load_dataset_by_name
from utils.scoring import is_correct, wrete_macro_f1, best_squad_f1, best_gold_recall   # ← modul scoring baru
from utils.semantic_entropy import SemanticEntropyCalculator

# ──────────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────────
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "microsoft/phi-1_5",
    # "Qwen/Qwen1.5-1.8B-Chat",
]

DATASETS = [
    {"name": "trivia_qa", "split": "validation", "n": 100, "csv_path": None},
    {"name": "bioasq",    "split": "factoid",    "n": 100, "csv_path": None},
    {"name": "facqa",     "split": None,         "n": 100,
     "csv_path": "data/raw/facqa/train_preprocess.csv"},
    {"name": "wrete",     "split": None,         "n": 100,
     "csv_path": "data/raw/wrete/train_preprocess.csv"},
]

# ── Prompt SHORT-PHRASE (Opsi A) ──────────────────────────────────────────
# Minta jawaban sependek mungkin: hanya entitas/nama/tanggal/angka.
SYSTEM_PROMPTS = {
    "en": (
        "You are a helpful assistant. Answer the question with the shortest "
        "possible answer, e.g. only a name, place, date, or number. "
        "Do not answer in a full sentence. Do not explain."
    ),
    "id": (
        "Anda adalah asisten yang membantu. Jawab pertanyaan dengan jawaban "
        "sesingkat mungkin, misalnya hanya nama, tempat, tanggal, atau angka. "
        "Jangan menjawab dalam kalimat lengkap. Jangan menjelaskan."
    ),
}

NLI_MODEL             = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLDS = {
    "en": 0.80,
    "id": 0.62,
}
M             = 10
MAX_TOKENS    = 30        # ← diturunkan dari 100 untuk skenario short-phrase
TEMPERATURE   = 0.5
TOP_P         = 0.95
DEVICE        = "cpu"
F1_THRESHOLD  = 0.5       # ← ambang correctness factoid (ala SQuAD short-phrase)

RESULTS_CSV = "results/metrics/tinyllama/se_results.csv"
AUROC_CSV   = "results/metrics/tinyllama/se_auroc_summary.csv"


# ──────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────
def run_experiment(model_name, dataset_cfg, se_calc, q_logger):
    dataset = load_dataset_by_name(
        name=dataset_cfg["name"],
        split=dataset_cfg.get("split", "validation"),
        n=dataset_cfg["n"],
        csv_path=dataset_cfg.get("csv_path"),
    )
    if not dataset:
        print(f"[Skip] Dataset kosong: {dataset_cfg['name']}")
        return None

    language = dataset[0]["language"]
    system_prompt = SYSTEM_PROMPTS[language]
    threshold     = SIMILARITY_THRESHOLDS[language]
    se_calc.similarity_threshold = threshold

    # WReTE: task entailment biner -> instruksikan jawab ya/tidak saja
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
    predictions = []          # ← simpan responses[0] untuk macro-F1 WReTE
    samples_kept = []
    all_throughput = []
    start_total = time.time()
    peak_ram_mb = get_ram_usage_mb()

    for q_idx, sample in enumerate(dataset):
        print(f"\n[Q {q_idx+1}/{len(dataset)}] {sample['question'][:65]}...")

        if sample.get("passage"):
            user_input = f"Bacaan: {sample['passage']}\n\nPertanyaan: {sample['question']}"
        else:
            user_input = sample["question"]

        if dataset_cfg["name"] == "wrete":
            user_input = user_input + "\n\nJawab dengan satu kata saja: ya atau tidak."

        prompt = build_prompt(
            tokenizer=tokenizer,
            model_name=model_name,
            system=system_prompt,
            user=user_input,
        )

        responses, gen_stats = generate_responses(
            model=model, tokenizer=tokenizer, prompt=prompt,
            M=M, max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, top_p=TOP_P,
        )
        all_throughput.append(gen_stats["tokens_per_sec"])

        se_start  = time.time()
        se_result = se_calc.semantic_entropy(responses)
        se_time   = time.time() - se_start

        current_ram = get_ram_usage_mb()
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram

        # ── Correctness UTAMA: gold_recall (factoid) atau polaritas (WReTE) ──
        correct = int(is_correct(responses[0], sample,
                                 dataset_name=dataset_cfg["name"],
                                 mode="gold_recall"))

        # ── Metrik PEMBANDING (hanya factoid): F1 SQuAD ala [3] ──
        if dataset_cfg["name"] == "wrete":
            f1_squad = None
        else:
            f1_squad = round(best_squad_f1(responses[0], sample), 4)

        predictions.append(responses[0])
        samples_kept.append(sample)

        print(f"  GT     : '{sample['answer']}'")
        print(f"  Pred   : '{responses[0][:100]}'")
        print(f"  Correct: {correct}  (F1_squad={f1_squad})")

        entropies.append(se_result["entropy"])
        correctness.append(correct)

        print(f"  ✓ SE={se_result['entropy']:.4f} | "
              f"clusters={se_result['n_clusters']}/{M} | "
              f"correct={correct} | lang={language}")

        q_logger.log({
            "model":        model_name,
            "dataset":      dataset_cfg["name"],
            "language":     language,
            "question_idx": q_idx,
            "question":     sample["question"][:100],
            "answer_gt":    sample["answer"],
            "prediction":   responses[0][:300],   # ← lebih panjang agar rescore dari CSV akurat
            "correct":      correct,
            "f1_squad":     f1_squad,
            "se_entropy":   se_result["entropy"],
            "n_clusters":   se_result["n_clusters"],
            "M":            M,
            "temperature":  TEMPERATURE,
            "se_time_s":    round(se_time, 3),
            "gen_time_s":   gen_stats["gen_time_s"],
            "latency_s":    gen_stats["gen_time_s"] + se_time,
            "throughput_tokens_per_sec": gen_stats["tokens_per_sec"],
        })

    total_time = time.time() - start_total

    from utils.metrics_old import compute_auroc, compute_aurac, compute_rejection_accuracy
    auroc = compute_auroc(correctness, entropies)
    aurac_result = compute_aurac(correctness, entropies)
    accuracy = sum(correctness) / len(correctness)

    # ── Macro-F1 khusus WReTE (dilaporkan terpisah) ──
    macro_f1 = None
    if dataset_cfg["name"] == "wrete":
        wf = wrete_macro_f1(predictions, samples_kept)
        macro_f1 = wf["macro_f1"]
        print(f"  WReTE macro-F1: {macro_f1:.4f} "
              f"(abstain={wf['abstain_count']})")

    print(f"\n  AUROC    : {auroc:.4f}" if not np.isnan(auroc) else "\n  AUROC    : N/A")
    print(f"  AURAC    : {aurac_result['aurac']:.4f}" if aurac_result['aurac'] is not None and not np.isnan(aurac_result['aurac']) else "  AURAC    : N/A")
    print(f"  Accuracy : {accuracy:.2%}")
    print(f"  Avg SE   : {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")

    if aurac_result.get("best_threshold") is not None:
        print(f"\n  Best threshold: SE ≤ {aurac_result['best_threshold']:.4f} (risk < 30%)")

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
        "wrete_macro_f1":  macro_f1,                     # ← None untuk non-WReTE
        "avg_entropy":     round(float(np.mean(entropies)), 4),
        "std_entropy":     round(float(np.std(entropies)), 4),
        "total_time_s":    round(total_time, 1),
        "similarity_calls_total": (M * (M - 1) // 2) * len(dataset),
        "avg_latency_s":    round(total_time / len(dataset), 2),
        "throughput_tok_s": round(float(np.mean(all_throughput)), 2),
        "peak_ram_mb":      round(peak_ram_mb, 1),
        "f1_threshold":     F1_THRESHOLD,
        "max_tokens":       MAX_TOKENS,
        **load_stats,
    }


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("SE BASELINE — OPSI A (short-phrase, F1>0.5)")
    print("=" * 55)
    print(f"Models  : {len(MODELS)}")
    print(f"Datasets: {[d['name'] for d in DATASETS]}")
    print(f"M       : {M} | MAX_TOKENS: {MAX_TOKENS} | F1_thr: {F1_THRESHOLD}")

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
    print("SELESAI. Summary:")
    cols = ["model", "dataset", "language", "auroc", "aurac", "accuracy",
            "wrete_macro_f1", "avg_entropy", "peak_ram_mb", "avg_latency_s", "throughput_tok_s"]
    print(auroc_logger.summary()[cols].to_string(index=False))

    print(f"\nFile tersimpan:")
    print(f"  {RESULTS_CSV}")
    print(f"  {AUROC_CSV}")


if __name__ == "__main__":
    main()
