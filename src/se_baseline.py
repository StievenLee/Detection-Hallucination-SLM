"""
SE Baseline — Multi-dataset & Multi-language  (OPSI A: short-phrase)
====================================================================
Support:
  EN: TriviaQA, BioASQ
  ID: FacQA, WReTE

Perubahan Opsi A + prompt per-dataset (anti-echo):
  - Prompt & format input DIOPTIMASI PER-DATASET karena karakteristik input beda.
  - FacQA: pertanyaan DULU lalu bacaan (dipangkas) -> kurangi echo/menyalin.
  - WReTE: few-shot 2 contoh ya/tidak -> model paham format biner.
  - Correctness factoid: gold_recall; F1 SQuAD sbg pembanding.
  - WReTE: polaritas ya/tidak; macro-F1 dilaporkan terpisah.

Jalankan:
  python src/se_baseline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.metrics import ResultsLogger, get_ram_usage_mb
from utils.model_utils import build_prompt, load_model_and_tokenizer, unload_model, generate_responses, generate_best_answer
from utils.data_loader import load_dataset_by_name
from utils.scoring import is_correct, wrete_macro_f1, best_squad_f1, best_gold_recall
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

# ── PROMPT PER-DATASET ────────────────────────────────────────────────────
# System prompt disesuaikan dengan karakteristik input tiap dataset.
SYSTEM_PROMPTS = {
    # TriviaQA / BioASQ: factoid tanpa passage, jawaban singkat
    "en": (
        "You are a helpful assistant. Answer the question with the shortest "
        "possible answer, e.g. only a name, place, date, or number. "
        "Do not answer in a full sentence. Do not explain."
    ),
    # FacQA: factoid Indonesia DENGAN bacaan -> tegaskan jangan menyalin bacaan
    "facqa": (
        "Anda menjawab pertanyaan berdasarkan bacaan. Berikan HANYA jawaban "
        "singkat berupa kata atau frasa. Jangan menyalin atau mengulang bacaan. "
        "Jangan menjelaskan."
    ),
    # WReTE: entailment biner -> few-shot agar paham format ya/tidak
    "wrete": (
        "Anda menentukan apakah pernyataan kedua dapat disimpulkan dari "
        "pernyataan pertama. Jawab HANYA satu kata: 'ya' atau 'tidak'.\n\n"
        "Contoh 1:\n"
        "Pernyataan 1: Semua kucing adalah hewan.\n"
        "Pernyataan 2: Kucing adalah hewan.\n"
        "Jawaban: ya\n\n"
        "Contoh 2:\n"
        "Pernyataan 1: Budi pergi ke pasar.\n"
        "Pernyataan 2: Budi membeli ikan.\n"
        "Jawaban: tidak"
    ),
    # fallback ID generic (tidak dipakai kalau facqa/wrete sudah spesifik)
    "id": (
        "Anda adalah asisten yang membantu. Jawab pertanyaan dengan jawaban "
        "sesingkat mungkin. Jangan menjawab dalam kalimat lengkap. Jangan menjelaskan."
    ),
}

PASSAGE_MAX_CHARS = 600   # pangkas bacaan FacQA agar tak membebani model kecil

NLI_MODEL             = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLDS = {"en": 0.80, "id": 0.62}
M             = 10
MAX_TOKENS    = 30
TEMPERATURE   = 0.5
BEST_TEMPERATURE = 0.1   # temperature rendah utk 'best generation' (correctness), ikut [3]
TOP_P         = 0.95
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F1_THRESHOLD  = 0.5

RESULTS_CSV = "results/metrics/tinyllama/se_results.csv"
AUROC_CSV   = "results/metrics/tinyllama/se_auroc_summary.csv"


# ──────────────────────────────────────────────────
# PROMPT BUILDER PER-DATASET
# ──────────────────────────────────────────────────
def build_dataset_prompt(dataset_name: str, sample: dict, language: str):
    """
    Kembalikan (system_prompt, user_input) yang dioptimasi per-dataset.
    Ini memusatkan semua logika prompt di satu tempat.
    """
    if dataset_name == "facqa":
        system_prompt = SYSTEM_PROMPTS["facqa"]
        passage = (sample.get("passage", "") or "")[:PASSAGE_MAX_CHARS]
        # PERTANYAAN DULU, baru bacaan, lalu trigger jawaban -> kurangi echo
        user_input = (
            f"Pertanyaan: {sample['question']}\n\n"
            f"Bacaan: {passage}\n\n"
            f"Jawaban singkat:"
        )

    elif dataset_name == "wrete":
        system_prompt = SYSTEM_PROMPTS["wrete"]
        # question sudah berisi premis+hipotesis dari loader; tambah trigger
        user_input = f"{sample['question']}\n\nJawaban (ya/tidak):"

    else:  # trivia_qa, bioasq
        system_prompt = SYSTEM_PROMPTS["en"]
        user_input = sample["question"]

    return system_prompt, user_input


# ──────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────
def run_experiment(model_name, dataset_cfg, se_calc, q_logger):
    ds_name = dataset_cfg["name"]
    dataset = load_dataset_by_name(
        name=ds_name,
        split=dataset_cfg.get("split", "validation"),
        n=dataset_cfg["n"],
        csv_path=dataset_cfg.get("csv_path"),
    )
    if not dataset:
        print(f"[Skip] Dataset kosong: {ds_name}")
        return None

    language = dataset[0]["language"]
    se_calc.similarity_threshold = SIMILARITY_THRESHOLDS[language]

    print(f"\n{'='*55}")
    print(f"MODEL  : {model_name}")
    print(f"DATASET: {ds_name} ({language.upper()}) — {len(dataset)} soal")
    print(f"{'='*55}")

    model, tokenizer, load_stats = load_model_and_tokenizer(model_name, DEVICE)

    entropies, correctness, predictions, samples_kept = [], [], [], []
    all_throughput = []
    start_total = time.time()
    peak_ram_mb = get_ram_usage_mb()

    for q_idx, sample in enumerate(dataset):
        print(f"\n[Q {q_idx+1}/{len(dataset)}] {sample['question'][:65]}...")

        # ── Prompt dioptimasi per-dataset ──
        system_prompt, user_input = build_dataset_prompt(ds_name, sample, language)

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

        # Best generation (T=0.1) TERPISAH utk menilai correctness (ikut [3]).
        # M sampel di atas tetap dipakai utk semantic entropy.
        best_answer = generate_best_answer(
            model=model, tokenizer=tokenizer, prompt=prompt,
            max_new_tokens=MAX_TOKENS, temperature=BEST_TEMPERATURE, top_p=TOP_P,
        )

        se_start  = time.time()
        se_result = se_calc.semantic_entropy(responses)
        se_time   = time.time() - se_start

        current_ram = get_ram_usage_mb()
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram

        correct = int(is_correct(best_answer, sample,
                                 dataset_name=ds_name, mode="gold_recall"))
        f1_squad = None if ds_name == "wrete" else round(best_squad_f1(best_answer, sample), 4)

        predictions.append(best_answer)
        samples_kept.append(sample)

        print(f"  GT     : '{sample['answer']}'")
        print(f"  Best   : '{best_answer[:100]}'  <- dinilai (T=0.1)")
        print(f"  Sample0: '{responses[0][:80]}'  (utk SE)")
        print(f"  Correct: {correct}  (F1_squad={f1_squad})")

        entropies.append(se_result["entropy"])
        correctness.append(correct)

        print(f"  SE={se_result['entropy']:.4f} | clusters={se_result['n_clusters']}/{M} | correct={correct}")

        q_logger.log({
            "model": model_name, "dataset": ds_name, "language": language,
            "question_idx": q_idx, "question": sample["question"][:100],
            "answer_gt": sample["answer"], "prediction": best_answer[:300],
            "sample0_for_se": responses[0][:200],
            "correct": correct, "f1_squad": f1_squad,
            "se_entropy": se_result["entropy"], "n_clusters": se_result["n_clusters"],
            "M": M, "temperature": TEMPERATURE, "se_time_s": round(se_time, 3),
            "gen_time_s": gen_stats["gen_time_s"],
            "latency_s": gen_stats["gen_time_s"] + se_time,
            "throughput_tokens_per_sec": gen_stats["tokens_per_sec"],
        })

    total_time = time.time() - start_total

    from utils.metrics import compute_auroc, compute_aurac
    auroc = compute_auroc(correctness, entropies)
    aurac_result = compute_aurac(correctness, entropies)
    accuracy = sum(correctness) / len(correctness)

    macro_f1 = None
    if ds_name == "wrete":
        wf = wrete_macro_f1(predictions, samples_kept)
        macro_f1 = wf["macro_f1"]
        print(f"  WReTE macro-F1: {macro_f1:.4f} (abstain={wf['abstain_count']})")

    print(f"\n  AUROC    : {auroc:.4f}" if not np.isnan(auroc) else "\n  AUROC    : N/A")
    print(f"  AURAC    : {aurac_result['aurac']}")
    print(f"  Accuracy : {accuracy:.2%}")
    print(f"  Avg SE   : {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")

    unload_model(model)

    return {
        "model": model_name, "dataset": ds_name, "language": language,
        "n_questions": len(dataset), "M": M, "temperature": TEMPERATURE,
        "auroc": round(auroc, 4) if not np.isnan(auroc) else "N/A",
        "aurac": aurac_result["aurac"], "best_threshold": aurac_result["best_threshold"],
        "accuracy": round(accuracy, 4), "wrete_macro_f1": macro_f1,
        "avg_entropy": round(float(np.mean(entropies)), 4),
        "std_entropy": round(float(np.std(entropies)), 4),
        "total_time_s": round(total_time, 1),
        "similarity_calls_total": (M * (M - 1) // 2) * len(dataset),
        "avg_latency_s": round(total_time / len(dataset), 2),
        "throughput_tok_s": round(float(np.mean(all_throughput)), 2),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "f1_threshold": F1_THRESHOLD, "max_tokens": MAX_TOKENS,
        **load_stats,
    }


def main():
    print("=" * 55)
    print("SE BASELINE")
    print("=" * 55)
    print(f"Models: {len(MODELS)} | Datasets: {[d['name'] for d in DATASETS]}")
    print(f"M: {M} | MAX_TOKENS: {MAX_TOKENS}")

    se_calc = SemanticEntropyCalculator(model_name=NLI_MODEL, similarity_threshold=SIMILARITY_THRESHOLDS)
    q_logger     = ResultsLogger(RESULTS_CSV)
    auroc_logger = ResultsLogger(AUROC_CSV)

    for dataset_cfg in DATASETS:
        for model_name in MODELS:
            result = run_experiment(model_name, dataset_cfg, se_calc, q_logger)
            if result:
                auroc_logger.log(result)

    se_calc.unload()

    print(f"\n{'='*55}\nSELESAI. Summary:")
    cols = ["model", "dataset", "language", "auroc", "aurac", "accuracy",
            "wrete_macro_f1", "avg_entropy", "peak_ram_mb", "avg_latency_s", "throughput_tok_s"]
    print(auroc_logger.summary()[cols].to_string(index=False))


if __name__ == "__main__":
    main()
