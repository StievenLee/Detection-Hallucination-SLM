"""
threshold_analysis.py — kalibrasi threshold yang BERSIH (tanpa leakage).

Menggabungkan dua pendekatan:
  1. DEV/TEST SPLIT: threshold dipilih di dev (30%), AUROC dilaporkan di test (70%).
     -> tidak ada data leakage.
  2. SENSITIVITAS: tampilkan AUROC di test across banyak threshold.
     -> tunjukkan hasil robust / tidak bergantung satu angka ajaib.

Input: file .jsonl berisi {"samples": [...M teks...], "correct": 0/1} per baris.
(Dihasilkan dari patch simpan-sampel di se_baseline.py)

Pakai:
  python threshold_analysis.py results/samples/facqa_samples.jsonl id
"""
import sys, json, math
import numpy as np
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer, util

ENC_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEV_FRAC = 0.30
SEED = 42


def entropy_at(embeddings, threshold):
    M = len(embeddings)
    sim = util.cos_sim(embeddings, embeddings)
    clusters = []
    for i in range(M):
        placed = False
        for cl in clusters:
            avg = sum(sim[i][j].item() for j in cl) / len(cl)
            if avg >= threshold:
                cl.append(i); placed = True; break
        if not placed:
            clusters.append([i])
    probs = [len(c) / M for c in clusters]
    return -sum(p * math.log(p) for p in probs if p > 0)


def auroc_at(embs, correctness, thr):
    ents = [entropy_at(e, thr) for e in embs]
    if len(set(round(x, 4) for x in ents)) < 2 or len(set(correctness)) < 2:
        return float('nan'), np.mean(ents)
    return roc_auc_score(correctness, [-e for e in ents]), np.mean(ents)


def main():
    if len(sys.argv) < 3:
        print("Pakai: python threshold_analysis.py <samples.jsonl> <lang>")
        return
    path, lang = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"Loaded {len(rows)} soal dari {path}\n")

    enc = SentenceTransformer(ENC_NAME)
    embs = [enc.encode(r["samples"], convert_to_tensor=True) for r in rows]
    correct = [r["correct"] for r in rows]

    # Split dev/test (stratified sederhana via shuffle berseed)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(rows)); rng.shuffle(idx)
    n_dev = int(len(rows) * DEV_FRAC)
    dev_idx, test_idx = idx[:n_dev], idx[n_dev:]

    embs_dev  = [embs[i] for i in dev_idx];  corr_dev  = [correct[i] for i in dev_idx]
    embs_test = [embs[i] for i in test_idx]; corr_test = [correct[i] for i in test_idx]

    print(f"Dev: {len(dev_idx)} soal (acc={np.mean(corr_dev):.2f}) | "
          f"Test: {len(test_idx)} soal (acc={np.mean(corr_test):.2f})\n")

    thresholds = np.arange(0.40, 0.91, 0.05)

    # === 1. Pilih threshold terbaik di DEV ===
    print("=== Kalibrasi di DEV ===")
    print(f"{'thr':>6} {'AUROC_dev':>10}")
    best_thr, best_dev_auroc = None, -1
    for thr in thresholds:
        a, _ = auroc_at(embs_dev, corr_dev, thr)
        mark = ""
        if not np.isnan(a) and a > best_dev_auroc:
            best_dev_auroc, best_thr = a, thr; mark = " <-"
        print(f"{thr:>6.2f} {a:>10.4f}{mark}")

    # === 2. Laporkan di TEST (threshold dari dev) ===
    print(f"\n=== Threshold terpilih (dari DEV): {best_thr:.2f} ===")
    test_auroc, test_ent = auroc_at(embs_test, corr_test, best_thr)
    print(f">>> AUROC_test @ thr={best_thr:.2f}: {test_auroc:.4f}  (avg_ent={test_ent:.3f})")

    # === 3. Sensitivitas di TEST (robustness) ===
    print("\n=== Sensitivitas AUROC di TEST (across thresholds) ===")
    print(f"{'thr':>6} {'AUROC_test':>11}")
    test_aurocs = []
    for thr in thresholds:
        a, _ = auroc_at(embs_test, corr_test, thr)
        test_aurocs.append(a)
        print(f"{thr:>6.2f} {a:>11.4f}")

    valid = [a for a in test_aurocs if not np.isnan(a)]
    if valid:
        print(f"\nRingkasan TEST: AUROC min={min(valid):.3f}, "
              f"max={max(valid):.3f}, mean={np.mean(valid):.3f}")
        print(f"Interpretasi: {'ROBUST (stabil across threshold)' if (max(valid)-min(valid))<0.1 else 'SENSITIF thd threshold'}")

    # Untuk paper
    print("\n--- Untuk paper ---")
    print(f"Lang={lang}: threshold={best_thr:.2f} (dipilih pada dev), "
          f"AUROC_test={test_auroc:.3f}, "
          f"rentang test [{min(valid):.3f}, {max(valid):.3f}].")


if __name__ == "__main__":
    main()
