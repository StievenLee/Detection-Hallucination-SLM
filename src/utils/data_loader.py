"""
Data loader untuk semua dataset yang digunakan:
  EN: TriviaQA, BioASQ
  ID: FacQA, WReTE
"""

import pandas as pd
from pathlib import Path

# ── Helpers ──────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower()

# def is_correct(prediction: str, sample: dict) -> bool:
#     pred = _normalize(prediction)
#     if _normalize(sample["answer"]) in pred:
#         return True
#     if pred in _normalize(sample["answer"]):
#         return True
#     for alias in sample.get("answer_aliases", []):
#         if _normalize(alias) in pred or pred in _normalize(alias):
#             return True
#     return False

def is_correct(prediction: str, sample: dict) -> bool:
    pred = _normalize(prediction)
    
    # Cek semua kandidat jawaban
    all_answers = [sample["answer"]] + sample.get("answer_aliases", [])
    
    for ans in all_answers:
        ans = _normalize(ans)
        if not ans:
            continue
        
        # 1. Substring match (untuk jawaban pendek)
        if ans in pred or pred in ans:
            return True
        
        # 2. Token F1 match (untuk jawaban panjang seperti BioASQ)
        pred_tokens = set(pred.split())
        ans_tokens  = set(ans.split())
        if ans_tokens:
            overlap = len(pred_tokens & ans_tokens) / len(ans_tokens)
            if overlap >= 0.4:   # 40% token overlap = benar
                return True
    
    return False

# ── English Datasets ─────────────────────────────

def load_trivia_qa(split: str = "validation", n: int = 100) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split=f"{split}[:{n}]")
    samples = []
    for item in ds:
        samples.append({
            "question":       item["question"],
            "answer":         _normalize(item["answer"]["value"]),
            "answer_aliases": [_normalize(a) for a in item["answer"]["aliases"]],
            "language":       "en",
            "dataset":        "trivia_qa",
        })
    print(f"[Loader] TriviaQA: {len(samples)} sampel")
    return samples

# def load_bioasq(split: str = "train", n: int = 100) -> list[dict]:
#     from datasets import load_dataset

#     # Ambil lebih banyak untuk antisipasi duplikat & yang difilter
#     ds = load_dataset("kroshan/BioASQ", split=f"{split}[:{n * 5}]")

#     samples = []
#     seen_questions = set()   # ← fix duplikat

#     for item in ds:
#         question = item.get("question", "").strip()
#         text     = item.get("text", "").strip()

#         if not question or not text:
#             continue

#         # Deduplikasi
#         if question in seen_questions:
#             continue

#         # Extract jawaban dari format <answer>...</answer>
#         if "<answer>" in text and "<context>" in text:
#             answer = text.split("<answer>")[1].split("<context>")[0].strip()
#         else:
#             continue

#         if not answer:
#             continue

#         seen_questions.add(question)
#         samples.append({
#             "question":       question,
#             "answer":         _normalize(answer),
#             "answer_aliases": [],
#             "language":       "en",
#             "dataset":        "bioasq",
#         })

#         if len(samples) >= n:
#             break

#     print(f"[Loader] BioASQ: {len(samples)} sampel")
#     return samples

def load_bioasq(split: str = "factoid", n: int = 100) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("jmhb/BioASQ", split=f"{split}[:{n*2}]")

    samples = []
    seen = set()

    for item in ds:
        question = item.get("question", "").strip()
        if not question or question in seen:
            continue

        # Coba 'answer' dulu, fallback ke 'ideal_answer'
        raw_ans = item.get("answer", "") or item.get("ideal_answer", "")

        if isinstance(raw_ans, list):
            flat = []
            for a in raw_ans:
                if isinstance(a, list):
                    flat.extend(a)
                else:
                    flat.append(str(a))
            answer  = flat[0] if flat else ""
            aliases = flat[1:] if len(flat) > 1 else []
        else:
            answer  = str(raw_ans)
            aliases = []

        answer = _normalize(answer)
        if not answer:
            continue

        seen.add(question)
        samples.append({
            "question":       question,
            "answer":         answer,
            "answer_aliases": [_normalize(a) for a in aliases],
            "language":       "en",
            "dataset":        "bioasq",
        })

        if len(samples) >= n:
            break

    print(f"[Loader] BioASQ: {len(samples)} sampel")
    return samples

# ── Indonesian Datasets ───────────────────────────
def load_facqa(csv_path: str, n: int = 100) -> list[dict]:
    """
    FacQA — QA faktoid Bahasa Indonesia dari IndoNLU.

    Format CSV:
      - question  : list token pertanyaan (string repr of list)
      - passage   : list token konteks   (string repr of list)
      - seq_label : list BIO label       (string repr of list)
                    B = awal jawaban, I = lanjutan, O = bukan jawaban

    Pipeline extract jawaban:
      zip(passage_tokens, bio_labels) → ambil token ber-label B atau I
    """
    import ast

    path = Path(csv_path)
    assert path.exists(), f"File tidak ditemukan: {csv_path}"

    df = pd.read_csv(csv_path)
    print(f"[Loader] FacQA kolom: {df.columns.tolist()}")

    def parse_list(raw: str) -> list[str]:
        try:
            result = ast.literal_eval(str(raw))
            return [str(t) for t in result]
        except Exception:
            return str(raw).strip().split()

    def extract_answer(passage_tokens: list, bio_labels: list) -> str:
        """
        Ambil token dari passage yang berlabel B atau I.
        Contoh:
          passage = ["hezb-ul", "mujahedeen", ",", "kelompok", ...]
          labels  = ["B",       "I",          "O", "O", ...]
          result  = "hezb-ul mujahedeen"
        """
        answer_tokens = [
            tok for tok, lbl in zip(passage_tokens, bio_labels)
            if lbl.upper() in ("B", "I")
        ]
        return " ".join(answer_tokens).strip()

    samples = []
    seen_questions = set()

    for _, row in df.iterrows():
        raw_q   = str(row.get("question",  ""))
        raw_p   = str(row.get("passage",   ""))
        raw_lbl = str(row.get("seq_label", ""))

        if not raw_q or not raw_p or not raw_lbl:
            continue

        question_tokens  = parse_list(raw_q)
        passage_tokens   = parse_list(raw_p)
        bio_labels       = parse_list(raw_lbl)

        question = " ".join(question_tokens).strip()
        answer   = extract_answer(passage_tokens, bio_labels)

        # Skip kalau tidak ada jawaban terextract
        if not answer or not question:
            continue

        # Skip duplikat
        if question in seen_questions:
            continue
        seen_questions.add(question)

        samples.append({
            "question":       question,
            "answer":         _normalize(answer),
            "answer_aliases": [],
            "language":       "id",
            "dataset":        "facqa",
        })

        if len(samples) >= n:
            break

    print(f"[Loader] FacQA: {len(samples)} sampel")
    return samples

def load_wrete(csv_path: str, n: int = 100) -> list[dict]:
    path = Path(csv_path)
    assert path.exists(), f"File tidak ditemukan: {csv_path}"

    df = pd.read_csv(csv_path)
    print(f"[Loader] WReTE kolom: {df.columns.tolist()}")

    LABEL_MAP = {
        "entail_or_paraphrase": "ya",
        "notentail":            "tidak",
    }

    samples = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).lower().strip()
        if label not in LABEL_MAP:
            continue

        # ✅ Fix: pakai sent_A & sent_B sesuai kolom asli
        premise    = str(row.get("sent_A", "")).strip()
        hypothesis = str(row.get("sent_B", "")).strip()

        if not premise or not hypothesis:
            continue

        question = (
            f"Berdasarkan pernyataan: '{premise}'\n"
            f"Apakah pernyataan berikut benar? '{hypothesis}'"
        )

        samples.append({
            "question":       question,
            "answer":         LABEL_MAP[label],
            "answer_aliases": ["benar"] if LABEL_MAP[label] == "ya" else ["salah", "tidak benar"],
            "language":       "id",
            "dataset":        "wrete",
        })

        if len(samples) >= n:
            break

    print(f"[Loader] WReTE: {len(samples)} sampel")
    return samples

# ── Utility ───────────────────────────────────────

def load_dataset_by_name(
    name: str,
    split: str = "validation",
    n: int = 100,
    csv_path: str = None,
) -> list[dict]:
    """
    Single entry point untuk semua dataset.

    name: "trivia_qa" | "bioasq" | "facqa" | "wrete"
    """
    loaders = {
        "trivia_qa": lambda: load_trivia_qa(split, n),
        "bioasq":    lambda: load_bioasq(split, n),
        "facqa":     lambda: load_facqa(csv_path, n),
        "wrete":     lambda: load_wrete(csv_path, n),
    }
    assert name in loaders, f"Dataset tidak dikenal: {name}. Pilihan: {list(loaders.keys())}"
    return loaders[name]()