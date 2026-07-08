"""
scoring.py (v2) — Penilaian correctness ala Farquhar et al. (Nature 2024), Opsi A.

MASALAH YANG DITANGANI:
  SLM kecil (mis. TinyLlama-1.1B) sering MENGABAIKAN instruksi "jawab singkat"
  dan tetap menjawab dalam kalimat penuh. Ini membuat F1 SQuAD standar terlalu
  rendah (precision hancur karena banyak token pengisi), sehingga jawaban yang
  sebenarnya BENAR dinilai salah.

SOLUSI:
  Sebelum menghitung F1, kita bersihkan output: buang pembuka basa-basi umum
  ("the answer is", "jawabannya adalah", dll) dan ambil kalimat pertama. Lalu
  kita gunakan gold-recall sebagai kriteria correctness: jawaban dianggap benar
  jika SELURUH token gold muncul dalam prediksi (recall gold = 1.0). Ini
  menghindari penalti terhadap token pengisi, namun tetap lebih ketat daripada
  substring mentah karena berbasis token ternormalisasi.

  Untuk pelaporan, F1 SQuAD standar tetap dihitung agar comparable dengan [3].
"""

import re
import string
from collections import Counter

_ARTICLES_EN = {"a", "an", "the"}

_PREAMBLE_PAT = re.compile(
    r'^(the answer is|answer:|the correct answer is|it is|it was|this is|'
    r'yes,?\s+(that\'?s correct[!.]?)?|jawabannya adalah|jawaban:|'
    r'jawabannya|adalah|itu adalah|yang|the)\s+',
    re.IGNORECASE,
)


def normalize_answer(text: str, lang: str = "en") -> str:
    text = str(text).lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    if lang == "en":
        tokens = [t for t in tokens if t not in _ARTICLES_EN]
    return " ".join(tokens)


def clean_prediction(prediction: str) -> str:
    p = str(prediction).strip()
    p = re.split(r'(?<=[.!?])\s', p, maxsplit=1)[0]
    prev = None
    while prev != p:
        prev = p
        p = _PREAMBLE_PAT.sub('', p).strip()
    return p


def squad_f1(prediction: str, gold: str, lang: str = "en") -> float:
    pred_toks = normalize_answer(prediction, lang).split()
    gold_toks = normalize_answer(gold, lang).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_squad_f1(prediction: str, sample: dict, clean: bool = True) -> float:
    lang = sample.get("language", "en")
    pred = clean_prediction(prediction) if clean else prediction
    golds = [sample["answer"]] + sample.get("answer_aliases", [])
    golds = [g for g in golds if str(g).strip()]
    if not golds:
        return 0.0
    return max(squad_f1(pred, g, lang) for g in golds)


def gold_recall(prediction: str, gold: str, lang: str = "en") -> float:
    pred_toks = set(normalize_answer(prediction, lang).split())
    gold_toks = normalize_answer(gold, lang).split()
    if not gold_toks:
        return 0.0
    hit = sum(1 for t in gold_toks if t in pred_toks)
    return hit / len(gold_toks)


def best_gold_recall(prediction: str, sample: dict, clean: bool = True) -> float:
    lang = sample.get("language", "en")
    pred = clean_prediction(prediction) if clean else prediction
    golds = [sample["answer"]] + sample.get("answer_aliases", [])
    golds = [g for g in golds if str(g).strip()]
    if not golds:
        return 0.0
    return max(gold_recall(pred, g, lang) for g in golds)


def is_correct_factoid(prediction: str, sample: dict,
                       mode: str = "gold_recall",
                       f1_threshold: float = 0.5,
                       recall_threshold: float = 1.0) -> bool:
    if mode == "f1":
        return best_squad_f1(prediction, sample) > f1_threshold
    return best_gold_recall(prediction, sample) >= recall_threshold


_POS_PAT = re.compile(r'\b(ya|iya|yes|benar|betul|correct|true|sesuai)\b')
_NEG_PAT = re.compile(r'\b(tidak|tak|no|not|bukan|salah|incorrect|false|nope)\b')


def extract_polarity(prediction: str):
    p = str(prediction).lower().strip()
    p = re.sub(r'"[^"]*"', ' ', p)
    p = re.sub(r"'[^']*'", ' ', p)
    first = re.split(r'[.!?\n]', p, maxsplit=1)[0]
    for scope in (first, p):
        neg = _NEG_PAT.search(scope)
        pos = _POS_PAT.search(scope)
        if neg and (not pos or neg.start() <= pos.start()):
            return 'tidak'
        if pos:
            return 'ya'
    return None


def is_correct_wrete(prediction: str, sample: dict) -> bool:
    pol = extract_polarity(prediction)
    gold = str(sample["answer"]).strip().lower()
    return pol is not None and pol == gold


def is_correct(prediction: str, sample: dict, dataset_name: str = None,
               mode: str = "gold_recall", f1_threshold: float = 0.5) -> bool:
    is_wrete = (dataset_name == "wrete"
                or str(sample.get("answer")).strip().lower() in ("ya", "tidak"))
    if is_wrete:
        return is_correct_wrete(prediction, sample)
    return is_correct_factoid(prediction, sample, mode=mode, f1_threshold=f1_threshold)


def wrete_macro_f1(predictions: list, samples: list) -> dict:
    labels = ["ya", "tidak"]
    preds = [extract_polarity(p) for p in predictions]
    golds = [str(s["answer"]).strip().lower() for s in samples]
    per_class = {}
    for lbl in labels:
        tp = sum(1 for pr, g in zip(preds, golds) if pr == lbl and g == lbl)
        fp = sum(1 for pr, g in zip(preds, golds) if pr == lbl and g != lbl)
        fn = sum(1 for pr, g in zip(preds, golds) if pr != lbl and g == lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    macro = sum(per_class[l]["f1"] for l in labels) / len(labels)
    abstain = sum(1 for pr in preds if pr is None)
    return {"macro_f1": round(macro, 4), "abstain_count": abstain, "per_class": per_class}


if __name__ == "__main__":
    cases = [
        ("The man behind The Chipmunks is David Seville, a fictional character created by the American musicia", "david seville", True),
        ('The musical "Phantom of the Opera" premiered in the US on December 10, 1993.', "sunset boulevard", False),
        ("The next British Prime Minister after Arthur Balfour was Winston Churchill.", "campbell-bannerman", False),
        ("The Japanese share index is called the Nikkei 225.", "nikkei", True),
        ('The name of Michael Jackson\'s autobiography written in 1988 is "Michael Jackson: His Own Story."', "moonwalk", False),
    ]
    print("mode=gold_recall (default):")
    for pred, gold, expected in cases:
        s = {"answer": gold, "answer_aliases": [], "language": "en"}
        got = is_correct_factoid(pred, s, mode="gold_recall")
        flag = "OK " if got == expected else "XX "
        print(f"  {flag} gold={gold!r:22} got={got} (harusnya {expected}) | "
              f"recall={best_gold_recall(pred, s):.2f} f1={best_squad_f1(pred, s):.2f}")
