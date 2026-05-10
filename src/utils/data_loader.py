"""
Loader untuk TriviaQA dataset.
Menghasilkan list of dict: {question, answer, answer_aliases}
"""
from datasets import load_dataset


def load_trivia_qa(split: str = "validation", n: int = 200) -> list[dict]:
    """
    Load TriviaQA rc.nocontext.
    n  : jumlah soal yang diambil (mulai kecil dulu: 50-100)
    """
    ds = load_dataset("trivia_qa", "rc.nocontext", split=f"{split}[:{n}]")

    samples = []
    for item in ds:
        samples.append({
            "question":       item["question"],
            "answer":         item["answer"]["value"].strip().lower(),
            "answer_aliases": [a.strip().lower() for a in item["answer"]["aliases"]],
        })

    print(f"[DataLoader] Loaded {len(samples)} TriviaQA samples")
    return samples


def is_correct(prediction: str, sample: dict) -> bool:
    """
    Cek apakah prediksi benar dengan substring match ke semua alias.
    Ini metode standar TriviaQA evaluation.
    """
    pred = prediction.strip().lower()
    if sample["answer"] in pred or pred in sample["answer"]:
        return True
    for alias in sample["answer_aliases"]:
        if alias in pred or pred in alias:
            return True
    return False


# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     # 1. Load data
#     samples = load_trivia_qa(split="validation", n=50)

#     # 2. Tampilkan contoh sample pertama
#     print("\n=== Contoh Sample ===")
#     print(f"Question : {samples[0]['question']}")
#     print(f"Answer   : {samples[0]['answer']}")
#     print(f"Aliases  : {samples[0]['answer_aliases'][:3]}")

#     # 3. Test is_correct — jawaban tepat
#     print("\n=== Test is_correct ===")
#     pred_benar = samples[0]["answer"]
#     pred_salah = "jawaban ngawur"
#     print(f"Prediksi benar  '{pred_benar}' -> {is_correct(pred_benar, samples[0])}")
#     print(f"Prediksi salah  '{pred_salah}' -> {is_correct(pred_salah, samples[0])}")