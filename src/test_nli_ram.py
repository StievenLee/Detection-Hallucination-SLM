# from transformers import pipeline
# import psutil, os
# b = psutil.Process(os.getpid()).memory_info().rss / 1024**2
# nli = pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768')
# a = psutil.Process(os.getpid()).memory_info().rss / 1024**2
# print(f'NLI RAM: {a-b:.0f} MB')

from sentence_transformers import SentenceTransformer, util
import psutil, os

before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(f"Similarity model RAM: {after - before:.0f} MB")

# Test similarity
emb1 = model.encode("Artificial intelligence is the simulation of human intelligence")
emb2 = model.encode("AI mimics human cognitive abilities")
emb3 = model.encode("Jakarta is the capital of Indonesia")
print(f"Similar pair  : {util.cos_sim(emb1, emb2).item():.4f}")  # harusnya tinggi
print(f"Unrelated pair: {util.cos_sim(emb1, emb3).item():.4f}")  # harusnya rendah

# Tambah ke test_nli_ram.py
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

THRESHOLD = 0.6

test_pairs = [
    # Harusnya SAMA (True)
    ("Paris is the capital of France",
     "The capital of France is Paris",                  True),
    ("Artificial intelligence simulates human intelligence",
     "AI mimics human cognitive abilities",             True),
    ("Kecerdasan buatan meniru kemampuan manusia",
     "AI adalah simulasi kecerdasan manusia",           True),
    # Harusnya BEDA (False)
    ("Paris is the capital of France",
     "Jakarta is the capital of Indonesia",             False),
    ("Artificial intelligence simulates human intelligence",
     "Machine learning uses statistical methods",       False),
]

print(f"{'Pair':<5} {'Score':>8} {'≥0.65?':>8} {'Expected':>10} {'OK?':>6}")
print("─" * 50)
for i, (a, b, expected) in enumerate(test_pairs):
    score = util.cos_sim(
        model.encode(a), model.encode(b)
    ).item()
    result = score >= THRESHOLD
    ok = "✅" if result == expected else "❌"
    print(f"P{i+1:<4} {score:>8.4f} {str(result):>8} {str(expected):>10} {ok:>6}")