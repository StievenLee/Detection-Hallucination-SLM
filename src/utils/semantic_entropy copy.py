"""
Semantic Entropy — implementasi sesuai paper:
  "Semantic Uncertainty: Linguistic Invariances for Uncertainty
   Estimation in Natural Language Generation" (Kuhn et al., 2023)

Pipeline:
  1. Generate M sampel dari satu pertanyaan
  2. NLI clustering: dua respons masuk cluster sama
     jika saling entail secara bidirectional
  3. Hitung probabilitas tiap cluster = ukuran cluster / M
  4. Semantic entropy = -sum(p_c * log(p_c))
"""

import math
import torch
from transformers import pipeline
from .metrics import get_ram_usage_mb


class SemanticEntropyCalculator:

    def __init__(self, nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"):
        print(f"[SE] Loading NLI model: {nli_model}")
        ram_before = get_ram_usage_mb()

        # Pipeline NLI untuk klasifikasi entailment
        self.nli = pipeline(
            "text-classification",
            model=nli_model,
            device=-1,          # CPU
            top_k=None,         # kembalikan semua label & score
        )
        self.ENTAILMENT_LABEL = "entailment"

        ram_after = get_ram_usage_mb()
        print(f"[SE] NLI model loaded. RAM delta: +{ram_after - ram_before:.0f} MB")

    def _entails(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        """
        Cek apakah premise meng-entail hypothesis.
        Gunakan format: "premise [SEP] hypothesis"
        """
        result = self.nli(f"{premise} [SEP] {hypothesis}")
        # result = list of [{'label': ..., 'score': ...}, ...]
        scores = {item["label"].lower(): item["score"] for item in result[0]}
        return scores.get(self.ENTAILMENT_LABEL, 0.0) >= threshold

    def _bidirectional_entails(
        self, response_a: str, response_b: str, threshold: float = 0.5
    ) -> bool:
        """
        Dua respons dianggap semantically equivalent jika
        A entails B DAN B entails A (bidirectional).
        """
        return (
            self._entails(response_a, response_b, threshold)
            and
            self._entails(response_b, response_a, threshold)
        )

    def cluster_responses(
        self, responses: list[str], threshold: float = 0.5
    ) -> list[list[int]]:
        """
        Kelompokkan respons ke dalam semantic clusters.
        Return: list of clusters, tiap cluster = list of response indices.

        Algoritma greedy: iterasi tiap respons, cek apakah
        masuk ke cluster existing. Kalau tidak, buat cluster baru.
        """
        clusters = []       # list of list of indices
        assigned = {}       # index → cluster_id

        for i, resp_i in enumerate(responses):
            placed = False
            for c_id, cluster in enumerate(clusters):
                # Bandingkan dengan representatif pertama cluster
                rep_idx = cluster[0]
                rep = responses[rep_idx]
                if self._bidirectional_entails(resp_i, rep, threshold):
                    clusters[c_id].append(i)
                    assigned[i] = c_id
                    placed = True
                    break
            if not placed:
                assigned[i] = len(clusters)
                clusters.append([i])

        return clusters

    def semantic_entropy(
        self, responses: list[str], threshold: float = 0.5
    ) -> dict:
        """
        Hitung semantic entropy dari list of responses.

        Return dict:
          - entropy      : float, nilai SE (lebih tinggi = lebih tidak pasti)
          - n_clusters   : int, jumlah cluster semantik unik
          - cluster_probs: list[float], distribusi probabilitas tiap cluster
          - clusters     : list[list[int]], indeks respons per cluster
        """
        M = len(responses)
        if M == 0:
            return {"entropy": 0.0, "n_clusters": 0, "cluster_probs": [], "clusters": []}

        clusters = self.cluster_responses(responses, threshold)
        n_clusters = len(clusters)

        # Probabilitas tiap cluster = ukuran cluster / M
        cluster_probs = [len(c) / M for c in clusters]

        # Shannon entropy: -sum(p * log(p))
        entropy = -sum(p * math.log(p) for p in cluster_probs if p > 0)

        return {
            "entropy":       round(entropy, 6),
            "n_clusters":    n_clusters,
            "cluster_probs": [round(p, 4) for p in cluster_probs],
            "clusters":      clusters,
        }
    
    def unload(self):
        """Bebaskan NLI model dari memory."""
        del self.nli
        import gc
        gc.collect()
        print("[SE] NLI model unloaded.")
