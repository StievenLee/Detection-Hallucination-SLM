from sentence_transformers import SentenceTransformer, util
import torch
import math
import gc
from .metrics_old import get_ram_usage_mb


class SemanticEntropyCalculator:

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.75,
    ):
        print(f"[SE] Loading similarity model: {model_name}")
        ram_before = get_ram_usage_mb()

        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

        ram_after = get_ram_usage_mb()
        print(f"[SE] Model loaded. RAM delta: +{ram_after - ram_before:.0f} MB")

    def _are_equivalent(self, response_a: str, response_b: str) -> bool:
        """
        Dua respons dianggap semantically equivalent jika
        cosine similarity >= threshold (bidirectional by nature
        karena cosine similarity simetris).
        """
        emb_a = self.model.encode(response_a, convert_to_tensor=True)
        emb_b = self.model.encode(response_b, convert_to_tensor=True)
        similarity = util.cos_sim(emb_a, emb_b).item()
        return similarity >= self.similarity_threshold

    def _are_equivalent_batch(
        self, responses: list[str]
    ) -> list[list[float]]:
        """
        Hitung cosine similarity matrix sekaligus untuk efisiensi.
        Lebih cepat daripada panggil _are_equivalent satu per satu.
        """
        embeddings = self.model.encode(responses, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)
        return sim_matrix

    def cluster_responses(self, responses, threshold=None):
        sim_matrix = self._are_equivalent_batch(responses)
        clusters = []
        t = threshold if threshold is not None else self.similarity_threshold

        for i in range(len(responses)):
            placed = False
            for c_id, cluster in enumerate(clusters):
                # Rata-rata similarity ke semua anggota cluster
                avg_sim = sum(sim_matrix[i][j].item() for j in cluster) / len(cluster)
                if avg_sim >= t:
                    clusters[c_id].append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])

        return clusters

    def semantic_entropy(
        self, responses: list[str], threshold: float = None
    ) -> dict:
        M = len(responses)
        if M == 0:
            return {
                "entropy": 0.0, "n_clusters": 0,
                "cluster_probs": [], "clusters": []
            }
        """threshold=None → pakai self.similarity_threshold"""
        t = threshold if threshold is not None else self.similarity_threshold
        clusters      = self.cluster_responses(responses, threshold=t)
        n_clusters    = len(clusters)
        cluster_probs = [len(c) / M for c in clusters]
        entropy       = -sum(
            p * math.log(p) for p in cluster_probs if p > 0
        )

        return {
            "entropy":       round(entropy, 6),
            "n_clusters":    n_clusters,
            "cluster_probs": [round(p, 4) for p in cluster_probs],
            "clusters":      clusters,
        }

    def unload(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[SE] Model unloaded.")