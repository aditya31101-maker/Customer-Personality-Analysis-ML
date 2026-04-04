from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    optimal_k: int
    labels: np.ndarray
    inertias: list[float]
    silhouette_scores: list[float]
    model: KMeans


class ClusterAnalyzer:
    def __init__(self, min_clusters: int = 2, max_clusters: int = 8, random_state: int = 42) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def fit(self, matrix: np.ndarray) -> ClusteringResult:
        inertias: list[float] = []
        silhouette_scores: list[float] = []
        candidate_models: dict[int, KMeans] = {}

        for cluster_count in range(self.min_clusters, self.max_clusters + 1):
            model = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=20)
            labels = model.fit_predict(matrix)
            inertias.append(model.inertia_)
            silhouette_scores.append(silhouette_score(matrix, labels))
            candidate_models[cluster_count] = model

        optimal_k = self._select_optimal_cluster_count(inertias, silhouette_scores)
        best_model = candidate_models[optimal_k]
        best_labels = best_model.predict(matrix)

        return ClusteringResult(
            optimal_k=optimal_k,
            labels=best_labels,
            inertias=inertias,
            silhouette_scores=silhouette_scores,
            model=best_model,
        )

    def _select_optimal_cluster_count(self, inertias: list[float], silhouette_scores: list[float]) -> int:
        ks = list(range(self.min_clusters, self.max_clusters + 1))
        inertia_array = np.array(inertias, dtype=float)

        if len(inertia_array) < 3:
            return ks[int(np.argmax(silhouette_scores))]

        first_diff = np.diff(inertia_array)
        second_diff = np.diff(first_diff)
        elbow_index = int(np.argmax(np.abs(second_diff))) + 1
        silhouette_index = int(np.argmax(silhouette_scores))

        elbow_k = ks[elbow_index]
        silhouette_k = ks[silhouette_index]
        return int(round((elbow_k + silhouette_k) / 2))
