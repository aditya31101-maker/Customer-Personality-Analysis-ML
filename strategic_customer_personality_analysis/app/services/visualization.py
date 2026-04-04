from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


class VisualizationService:
    def __init__(self, output_folder: Path) -> None:
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def create_elbow_silhouette_plot(
        self, inertias: list[float], silhouette_scores: list[float], min_clusters: int, optimal_k: int
    ) -> str:
        ks = list(range(min_clusters, min_clusters + len(inertias)))
        output_path = self.output_folder / "elbow_silhouette.png"

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(ks, inertias, marker="o", color="#0f766e", label="Inertia")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("Inertia", color="#0f766e")
        ax1.tick_params(axis="y", labelcolor="#0f766e")
        ax1.axvline(optimal_k, color="#1d4ed8", linestyle="--", linewidth=1.5)

        ax2 = ax1.twinx()
        ax2.plot(ks, silhouette_scores, marker="s", color="#b45309", label="Silhouette score")
        ax2.set_ylabel("Silhouette Score", color="#b45309")
        ax2.tick_params(axis="y", labelcolor="#b45309")

        plt.title("Cluster Selection via Elbow Method and Silhouette Score")
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return f"outputs/{output_path.name}"

    def create_pca_plot(self, transformed_matrix, labels) -> tuple[str, pd.DataFrame]:
        output_path = self.output_folder / "pca_clusters.png"
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(transformed_matrix)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap="viridis",
            s=45,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.set_title("PCA Projection of Customer Segments")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        legend = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend)
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        pca_frame = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        pca_frame["Cluster"] = labels
        return f"outputs/{output_path.name}", pca_frame
