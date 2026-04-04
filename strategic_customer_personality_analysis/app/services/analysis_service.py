from __future__ import annotations

from pathlib import Path

from .clustering import ClusterAnalyzer
from .data_loader import DataLoader
from .insights import InsightGenerator
from .preprocessor import CustomerPreprocessor
from .visualization import VisualizationService


class AnalysisService:
    def __init__(self, upload_folder: Path, default_dataset: Path, output_folder: Path) -> None:
        self.upload_folder = upload_folder
        self.default_dataset = default_dataset
        self.output_folder = output_folder

    def run_analysis(self, dataset_path: Path) -> dict:
        dataframe = DataLoader.load_dataset(dataset_path)

        preprocessor = CustomerPreprocessor()
        processed = preprocessor.preprocess(dataframe)

        clusterer = ClusterAnalyzer()
        clustering_result = clusterer.fit(processed.transformed_matrix)

        visualization_service = VisualizationService(self.output_folder)
        elbow_plot = visualization_service.create_elbow_silhouette_plot(
            clustering_result.inertias,
            clustering_result.silhouette_scores,
            clusterer.min_clusters,
            clustering_result.optimal_k,
        )
        pca_plot, pca_frame = visualization_service.create_pca_plot(
            processed.transformed_matrix,
            clustering_result.labels,
        )

        profiled_df = processed.original_df.copy()
        profiled_df["Cluster"] = clustering_result.labels

        insight_generator = InsightGenerator()
        cluster_profiles = insight_generator.build_cluster_profiles(processed.original_df, clustering_result.labels)

        preview_rows = profiled_df.head(10).fillna("").to_dict(orient="records")
        preview_columns = profiled_df.columns.tolist()

        return {
            "dataset_path": str(dataset_path),
            "row_count": int(profiled_df.shape[0]),
            "column_count": int(profiled_df.shape[1]),
            "optimal_k": clustering_result.optimal_k,
            "silhouette_score": round(max(clustering_result.silhouette_scores), 4),
            "cluster_counts": profiled_df["Cluster"].value_counts().sort_index().to_dict(),
            "cluster_profiles": cluster_profiles,
            "elbow_plot": elbow_plot,
            "pca_plot": pca_plot,
            "pca_points": pca_frame.round(3).head(15).to_dict(orient="records"),
            "preview_columns": preview_columns,
            "preview_rows": preview_rows,
        }
