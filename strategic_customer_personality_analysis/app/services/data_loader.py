from pathlib import Path

import pandas as pd


class DataLoader:
    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt"}

    @staticmethod
    def load_dataset(dataset_path: Path) -> pd.DataFrame:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. Place the Kaggle dataset there or upload one in the app."
            )

        suffix = dataset_path.suffix.lower()
        if suffix not in DataLoader.SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported file type. Please use a CSV, TSV, or TXT dataset.")

        dataframe = DataLoader._read_with_fallback(dataset_path)

        if dataframe.empty:
            raise ValueError("The dataset is empty.")

        return dataframe

    @staticmethod
    def _read_with_fallback(dataset_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(dataset_path, sep=None, engine="python")
        except Exception:
            if dataset_path.suffix.lower() in {".tsv", ".txt"}:
                return pd.read_csv(dataset_path, sep="\t")
            return pd.read_csv(dataset_path)
