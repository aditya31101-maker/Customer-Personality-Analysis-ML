import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "replace-this-in-production")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
    DEFAULT_DATASET_PATH = BASE_DIR / "data" / "marketing_campaign.csv"
    OUTPUT_FOLDER = BASE_DIR / "app" / "static" / "outputs"
