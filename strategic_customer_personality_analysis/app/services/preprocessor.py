from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ProcessedData:
    original_df: pd.DataFrame
    feature_df: pd.DataFrame
    transformed_matrix: np.ndarray


class CustomerPreprocessor:
    def __init__(self) -> None:
        self.pipeline: ColumnTransformer | None = None

    def preprocess(self, dataframe: pd.DataFrame) -> ProcessedData:
        df = dataframe.copy()
        df.columns = [column.strip() for column in df.columns]

        if "Dt_Customer" in df.columns:
            df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
            reference_date = df["Dt_Customer"].max()
            df["Customer_Days"] = (reference_date - df["Dt_Customer"]).dt.days

        if "Year_Birth" in df.columns:
            df["Age"] = datetime.now().year - df["Year_Birth"]

        if {"Kidhome", "Teenhome"}.issubset(df.columns):
            df["Children_Count"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)

        spending_columns = [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
        ]
        available_spending = [column for column in spending_columns if column in df.columns]
        if available_spending:
            df["Total_Spending"] = df[available_spending].sum(axis=1)

        purchase_columns = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
        available_purchase = [column for column in purchase_columns if column in df.columns]
        if available_purchase:
            df["Total_Purchases"] = df[available_purchase].sum(axis=1)

        campaign_columns = [column for column in df.columns if column.startswith("AcceptedCmp")]
        if "Response" in df.columns:
            campaign_columns.append("Response")
        if campaign_columns:
            df["Campaign_Engagement"] = df[campaign_columns].sum(axis=1)

        if {"Total_Spending", "Total_Purchases"}.issubset(df.columns):
            df["Average_Order_Value"] = df["Total_Spending"] / df["Total_Purchases"].replace(0, np.nan)
            df["Average_Order_Value"] = df["Average_Order_Value"].fillna(0)

        removable_columns = ["ID", "Z_CostContact", "Z_Revenue", "Dt_Customer", "Year_Birth"]
        feature_df = df.drop(columns=[column for column in removable_columns if column in df.columns])

        numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

        transformed_matrix = self.pipeline.fit_transform(feature_df)
        if hasattr(transformed_matrix, "toarray"):
            transformed_matrix = transformed_matrix.toarray()

        return ProcessedData(
            original_df=df,
            feature_df=feature_df,
            transformed_matrix=transformed_matrix,
        )
