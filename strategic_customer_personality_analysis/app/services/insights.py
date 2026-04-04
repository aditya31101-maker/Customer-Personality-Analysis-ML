from __future__ import annotations

import pandas as pd


class InsightGenerator:
    def build_cluster_profiles(self, dataframe: pd.DataFrame, labels) -> list[dict]:
        profiled_df = dataframe.copy()
        profiled_df["Cluster"] = labels

        numeric_df = profiled_df.select_dtypes(include=["number"])
        cluster_summary = numeric_df.groupby("Cluster").mean().round(2)
        overall_means = numeric_df.mean()

        profiles: list[dict] = []
        for cluster_id, row in cluster_summary.iterrows():
            label = self._generate_label(row, overall_means)
            insights = self._generate_insights(row, overall_means)
            size = int((profiled_df["Cluster"] == cluster_id).sum())

            profiles.append(
                {
                    "cluster_id": int(cluster_id),
                    "label": label,
                    "size": size,
                    "summary": row.to_dict(),
                    "insights": insights,
                }
            )

        profiles.sort(key=lambda item: item["cluster_id"])
        return profiles

    def _generate_label(self, row: pd.Series, overall_means: pd.Series) -> str:
        high_spend = row.get("Total_Spending", 0) > overall_means.get("Total_Spending", 0)
        high_engagement = row.get("Campaign_Engagement", 0) > overall_means.get("Campaign_Engagement", 0)
        high_income = row.get("Income", 0) > overall_means.get("Income", 0)
        family_oriented = row.get("Children_Count", 0) > overall_means.get("Children_Count", 0)

        if high_spend and high_engagement and high_income:
            return "High-Value Brand Advocates"
        if high_spend and not family_oriented:
            return "Affluent Lifestyle Shoppers"
        if family_oriented and high_engagement:
            return "Promotion-Receptive Families"
        if family_oriented and not high_spend:
            return "Budget-Conscious Households"
        return "Emerging Opportunity Customers"

    def _generate_insights(self, row: pd.Series, overall_means: pd.Series) -> list[str]:
        insights: list[str] = []

        if row.get("Total_Spending", 0) > overall_means.get("Total_Spending", 0):
            insights.append("Spending is above the portfolio average, making this segment attractive for premium cross-sell plays.")
        else:
            insights.append("Spending trails the average, so value messaging and targeted bundles are likely to outperform premium positioning.")

        if row.get("Campaign_Engagement", 0) > overall_means.get("Campaign_Engagement", 0):
            insights.append("Campaign response is strong, suggesting this segment is receptive to lifecycle journeys and personalized offers.")
        else:
            insights.append("Campaign response is weaker than average, so retention should lean on simpler offers and channel optimization.")

        if row.get("NumWebPurchases", 0) > overall_means.get("NumWebPurchases", 0):
            insights.append("Digital purchasing is above average, making web-first experiences and online recommendations especially relevant.")
        else:
            insights.append("Digital purchasing is lower than average, which points to an opportunity to improve omnichannel education and convenience.")

        return insights
