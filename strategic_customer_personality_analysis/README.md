# Strategic Customer Personality Analysis via Unsupervised Machine Learning

This project is a production-ready Flask application that performs customer segmentation on the Kaggle **Customer Personality Analysis** dataset. It includes preprocessing, feature engineering, K-Means clustering, elbow and silhouette-based cluster selection, PCA visualization, and business-friendly cluster insights.

## Features

- Modular ML pipeline for loading, preprocessing, clustering, visualization, and insight generation
- Feature engineering for age, customer tenure, family size, spending, purchases, and engagement
- Automated cluster selection using both Elbow Method heuristics and Silhouette Score
- PCA visualization for two-dimensional exploration of discovered customer segments
- Flask UI for dataset upload, default dataset usage, charts, segment labels, and preview table
- Deployment-ready structure for Render and Azure App Service

## Project Structure

```text
strategic_customer_personality_analysis/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── marketing_campaign.csv
│   └── uploads/
└── app/
    ├── __init__.py
    ├── config.py
    ├── routes.py
    ├── services/
    │   ├── analysis_service.py
    │   ├── clustering.py
    │   ├── data_loader.py
    │   ├── insights.py
    │   ├── preprocessor.py
    │   └── visualization.py
    ├── static/
    │   ├── css/styles.css
    │   └── outputs/
    └── templates/
        ├── base.html
        └── index.html
```

## Dataset

1. Download the Kaggle **Customer Personality Analysis** dataset.
2. Place the file at `data/marketing_campaign.csv`.
3. If your downloaded file is tab-separated, keep the `.csv` extension or upload it through the web app.

Expected columns include fields such as `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, purchase counts, spend columns, campaign response columns, and `Dt_Customer`.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## How It Works

### 1. Data Preprocessing

- Standardizes column names and handles mixed CSV/TSV input
- Converts `Dt_Customer` into customer tenure in days
- Derives customer age from `Year_Birth`
- Builds `Children_Count`, `Total_Spending`, `Total_Purchases`, `Campaign_Engagement`, and `Average_Order_Value`
- Applies median imputation and scaling to numeric columns
- Applies frequent-value imputation and one-hot encoding to categorical columns

### 2. Clustering

- Evaluates K-Means from 2 to 8 clusters
- Tracks inertia for elbow-style analysis
- Tracks silhouette scores for cluster quality
- Selects an `optimal_k` using a blended elbow and silhouette heuristic

### 3. PCA Visualization

- Reduces the transformed feature space to two principal components
- Produces a cluster scatter plot for easy interpretation

### 4. Business Insights

- Labels each cluster using spending, income, campaign responsiveness, and family composition
- Generates concise actions for marketing and customer strategy teams

## Deployment

### Render

1. Push the repository to GitHub.
2. Create a new **Web Service** on Render.
3. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Add the Kaggle dataset to `data/marketing_campaign.csv` before deploy, or use upload in the UI.

### Azure App Service

1. Create a Python Web App in Azure App Service.
2. Configure the startup command as:

```bash
gunicorn app:app
```

3. Deploy the project using GitHub Actions, ZIP deploy, or Local Git.
4. Ensure the dataset is packaged with the app or uploaded through the interface.

## Notes for Production

- Replace `SECRET_KEY` in `app/config.py` with an environment-backed secret.
- Add persistent storage if you want uploaded datasets and generated plots to survive restarts.
- For large-scale usage, store artifacts in Azure Blob Storage or an S3-compatible bucket.
