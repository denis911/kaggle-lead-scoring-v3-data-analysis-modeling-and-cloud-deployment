# Kaggle Lead Scoring System

A production-ready machine learning pipeline to score sales leads, built with XGBoost, FastAPI, and Docker.

Data source - lead (diabetic patients) scoring CRM dump for xyz healthcare company (https://www.kaggle.com/datasets/bandilswati/lead-scoring-for-xyz-company)

## Project Overview

This system predicts the probability of a lead converting into a customer using historical CRM data. It includes:
-   **EDA:** Reproducible analysis handling categorical data, missing values, and leakage prevention.
-   **Training:** An automated pipeline (`train.py`) that trains an XGBoost model with ~0.88 AUC.
-   **Inference:** A FastAPI service (`predict.py`) for real-time scoring.
-   **Deployment:** Dockerized for easy cloud deployment.

## Quick Start

### Prerequisites
-   [uv](https://github.com/astral-sh/uv) (for dependency management)
-   Python 3.9+

### Installation
```bash
# 1. Clone the repo
git clone ...

# 2. Install dependencies
uv sync
```

### 1. Data Analysis (EDA)
Run the notebook to explore the data and see model comparison:
```bash
uv run jupyter notebook EDA.ipynb
```

### 2. Train the Model
Train the model and save the artifact to `models/pipeline.pkl`:
```bash
uv run python train.py
```

### 3. Run Inference API
Start the FastAPI service (runs on port 8001):
```bash
uv run python predict.py
```
*Note: The service listens on port 8001.*

Swagger UI (interactive docs):
http://localhost:8001/docs

### 4. Test Prediction
Send a sample request:
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [{"TotalVisits": 5, "City": "Mumbai", "Lead Origin": "API", "Lead Source": "Olark Chat", "Specialization": "Select", "What is your current occupation": "Unemployed"}]}'
```

Response:
```json
{
  "predictions": [
    {
      "conversion_probability": 0.084,
      "converted_prediction": false
    }
  ]
}
```

### 5. Docker Deployment
Build and run the container:
```bash
docker build -t lead-scorer .
docker run -p 8001:8001 lead-scorer
```

## Evaluation & Insights

### Model Performance
After rigorous leakage prevention (removing `Customer` status and post-conversion indicators), we achieved the following results on the test set:

| Model | AUC Score |
| :--- | :--- |
| **XGBoost** | **0.8803** |
| Random Forest | 0.8513 |
| Logistic Regression | 0.8447 |

### Top 5 Predictive Features
Based on Random Forest feature importance, the primary drivers for conversion are:
1.  **Lead Age (36.8%)**: Older leads in the CRM (returning users) have significantly higher conversion probability.
2.  **Purchase Page Hits (23.7%)**: Visiting the `/purchase/` path is the strongest direct signal of intent.
3.  **Call Back Counter (7.3%)**: Number of follow-ups correlates strongly with eventual conversion.
4.  **Homepage Activity (5.9%)**: General engagement on the root `/` landing page.
5.  **Book Free Session (4.3%)**: Interaction with the lead-magnet session booking.

### Data Cleaning Decisions
**Why we kept `NaN` values:**
Early analysis suggested removing "Non-Diabetic" leads. However, the data revealed that the `Diabetes or Prediabetes` column has 95% missing values (`NaN`). Crucially, **2,310 Customers** (paying users) have `NaN` in this column, compared to only 208 with explicit "Diabetes" status. Filtering out `NaN` values would have removed 90% of the positive class. Thus, we treat `NaN` as a valid category. "Non-Diabetic" does not exist as an explicit label in the dataset.


