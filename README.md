# Kaggle Lead Scoring System

A production-ready machine learning pipeline to score sales leads, built with XGBoost, FastAPI, and Docker.

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

## Modeling Details
-   **Model:** XGBoost Classifier
-   **Metric:** AUC ~0.88
-   **Leakage Prevention:** Removed `Customer` column and post-conversion signals (`Payment Status`, etc.).
-   **Features:** Uses demographics, source attribution, and web behavior (`TotalVisits`, `Time Per Visit`).

### Data Cleaning Decisions
**Why we kept `NaN` values:**
Early analysis suggested removing "Non-Diabetic" leads. However, the data revealed that the `Diabetes or Prediabetes` column has 95% missing values (`NaN`). Crucially, **2,310 Customers** (paying users) have `NaN` in this column, compared to only 208 with explicit "Diabetes" status. Filtering out `NaN` values would have removed 90% of the positive class. Thus, we treat `NaN` as a valid category. "Non-Diabetic" does not exist as an explicit label in the dataset.

