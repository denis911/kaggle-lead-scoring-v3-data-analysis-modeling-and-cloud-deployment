import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

# Initialize App
app = FastAPI(title="Lead Scoring API", version="1.0")

import numpy as np

# Load Model
try:
    model_pipeline = joblib.load('models/pipeline.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

class LeadData(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None}

@app.post("/predict")
def predict(payload: LeadData):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(payload.data)
        
        # Preprocessing rules (Must match train.py logic that isn't in pipeline)
        # Note: The pipeline handles Imputation, Scaling, Encoding.
        # But 'Select' -> NaN mapping was done BEFORE pipeline in train.py
        # We must replicate that here.
        
        # 1. Replace 'Select' with NaN
        df = df.replace('Select', np.nan)
        
        # ADDED: Align columns with model expectations
        if hasattr(model_pipeline, 'feature_names_in_'):
            expected_cols = model_pipeline.feature_names_in_
            # Add missing cols with NaN
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan
            # Reorder/Drop extra
            df = df[expected_cols]
        
        # Make predictions
        probabilities = model_pipeline.predict_proba(df)[:, 1]

        
        # Result
        results = [
            {"conversion_probability": float(prob), "converted_prediction": bool(prob > 0.5)}
            for prob in probabilities
        ]
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

