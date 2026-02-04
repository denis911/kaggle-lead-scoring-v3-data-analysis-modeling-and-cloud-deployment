import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, encoding='latin1')
    return df

def clean_data(df):
    print("Cleaning data...")
    # 1. Replace 'Select'
    df = df.replace('Select', np.nan)
    
    # 2. Drop >40% missing (mimicking EDA logic)
    missing_percent = df.isnull().mean()
    drop_cols = missing_percent[missing_percent > 0.4].index.tolist()
    
    # Exempt Diabetes column if present
    diab_cols = [c for c in df.columns if 'Diabetes' in c and 'Prediabetes' in c]
    if diab_cols:
        diab_col = diab_cols[0]
        if diab_col in drop_cols:
            drop_cols.remove(diab_col)
            # Filter Non-Diabetics
            print(f"Filtering Non-Diabetics using {diab_col}...")
            df = df[df[diab_col] != 'Non Diabetic'].copy()
            
    df = df.drop(columns=drop_cols)
    
    return df

def prepare_features(df):
    print("Preparing features...")
    # Target
    df['Converted'] = df['Lead Stage'].apply(lambda x: 1 if x == 'Customer' else 0)
    y = df['Converted']
    
    # Exclude Leakage and ID columns
    exclude_cols = ['Prospect ID', 'Lead Origin', 'Lead Stage', 'Lead', 'Valid', 'Interested', 'New', 'Lead Status',
                    'Payment ID', 'Payment Link', 'Payment Status', 'Order Value', 'Call Disposition', 'Sub Disposition',
                    'Lead Score', 'Lead Quality', 'Engagement Score', 'Customer'] 
    
    X = df.drop(columns=['Converted'] + exclude_cols, errors='ignore')
    
    # Drop high cardinality columns to avoid explosion
    high_card_cols = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() > 50]
    if high_card_cols:
        print(f"Dropping high cardinality columns: {high_card_cols}")
        X = X.drop(columns=high_card_cols)
        
    return X, y

def train():
    # Setup
    os.makedirs('models', exist_ok=True)
    
    # Load & Clean
    df = load_data('Xyz.csv')
    df = clean_data(df)
    X, y = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features.")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Model (XGBoost)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc:.4f}")
    
    # Save
    joblib.dump(pipeline, 'models/pipeline.pkl')
    print("Model pipeline saved to models/pipeline.pkl")

if __name__ == "__main__":
    train()
