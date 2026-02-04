import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cells
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("# Lead Scoring EDA\n\nExploratory Data Analysis for the Diabetes Lead Scoring project."))

# Imports
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

%matplotlib inline
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading & Cleaning"))
cells.append(nbf.v4.new_code_cell("""
# Load with latin1 encoding
df = pd.read_csv('Xyz.csv', encoding='latin1')
print(f"Original Shape: {df.shape}")
df.head()
"""))

# Cleaning Logic
cells.append(nbf.v4.new_code_cell("""
# 1. Replace 'Select' with NaN
df = df.replace('Select', np.nan)

# 2. Drop columns with > 40% missing values
missing_percent = df.isnull().mean()
drop_cols = missing_percent[missing_percent > 0.4].index.tolist()

# Exempt Critical Columns
# identifying diabetes column dynamically
diab_cols = [c for c in df.columns if 'Diabetes' in c and 'Prediabetes' in c]
if diab_cols:
    diab_col = diab_cols[0]
    if diab_col in drop_cols:
        drop_cols.remove(diab_col)
        print(f"Exempted critical column: {diab_col}")

df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} columns. New Shape: {df.shape}")

# 3. Disqualification Logic
# Remove 'Non Diabetic' leads based on 'Diabetes or  Prediabetes'


# Check values before filtering
print(df[diab_col].value_counts(dropna=False))

# Filter
df_clean = df[df[diab_col] != 'Non Diabetic'].copy()
print(f"Shape after filtering Non-Diabetics: {df_clean.shape}")
"""))

# Target Engineering
cells.append(nbf.v4.new_markdown_cell("## 2. Target Engineering"))
cells.append(nbf.v4.new_code_cell("""
# Map 'Lead Stage' to Binary 'Converted'
# Customer -> 1, Else -> 0
df_clean['Converted'] = df_clean['Lead Stage'].apply(lambda x: 1 if x == 'Customer' else 0)
print(df_clean['Converted'].value_counts(normalize=True))
print(df_clean['Converted'].value_counts())
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("## 3. Feature Engineering"))
cells.append(nbf.v4.new_code_cell("""
# Select useful features
# We drop ID columns and disjoint features
exclude_cols = ['Prospect ID', 'Lead Origin', 'Lead Stage', 'Lead', 'Valid', 'Interested', 'New', 'Lead Status',
                'Payment ID', 'Payment Link', 'Payment Status', 'Order Value', 'Call Disposition', 'Sub Disposition',
                'Lead Score', 'Lead Quality', 'Engagement Score'] 

# For simplicity, let's keep numeric and low-cardinality categorical

# Identify columns
numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Remove target from features
if 'Converted' in numeric_cols: numeric_cols.remove('Converted')

print(f"Numeric: {len(numeric_cols)}")
print(f"Categorical: {len(categorical_cols)}")

# Basic Correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean[numeric_cols + ['Converted']].corr(), annot=True, cmap='coolwarm')
plt.show()
"""))

# Model Comparison
cells.append(nbf.v4.new_markdown_cell("## 4. Model Comparison"))
cells.append(nbf.v4.new_code_cell("""
# Split Data
X = df_clean.drop(columns=['Converted', 'Lead Stage'] + exclude_cols, errors='ignore')
y = df_clean['Converted']

# Handle high cardinality
# We'll drop categorical columns with > 50 unique values to prevent explosion for this baseline
high_card_cols = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() > 50]
X = X.drop(columns=high_card_cols)
print(f"Dropped high cardinality cols: {high_card_cols}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing Pipeline
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

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

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    results[name] = auc
    print(f"{name} AUC: {auc:.4f}")

    if name == 'Random Forest':
        # Feature Importance
        importances = clf.named_steps['classifier'].feature_importances_
        # We need feature names
        # Numeric names
        num_names = numeric_features.tolist()
        # Cat names
        cat_encoder = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        try:
            cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        except:
             cat_names = [f"cat_{i}" for i in range(len(cat_encoder.get_feature_names_out(categorical_features)))]
        
        feature_names = num_names + cat_names
        
        # Sort
        indices = np.argsort(importances)[::-1]
        print(f"\\nTop 10 Feature Importances:")
        for f in range(10):
            if f < len(feature_names):
               print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

best_model_name = max(results, key=results.get)
print(f"\\nBest Model: {best_model_name}")
"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("## 5. Conclusion\nThe best performing model will be selected for production."))

nb['cells'] = cells

with open('EDA.ipynb', 'w') as f:
    nbf.write(nb, f)

print("EDA.ipynb created successfully.")
