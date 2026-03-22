# evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/health_risk_engineered.csv")

model_features = [
    'Age', 'BMI', 'HighChol',
    'PhysActivity', 'Smoker',
    'PreventiveCareIndex', 'RiskScore'
]

multi_targets = [
    'Diabetes_binary',
    'HeartDiseaseorAttack',
    'Stroke',
    'HighBP'
]

X = df[model_features]
Y = df[multi_targets]

# =========================
# LOAD MODELS
# =========================
models = {}

for target in multi_targets:
    name = target.lower()
    
    model = joblib.load(f"models/{name}_model.pkl")
    scaler = joblib.load(f"models/{name}_scaler.pkl")
    
    models[target] = (model, scaler)

# =========================
# EVALUATE
# =========================
for target in multi_targets:
    
    model, scaler = models[target]
    
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    print(f"\n📊 Evaluation for {target}")
    print(classification_report(Y[target], y_pred))
    
    auc = roc_auc_score(Y[target], y_proba)
    print(f"ROC-AUC: {auc:.4f}")