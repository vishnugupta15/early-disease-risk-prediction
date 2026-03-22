# train.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/health_risk_engineered.csv")

# =========================
# FEATURES & TARGETS
# =========================
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
# SPLIT
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =========================
# TRAIN XGBOOST + SMOTE
# =========================
models = {}

for target in multi_targets:
    
    print(f"🚀 Training for {target}")
    
    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, Y_train[target])
    
    # Scaling
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    
    # Model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_res_scaled, y_res)
    
    models[target] = (model, scaler)

# =========================
# SAVE MODELS
# =========================
os.makedirs("models", exist_ok=True)

for target in multi_targets:
    
    model, scaler = models[target]
    name = target.lower()
    
    joblib.dump(model, f"models/{name}_model.pkl")
    joblib.dump(scaler, f"models/{name}_scaler.pkl")

joblib.dump(X_train.sample(200), "models/background.pkl")
print("✅ All models saved successfully")