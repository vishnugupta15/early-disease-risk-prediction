import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

# ================================
# 📁 Paths
# ================================
MODEL_DIR = Path("models")

# ================================
# 🔹 Feature Names (IMPORTANT)
# ================================
FEATURE_NAMES = [
    'Age',
    'BMI',
    'HighChol',
    'PhysActivity',
    'Smoker',
    'PreventiveCareIndex',
    'RiskScore'
]

# ================================
# 🔹 Load Model & Scaler
# ================================
def load_model_and_scaler(disease):
    model = joblib.load(MODEL_DIR / f"{disease}_model.pkl")
    scaler = joblib.load(MODEL_DIR / f"{disease}_scaler.pkl")
    return model, scaler


# ================================
# 🔹 SHAP Analysis Function
# ================================
def shap_analysis(disease, input_df):
    print(f"\n🔍 SHAP Analysis for: {disease}")

    # Load model and scaler
    model, scaler = load_model_and_scaler(disease)

    # Scale input
    X_scaled = scaler.transform(input_df)
    print("Model prediction:", model.predict_proba(X_scaled))
    
    # ================================
    # 🔥 SHAP Explainer (FINAL FIX)
    # ================================
    # Use real training data as background (BEST PRACTICE)
    background = scaler.transform(joblib.load("models/background.pkl"))

    explainer = shap.Explainer(
        model.predict_proba,
        background
    )

    shap_values = explainer(X_scaled)

    # ================================
    # 📊 Extract class 1 (positive class)
    # ================================
    shap_array = shap_values.values[:, :, 1]

    shap_df = pd.DataFrame(
        shap_array,
        columns=FEATURE_NAMES
    )

    print("\n📊 SHAP Feature Contributions (Class 1 - Disease Risk):")
    print(shap_df)

    # ================================
    # 🔥 SUMMARY PLOT
    # ================================
    # shap.summary_plot(
    #     shap_array,
    #     X_scaled,
    #     feature_names=FEATURE_NAMES
    # )

    # ================================
    # 🔥 BAR PLOT (Feature Importance)
    # ================================
    # shap.summary_plot(
    #     shap_array,
    #     X_scaled,
    #     feature_names=FEATURE_NAMES,
    #     plot_type="bar"
    # )

    # ================================
    # 🔥 FORCE PLOT (Single Prediction)
    # ================================
    shap.initjs()

    force_plot = shap.force_plot(
        shap_values.base_values[0][1],   # class 1 base value
        shap_array[0],
        input_df.iloc[0],
        feature_names=FEATURE_NAMES
    )

    return shap_df, force_plot


# ================================
# 🔹 Example Run
# ================================
if __name__ == "__main__":

    sample_input = pd.DataFrame([
        {
            'Age': 50,
            'BMI': 30,
            'HighChol': 1,
            'PhysActivity': 0,
            'Smoker': 1,
            'PreventiveCareIndex': 1,
            'RiskScore': 0.7
        }
    ])

    diseases = [
        "diabetes_binary",
        "heartdiseaseorattack",
        "stroke",
        "highbp"
    ]

    for disease in diseases:
        shap_analysis(disease, sample_input)