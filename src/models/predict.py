import joblib
import numpy as np
import pandas as pd
from pathlib import Path

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
# 🔹 Disease List
# ================================
DISEASES = [
    "diabetes_binary",
    "heartdiseaseorattack",
    "stroke",
    "highbp"
]

# ================================
# 🔹 Load Models & Scalers
# ================================
def load_models():
    models = {}
    scalers = {}

    for disease in DISEASES:
        model_path = MODEL_DIR / f"{disease}_model.pkl"
        scaler_path = MODEL_DIR / f"{disease}_scaler.pkl"

        models[disease] = joblib.load(model_path)
        scalers[disease] = joblib.load(scaler_path)

    return models, scalers


# ================================
# 🔹 Risk Category Function
# ================================
def risk_category(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"


# ================================
# 🔹 Prediction Function
# ================================
def predict(input_data: dict):
    """
    input_data: dictionary with feature values
    """

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    models, scalers = load_models()

    results = {}
    probabilities = []

    for disease in DISEASES:

        model = models[disease]
        scaler = scalers[disease]

        # Scale input
        X_scaled = scaler.transform(input_df)

        # Prediction
        pred = model.predict(X_scaled)[0]

        # Probability (class 1)
        prob = model.predict_proba(X_scaled)[0][1]

        # Risk level
        risk = risk_category(prob)

        results[disease] = {
            "prediction": int(pred),
            "probability": float(round(prob, 4)),
            "risk_level": risk
        }

        probabilities.append(prob)

    # ================================
    # 🔥 Combined Risk Score
    # ================================
    combined_score = np.mean(probabilities)
    combined_level = risk_category(combined_score)

    results["combined_risk"] = {
        "score": float(round(combined_score, 4)),
        "level": combined_level
    }

    return results


# ================================
# 🔹 Example Run (CLI Testing)
# ================================
if __name__ == "__main__":

    sample_input = {
        'Age': 50,
        'BMI': 30,
        'HighChol': 1,
        'PhysActivity': 0,
        'Smoker': 1,
        'PreventiveCareIndex': 1,
        'RiskScore': 0.7
    }

    output = predict(sample_input)

    print("\n🔮 Prediction Results:\n")

    for disease, result in output.items():
        print(f"{disease.upper()}: {result}")