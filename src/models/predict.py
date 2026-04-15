import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import shap

# ================================
# 📁 Paths
# ================================
MODEL_DIR = Path("models")

# ================================
# 🔹 Feature Names
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
# 🔹 Load Models (LOAD ONCE)
# ================================
def load_models():
    models = {}
    scalers = {}

    for disease in DISEASES:
        models[disease] = joblib.load(MODEL_DIR / f"{disease}_model.pkl")
        scalers[disease] = joblib.load(MODEL_DIR / f"{disease}_scaler.pkl")

    return models, scalers


# Load once globally
MODELS, SCALERS = load_models()

# ================================
# 🔹 SHAP Explanation
# ================================
def generate_explanation(model, scaler, input_df):
    X_scaled = scaler.transform(input_df)

    # Better background (IMPORTANT)
    background = np.zeros((50, X_scaled.shape[1]))

    explainer = shap.Explainer(model.predict_proba, background)
    shap_values = explainer(X_scaled)

    # Correct extraction
    values = shap_values.values

    # Handle shapes safely
    if len(values.shape) == 3:
        values = values[0, :, 1]   # binary class 1
    else:
        values = values[0]

    feature_impact = list(zip(FEATURE_NAMES, values))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    explanation = []

    for feature, value in feature_impact[:5]:
        if abs(value) < 1e-4:
            continue  # skip meaningless features

        direction = "" if value > 0 else ""
        explanation.append(f"{feature} {direction}")

    # fallback (if empty)
    if not explanation:
        explanation = ["No strong contributing factors detected"]

    return explanation

# ================================
# 🔹 Risk Category
# ================================
def risk_category(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"


# ================================
# 🔹 Prediction Function
# ================================
def predict(input_data: dict):

    # Ensure correct feature order
    input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

    results = {}
    probabilities = []

    for disease in DISEASES:

        model = MODELS[disease]
        scaler = SCALERS[disease]

        X_scaled = scaler.transform(input_df)

        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0][1])

        risk = risk_category(prob)

        explanation = generate_explanation(model, scaler, input_df)

        results[disease] = {
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": risk,
            "top_factors": explanation
        }

        probabilities.append(prob)

    # ================================
    # 🔥 Combined Risk
    # ================================
    combined_score = float(np.mean(probabilities))

    results["combined_risk"] = {
        "score": round(combined_score, 4),
        "level": risk_category(combined_score)
    }

    return results


# ================================
# 🔹 Example Run
# ================================
if __name__ == "__main__":

    sample_input = {
        'Age': 25,
        'BMI': 21.5,
        'HighChol': 0,
        'PhysActivity': 1,
        'Smoker': 0,
        'PreventiveCareIndex': 5,
        'RiskScore': 0.7
    }

    output = predict(sample_input)

    print("\n🔮 Prediction Results:\n")

    for disease, result in output.items():
        print(f"{disease.upper()}:")
        print(result)
        print("-" * 40)