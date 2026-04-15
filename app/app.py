import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.predict import predict

# =========================
# 🎨 Page Config
# =========================
st.set_page_config(
    page_title="Early Disease Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# =========================
# 🎨 Custom CSS
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: black;
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .high {color: red; font-weight: bold;}
    .medium {color: orange; font-weight: bold;}
    .low {color: green; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# =========================
# 🏷️ Title
# =========================
st.title("🧠 Early Disease Risk Prediction System")
st.markdown("### Predict multiple diseases with AI + Explainability (SHAP)")

# =========================
# 🧭 Tabs
# =========================
tab1, tab2 = st.tabs(["🔮 Prediction", "📘 How It Works"])

# =========================
# 🔮 TAB 1: Prediction
# =========================
with tab1:

    st.sidebar.header("📊 Enter Your Details")

    age = st.sidebar.slider("Age", 18, 100, 40)
    bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
    highchol = st.sidebar.selectbox("High Cholesterol", [0, 1])
    activity = st.sidebar.selectbox("Physical Activity", [0, 1])
    smoker = st.sidebar.selectbox("Smoker", [0, 1])
    preventive = st.sidebar.slider("Preventive Care Index", 0, 5, 2)
    risk_score = st.sidebar.slider("General Risk Score", 0, 2, 1)

    if st.sidebar.button("🚀 Predict Risk"):

        input_data = {
            'Age': age,
            'BMI': bmi,
            'HighChol': highchol,
            'PhysActivity': activity,
            'Smoker': smoker,
            'PreventiveCareIndex': preventive,
            'RiskScore': risk_score
        }

        results = predict(input_data)

        st.success("✅ Prediction Completed!")

        # Combined Risk
        combined = results["combined_risk"]

        st.markdown("## 🧾 Overall Risk Assessment")

        level_class = combined["level"].lower()

        st.markdown(f"""
            <div class="card">
                <h3>Combined Risk Score: {combined['score']}</h3>
                <p class="{level_class}">Risk Level: {combined['level']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Disease-wise Results
        st.markdown("## 🩺 Disease-wise Analysis")

        cols = st.columns(2)

        i = 0
        for disease, result in results.items():

            if disease == "combined_risk":
                continue

            col = cols[i % 2]
            level_class = result["risk_level"].lower()

            with col:
                st.markdown(f"""
                    <div class="card">
                        <h3>{disease.upper()}</h3>
                        <p><b>Prediction:</b> {result['prediction']}</p>
                        <p><b>Probability:</b> {result['probability']}</p>
                        <p class="{level_class}">Risk Level: {result['risk_level']}</p>
                        <hr>
                        <b>Top Factors:</b>
                    </div>
                """, unsafe_allow_html=True)

                for factor in result["top_factors"]:
                    st.write("•", factor)

            i += 1

# =========================
# 📘 TAB 2: Feature Explanation
# =========================
with tab2:

    st.header("📘 Feature Engineering Explained")

    st.markdown("### 🧮 Preventive Care Index")

    st.code("""
PreventiveCareIndex =
    PhysActivity +
    Fruits +
    Veggies +
    (1 - HvyAlcoholConsump) +
    (1 - Smoker)
""", language="python")

    st.markdown("""
**Interpretation:**
- Higher score → healthier lifestyle
- Range: 0 to 5
""")

    st.markdown("---")

    st.markdown("### ⚠️ Risk Score")

    st.code("""
RiskScore =
    HighBP +
    HighChol +
    (1 if BMI >= 30 else 0) +
    (1 if GenHlth >= 4 else 0)
""", language="python")

    st.markdown("""
**Interpretation:**
- Higher score → higher medical risk
""")

    st.markdown("---")

    st.markdown("### 🚦 Risk Level Mapping")

    st.code("""
if score <= 1:
    Low
elif score <= 3:
    Medium
else:
    High
""", language="python")

    st.markdown("""
**Interpretation:**
- Low → minimal risk  
- Medium → moderate concern  
- High → serious attention needed  
""")

    st.markdown("---")

    st.markdown("### 🧠 Why This Matters")

    st.info("""
These engineered features help the model:
- Capture lifestyle behavior
- Combine multiple weak signals into strong predictors
- Improve prediction accuracy and interpretability
""")

# =========================
# 📌 Footer
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning + Explainable AI (SHAP)")