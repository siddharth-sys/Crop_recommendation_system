import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.card {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("model/agroman.pkl")

model = load_model()

# -------------------- HEADER --------------------
st.title("🌾 Smart Crop Recommendation Dashboard")
st.caption("AI-powered crop suggestions based on soil and environmental conditions")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌱 Input Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

predict_btn = st.sidebar.button("🌾 Predict")

# -------------------- DATA --------------------
crop_images = {
    "rice": "images/rice.jfif",
    "wheat": "images/wheat.jfif",
    "maize": "images/maize.jfif",
    "cotton": "images/cotton.jfif",
    "sugarcane": "images/sugarcane.jfif",
    "papaya": "images/papaya.jfif"
}

crop_info = {
    "rice": "🌾 Requires high rainfall and humidity.",
    "wheat": "🌱 Grows in moderate temperature.",
    "maize": "🌽 Needs warm climate and fertile soil.",
    "cotton": "🧵 Requires hot climate and low humidity.",
    "sugarcane": "🍬 Needs high water supply.",
    "papaya": "🥭 Tropical fruit, grows in warm regions."
}

# -------------------- PREDICTION --------------------
if predict_btn:

    if ph <= 0 or ph > 14:
        st.error("❌ pH must be between 0 and 14")
        st.stop()

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    with st.spinner("🔍 Analyzing soil conditions..."):
        prediction = model.predict(input_data)
        crop = prediction[0]

    # -------------------- RESULT CARD --------------------
    st.markdown(f"""
    <div class="card">
        <h2>🌾 Recommended Crop</h2>
        <h1 style="color:green;">{crop.upper()}</h1>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- CONFIDENCE + INPUT --------------------
    try:
        probabilities = model.predict_proba(input_data)
        confidence = np.max(probabilities) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Confidence")
            st.progress(int(confidence))
            st.write(f"{confidence:.2f}%")

            if confidence > 80:
                st.success("High confidence prediction ✅")
            elif confidence > 60:
                st.warning("Moderate confidence ⚠️")
            else:
                st.error("Low confidence ❌ Try adjusting inputs")

        with col2:
            st.markdown("### 📋 Input Summary")
            st.write(input_data)

    except Exception as e:
        st.warning(f"Confidence not available: {e}")

    # -------------------- IMAGE + INFO --------------------
    col1, col2 = st.columns(2)

    with col1:
        if crop.lower() in crop_images:
            st.image(crop_images[crop.lower()], use_container_width=True)

    with col2:
        st.markdown("### ℹ️ Crop Info")
        st.write(crop_info.get(crop.lower(), "No info available"))

    # -------------------- COMPARISON CHART --------------------
    avg_values = [50, 50, 50, 25, 60, 6.5, 100]

    compare_df = pd.DataFrame({
        "Feature": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Your Input": [N, P, K, temperature, humidity, ph, rainfall],
        "Average": avg_values
    })

    st.markdown("### 📊 Input vs Average")
    st.bar_chart(compare_df.set_index("Feature"), use_container_width=True)

    # -------------------- FEATURE IMPORTANCE --------------------
    try:
        rf_model = model.named_steps["model"]
        importances = rf_model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.markdown("### 📈 Feature Importance")
        st.bar_chart(feature_df.set_index("Feature"), use_container_width=True)

    except Exception as e:
        st.warning(f"Feature importance not available: {e}")