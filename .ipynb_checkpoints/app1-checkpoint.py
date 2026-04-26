import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="centered"
)

# -------------------- LOAD MODEL (CACHED) --------------------
@st.cache_resource
def load_model():
    return joblib.load("model/agroman.pkl")

model = load_model()

# -------------------- HEADER --------------------
st.title("🌾 Smart Crop Recommendation System")
st.caption("AI-powered crop suggestions based on soil and environmental conditions")

st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader("🌱 Enter Soil & Weather Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 50)
    K = st.slider("Potassium (K)", 0, 200, 50)
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.slider("pH Value", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

st.divider()

# -------------------- BUTTONS --------------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_btn = st.button("🌱 Predict Best Crop")

with col_btn2:
    if st.button("🔄 Reset"):
        st.rerun()

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

    # Input validation
    if ph <= 0 or ph > 14:
        st.error("❌ pH must be between 0 and 14")
        st.stop()

    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    # Loading spinner
    with st.spinner("🔍 Analyzing soil conditions..."):
        prediction = model.predict(input_data)
        crop = prediction[0]

    # -------------------- OUTPUT --------------------
    st.markdown("## 🌾 Recommended Crop")
    st.markdown(f"# **{crop.upper()}**")

    # -------------------- CONFIDENCE --------------------
    try:
        probabilities = model.predict_proba(input_data)
        confidence = np.max(probabilities) * 100

        st.progress(int(confidence))
        st.caption(f"📊 Confidence: {confidence:.2f}%")

    except Exception as e:
        st.warning(f"⚠️ Confidence score not available: {e}")

    # -------------------- IMAGE --------------------
    if crop.lower() in crop_images:
        try:
            st.image(
                crop_images[crop.lower()],
                caption=f"{crop.capitalize()} Crop",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ Image not found: {e}")

    # -------------------- INFO --------------------
    st.markdown("### ℹ️ Crop Info")

    if crop.lower() in crop_info:
        st.write(crop_info[crop.lower()])
    else:
        st.write("No additional information available.")

    # -------------------- EXPLANATION --------------------
    st.markdown("### 🧠 Why this crop?")
    st.write(
        "This recommendation is based on matching your soil nutrients and "
        "environmental conditions with historical agricultural data patterns."
    )