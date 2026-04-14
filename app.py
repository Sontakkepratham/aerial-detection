import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Aerial Classifier",
    page_icon="🛰️",
    layout="centered"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        .title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            color: #1e293b;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 30px;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="title">🛰️ Aerial Object Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to classify it as Bird 🐦 or Drone 🚁</div>', unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "model.onnx")
    session = ort.InferenceSession(model_path)
    return session

session = load_model()

IMG_SIZE = (224, 224)

# -------------------------------
# Upload Section
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction Section
# -------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize(IMG_SIZE)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Predict
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: img})
        prediction = float(output[0][0][0])

        THRESHOLD = 0.5

        if prediction > THRESHOLD:
            label = "🚁 Drone"
            confidence = prediction
            color = "red"
        else:
            label = "🐦 Bird"
            confidence = 1 - prediction
            color = "green"

        with col2:
            st.markdown(f"""
                <div class="card">
                    <h2 style="color:{color};">{label}</h2>
                    <p style="font-size:18px;">Confidence: {confidence:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence * 100))

    except Exception as e:
        st.error("Error processing image")
        st.text(str(e))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Model: MobileNetV2 + ONNX | Built for Labmentix Project")
