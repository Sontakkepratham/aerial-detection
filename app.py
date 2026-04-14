import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# PAGE SETUP

st.set_page_config(page_title="Aerial Classifier", layout="centered")

st.title("Aerial Object Classification")
st.write("Upload an aerial image to classify it as **Bird or Drone **")

# LOAD MODEL

@st.cache_resource
def load_model():
  model = tf.keras.models_load_model("final_model.h5")
  return model

model = load_model()

IMG_SIZE = (224, 224)

# UPLOAD IMAGE

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  try:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img,axix=0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    prediction = output[0][0][0]
    THRESHOLD = 0.5
    if prediction > THRESHOLD:
      label = "DRONE"
      confidence = prediction
    else:
      label = "BIRD"
      confidence = 1-prediction
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence:{confidence: .2f}")
    st.progress(int(confidence * 100))

except Exception as e:
    st.error("Error processing image. Try another image.")
    st.text(str(e))
# FOOTER
st.markdown("---")
st.caption("Built as part of Labmentix Project | Custom CNN Model")
