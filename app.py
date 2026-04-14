import streamlit as st
import numpy as np
from PIL import image
import onnxruntime as ort
import os

# PAGE SETUP
st.set_page_config(page_title="Aerial Classifier", layout="centered")

st.title("Aerial Object Classification")
st.write("Classify aerial image as **Bird or Drone**")

# LOAD MODEL
@st.cache_resource
def load_model():
  model_path = os.path.join(os.getcwd(), "model.onnx")
  session = ort.InferenceSession(model_path)
  return session
session = load_model()

# DEBUG (REMOVE LATER)
st.write("Model Input Shape:", session.get_inputs()[0].shape)

IMG_SIZE = (224, 224)

# UPLOAD IMAGE
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  try:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_widt=True)
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    input_shape = session.get_inputs()[0].shape
    if input_shape[1] == 3:
      img = np.transpose(img, (0, 3, 1, 2))
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})
    prediction = float(output[0][0][0])
    THRESHOLD = 0.5
    if prediction > THRESHOLD:
      label = "Drone"
      confidence = prediction
    else:
      label = "Bird"
      confidence = 1-prediction
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")
    st.progress(int(confidence *100))
  except Exception as e:
    st.error("Error processing image")
    st.text(str(e))

st.markdown("---")
st.caption("Model: MobileNetV2 + ONNX | Labmentix Project")



              
         
