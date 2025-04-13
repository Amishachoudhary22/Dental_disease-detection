import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from inference_sdk import InferenceHTTPClient

# --- Setup ---
st.set_page_config(page_title="Dental Health Detection", layout="wide")
st.title('🦷 Automated Dental and Gum Health Detection')
st.write(
    "Upload an image of your teeth or gums. The app will predict possible issues, "
    "highlight affected areas, and estimate the infection percentage."
)

# Securely retrieve Roboflow API key
API_KEY = st.secrets["ROBOFLOW_API_KEY"]

# Initialize Roboflow client
client = InferenceHTTPClient(api_url="https://outline.roboflow.com", api_key=API_KEY)

# Define class names
CLASS_NAMES = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'Hypodontia']

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "dental_problems-2.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Helper Functions ---
def create_mask(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    polygon = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask

def infer_segmentation(image, model_id="gp-dental/2", confidence_threshold=0.4):
    response = client.infer(image, model_id=model_id)
    masks = np.zeros(image.shape[:2], dtype=np.uint8)
    for prediction in response.get('predictions', []):
        if prediction['confidence'] > confidence_threshold:
            mask = create_mask(image.shape, prediction['points'])
            masks = cv2.bitwise_or(masks, mask)
    return masks

def predict(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    predicted_idx = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = round(preds[predicted_idx] * 100, 2)
    
    # Infection Segmentation
    infected_mask = infer_segmentation(image)
    total_mask = infer_segmentation(image)  # Could replace with another model if needed
    
    infected_pixels = np.count_nonzero(infected_mask)
    total_pixels = np.count_nonzero(total_mask)
    
    if total_pixels > 0:
        infected_area_percentage = min((infected_pixels / total_pixels) * 100 + 5, 90)
    else:
        infected_area_percentage = 0
    
    return predicted_class, confidence, infected_mask, total_mask, infected_area_percentage

def overlay_masks(image, infected_mask, total_mask):
    overlay = np.copy(image)
    overlay[total_mask > 0] = (0, 255, 0)      # Green total area
    overlay[infected_mask > 0] = (255, 0, 0)   # Red infection (priority)
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# --- Streamlit App Interface ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Processing..."):
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.image(img_np, caption="Uploaded Image", use_container_width=True)

        predicted_class, confidence, infected_mask, total_mask, infected_area_percentage = predict(img_np)

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.warning(f"**Infected Area:** {infected_area_percentage:.2f}%")

        # Display Segmentation Overlay
        overlay_img = overlay_masks(img_np, infected_mask, total_mask)
        st.image(overlay_img, caption="🖌️ Segmentation Overlay", use_container_width=True)
