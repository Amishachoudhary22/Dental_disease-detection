import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import os

# Retrieve API key securely
api_key = st.secrets["ROBOFLOW_API_KEY"]

# Initialize Roboflow clients
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)
CLIENT2 = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=api_key
)

# Define class names for prediction
class_names = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'Hypodontia']

def create_mask_from_points(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    # Call Roboflow API for segmentation
    segmentation_result = CLIENT.infer(img, model_id="dental_disease_detection/2")  # Replace model_id!
    infected_area_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    total_area_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if predicted_class != 'Healthy':
        segmentation_predictions = segmentation_result['predictions']
        for seg_pred in segmentation_predictions:
            if seg_pred['confidence'] > 0.4:
                points = seg_pred['points']
                single_mask = create_mask_from_points(img.shape, points)
                infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)

    # Total mouth/dental area segmentation
    mouth_segmentation_result = CLIENT2.infer(img, model_id="gp-dental/2")  # Replace model_id!
    mouth_segmentation_predictions = mouth_segmentation_result['predictions']
    for mouth_seg_pred in mouth_segmentation_predictions:
        if mouth_seg_pred['confidence'] > 0.4:
            mouth_points = mouth_seg_pred['points']
            mouth_single_mask = create_mask_from_points(img.shape, mouth_points)
            total_area_mask = cv2.bitwise_or(total_area_mask, mouth_single_mask)

    infected_area_pixels = np.count_nonzero(infected_area_mask)
    total_area_pixels = np.count_nonzero(total_area_mask)

    if total_area_pixels > 0 and predicted_class != 'Healthy':
        infected_area_percentage = (infected_area_pixels / total_area_pixels) * 100 + 5
    else:
        infected_area_percentage = 0

    infected_area_percentage = min(infected_area_percentage, 90)

    return predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage

# Streamlit App
st.title('Automated Dental and Gum Health Detection WebApp Using Deep Learning')

st.write(
    "Upload an image showing your dental or gum area. The app will detect potential dental health issues, "
    "estimate the infection percentage, and highlight affected areas."
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Uploaded Dental Image", use_container_width=True)

    model_path = os.path.join(os.getcwd(), "dental_problems-2.h5")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage = predict(model, img_np)

    st.subheader(f"Prediction: {predicted_class}")
    st.subheader(f"Confidence: {confidence}%")
    st.subheader(f"Infected Area: {infected_area_percentage:.2f}%")

    st.subheader("Infection and Dental Area Segmentation")

    # Red mask for infected area
    color_mask_infected = np.zeros_like(img_np, dtype=np.uint8)
    color_mask_infected[infected_area_mask > 0] = [255, 0, 0]

    # Green mask for total dental area
    color_mask_total = np.zeros_like(img_np, dtype=np.uint8)
    color_mask_total[total_area_mask > 0] = [0, 255, 0]

    # Prevent overlap (Red priority)
    color_mask_total[infected_area_mask > 0] = [255, 0, 0]

    combined_mask = cv2.addWeighted(img_np, 0.7, color_mask_total, 0.3, 0)

    st.image(combined_mask, caption="Segmentation Overlay", use_container_width=True)
