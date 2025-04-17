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
ROBOFLOW_CLASS_MAPPING = {
    "caries": "Data caries",
    "preview": "Calculus",
    "tooth discoloration original dataset": "Tooth Discoloration",
    "tooth discoloration": "Tooth Discoloration",
    "calculus": "Calculus",
    "gingivitis": "Gingivitis",
    "ulcer": "Mouth Ulcer",
    "hypodontia": "Hypodontia"
}

def create_mask_from_points(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if not points:
        return mask
    try:
        points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
    except (TypeError, KeyError, ValueError) as e:
        st.warning(f"Could not create mask from points: {points}. Error: {e}")
    return mask

def predict(img): # Remove the 'model' argument
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        st.error("Invalid image type passed to predict function.")
        return None, None, None, None, None

    img_shape = img_np.shape

    try:
        classification_result = CLIENT.infer(
            img_np,
            model_id="sinistroodonto/1"
        )

        if classification_result and classification_result['predictions']:
            top_prediction = classification_result['predictions'][0]
            predicted_class_raw = top_prediction['class']
            confidence = round(top_prediction['confidence'] * 100, 2)

            predicted_class = ROBOFLOW_CLASS_MAPPING.get(predicted_class_raw.lower(), "Unknown")

        else:
            st.warning("No classification predictions found.")
            return "Unknown", 0, None, None, 0

    except Exception as e:
        st.error(f"Error during classification API call: {e}")
        return "Error", 0, None, None, 0

    disease_segmentation_model_ids = {
        'Calculus': 'data_teeth/3',
        'Data caries': 'caries-sfptw/1',
        'Gingivitis': 'dental_project-ee1ur/2',
        'Mouth Ulcer': 'mouth-ulser/1',
        'Tooth Discoloration': 'data_teeth/3',
        'Hypodontia': None
    }

    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)

    disease_model_id = disease_segmentation_model_ids.get(predicted_class)
    if disease_model_id:
        try:
            segmentation_result = CLIENT.infer(img_np, model_id=disease_model_id)
            if 'predictions' in segmentation_result:
                for seg_pred in segmentation_result['predictions']:
                    if seg_pred.get('confidence', 0) > 0.4 and 'points' in seg_pred:
                        points = seg_pred['points']
                        if points:
                            single_mask = create_mask_from_points(img_shape, points)
                            infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)
        except Exception as e:
            st.error(f"Error during {predicted_class} segmentation API call: {e}")
    elif predicted_class != 'Hypodontia':
        st.warning(f"No segmentation model ID found for {predicted_class}.")

    # --- Total Mouth/Dental Area Segmentation ---
    try:
        mouth_segmentation_result = CLIENT2.infer(img_np, model_id="data_teeth/3")
        if 'predictions' in mouth_segmentation_result:
            for mouth_seg_pred in mouth_segmentation_result.get('predictions', []):
                if mouth_seg_pred.get('confidence', 0) > 0.4 and 'points' in mouth_seg_pred:
                    mouth_points = mouth_seg_pred['points']
                    if mouth_points:
                        mouth_single_mask = create_mask_from_points(img_shape, mouth_points)
                        total_area_mask = cv2.bitwise_or(total_area_mask, mouth_single_mask)
    except Exception as e:
        st.error(f"Error during total area segmentation API call: {e}")

    infected_area_pixels = np.count_nonzero(infected_area_mask)
    total_area_pixels = np.count_nonzero(total_area_mask)

    infected_area_percentage = 0
    if total_area_pixels > 0 and predicted_class != 'Hypodontia':
        infected_area_percentage = (infected_area_pixels / total_area_pixels) * 100
        infected_area_percentage = min(infected_area_percentage, 100)
    elif predicted_class == 'Hypodontia':
        infected_area_percentage = 0

    
    return predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage

# --- Rest of your Streamlit App Code ---
st.title('Automated Dental and Gum Health Detection WebApp Using Deep Learning')

st.write(
    "Upload an image showing your dental or gum area. The app will detect potential dental health issues, "
    "estimate the infection percentage, and highlight affected areas."
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        img_np = np.array(img)

        st.image(img, caption="Uploaded Dental Image", use_container_width=True)

        prediction_result = predict(img_np)
        if prediction_result and all(val is not None for val in prediction_result):
            predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage = prediction_result

            st.subheader(f"Prediction: {predicted_class}")
            st.subheader(f"Confidence: {confidence}%")
            st.subheader(f"Infected Area: {infected_area_percentage:.2f}%")

            st.subheader("Infection and Dental Area Segmentation")

            infected_mask_bool = infected_area_mask > 0
            total_mask_bool = total_area_mask > 0

            color_mask_infected = np.zeros_like(img_np, dtype=np.uint8)
            color_mask_infected[infected_mask_bool] = [255, 0, 0] # Red

            color_mask_combined = np.zeros_like(img_np, dtype=np.uint8)
            color_mask_combined[total_mask_bool] = [0, 255, 0] # Green
            color_mask_combined[infected_mask_bool] = [255, 0, 0] # Red (priority)

            alpha = 0.3
            beta = 1.0 - alpha
            combined_display = cv2.addWeighted(img_np, beta, color_mask_combined, alpha, 0.0)

            st.image(combined_display, caption="Segmentation Overlay", use_container_width=True)
        else:
            st.error("Prediction failed. Please check the logs or try a different image.")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.exception(e)
