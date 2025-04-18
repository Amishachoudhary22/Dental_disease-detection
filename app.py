import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import os

# Retrieve API key securely
api_key = st.secrets["ROBOFLOW_API_KEY"]

# Initialize Roboflow clients with the CORRECT API URL
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
CLIENT2 = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
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
        points_array = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
        if points_array.shape[0] >= 3:
            cv2.fillPoly(mask, [points_array], 1)
    except (TypeError, KeyError, ValueError, OverflowError) as e:
        st.warning(f"Could not create mask from points. Error: {e}")
    return mask

def predict(img):
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        st.error("Invalid image type passed to predict function.")
        return None, None, None, None, None

    img_shape = img_np.shape

    # Classification
    predicted_class = "Unknown"
    confidence = 0
    try:
        classification_result = CLIENT.infer(img_np, model_id="sinistroodonto/1")
        st.write(f"DEBUG: Raw Classification Result: {classification_result}") # Debug
        if classification_result and classification_result.get('predictions'):
            sorted_predictions = sorted(classification_result['predictions'], key=lambda p: p['confidence'], reverse=True)
            top_prediction = sorted_predictions[0]
            predicted_class_raw = top_prediction['class']
            confidence = round(top_prediction['confidence'] * 100, 2)
            predicted_class = ROBOFLOW_CLASS_MAPPING.get(predicted_class_raw.lower(), "Unknown")
            st.write(f"DEBUG: Predicted Class: {predicted_class}, Confidence: {confidence}%") # Debug
        else:
            return "Unknown", 0, None, None, 0
    except Exception as e:
        st.error(f"Error during classification API call: {e}")
        return "Error", 0, None, None, 0

    disease_segmentation_model_ids = {
        'Calculus': 'data_teeth/3',  # FIXME: Needs specific Calculus segmentation model ID
        'Data caries': 'caries-sfptw/1',
        'Gingivitis': 'gingivitis_is/1',
        'Mouth Ulcer': 'dental_project-xcawb/1',
        'Tooth Discoloration': 'data_teeth/3', # FIXME: Needs specific Discoloration segmentation model ID
        'Hypodontia': None
    }

    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # Disease Area Segmentation
    disease_model_id = disease_segmentation_model_ids.get(predicted_class)
    if disease_model_id:
        st.write(f"DEBUG: Attempting disease segmentation for '{predicted_class}' using model ID: {disease_model_id}") # Debug
        try:
            segmentation_result = CLIENT.infer(img_np, model_id=disease_model_id)
            st.write(f"DEBUG: Raw Segmentation Result BEFORE Confidence Check: {segmentation_result}") # Debug (NEW LINE)
            if 'predictions' in segmentation_result:
                st.write(f"DEBUG: Found {len(segmentation_result['predictions'])} segmentation predictions.") # Debug
                for i, seg_pred in enumerate(segmentation_result['predictions']):
                    pred_conf = seg_pred.get('confidence', 0)
                    st.write(f"DEBUG: Prediction {i+1} - Confidence: {pred_conf}, Keys: {seg_pred.keys()}") # Debug
                    if pred_conf > 0.1 and 'points' in seg_pred: # Lowered confidence threshold for debugging
                        points = seg_pred['points']
                        st.write(f"DEBUG: Prediction {i+1} - Found {len(points)} points.") # Debug
                        if points:
                            single_mask = create_mask_from_points(img_shape, points)
                            infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)
        except Exception as e:
            st.error(f"Error during '{predicted_class}' segmentation API call: {e}")
            st.exception(e)
    elif predicted_class != 'Hypodontia':
        st.warning(f"No specific segmentation model ID found for {predicted_class}. Infected area mask will be empty.")

    # Total Mouth/Dental Area Segmentation
    total_area_model_id = "dental-ai-yerxe/3"
    try:
        mouth_segmentation_result = CLIENT2.infer(img_np, model_id=total_area_model_id)
        if 'predictions' in mouth_segmentation_result:
            for mouth_seg_pred in mouth_segmentation_result.get('predictions', []):
                pred_conf = mouth_seg_pred.get('confidence', 0)
                if pred_conf > 0.4 and 'points' in mouth_seg_pred:
                    mouth_points = mouth_seg_pred['points']
                    if mouth_points:
                        mouth_single_mask = create_mask_from_points(img_shape, mouth_points)
                        total_area_mask = cv2.bitwise_or(total_area_mask, mouth_single_mask)
    except Exception as e:
        st.error(f"Error during total area segmentation API call: {e}")
        st.exception(e)

    logically_correct_infected_mask = cv2.bitwise_and(infected_area_mask, total_area_mask)
    infected_area_pixels_corrected = np.count_nonzero(logically_correct_infected_mask)
    total_area_pixels = np.count_nonzero(total_area_mask)

    infected_area_percentage = 0
    if total_area_pixels > 0 and predicted_class != 'Hypodontia':
        infected_area_percentage = (float(infected_area_pixels_corrected) / float(total_area_pixels)) * 100.0
        infected_area_percentage = max(0.0, min(infected_area_percentage, 100.0))
    elif predicted_class == 'Hypodontia':
        infected_area_percentage = 0
    elif total_area_pixels == 0 and predicted_class != 'Hypodontia':
        st.warning("Total dental area segmentation resulted in zero pixels. Cannot calculate percentage.")
        infected_area_percentage = 0

    return predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage

# --- Streamlit App Code (No changes needed) ---
st.title('Automated Dental and Gum Health Detection WebApp Using Deep Learning')

st.write(
    "Upload an image showing your dental or gum area. The app will detect potential dental health issues, "
    "estimate the infection percentage, and highlight affected areas."
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.image(img_np, caption="Uploaded Dental Image", use_container_width=True)

        prediction_result = predict(img_np)

        if prediction_result and all(val is not None for val in prediction_result):
            predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage = prediction_result

            st.subheader(f"Prediction: {predicted_class}")
            st.subheader(f"Confidence: {confidence}%")
            st.subheader(f"Infected Area: {infected_area_percentage:.2f}%")

            st.subheader("Infection and Dental Area Segmentation")

            infected_mask_bool = infected_area_mask.astype(bool)
            total_mask_bool = total_area_mask.astype(bool)

            color_mask_combined = np.zeros_like(img_np, dtype=np.uint8)
            color_mask_combined[total_mask_bool] = [0, 255, 0] # Green
            color_mask_combined[infected_mask_bool] = [255, 0, 0] # Red

            if color_mask_combined.shape == img_np.shape:
                combined_display = cv2.addWeighted(img_np, 0.6, color_mask_combined, 0.4, 0.0)
                st.image(combined_display, caption="Segmentation Overlay (Red=Infected, Green=Total)", use_container_width=True)
            else:
                st.error(f"Shape mismatch: Image ({img_np.shape}), Combined Mask ({color_mask_combined.shape})")
                st.image(img_np, caption="Original Image (Overlay Failed)", use_container_width=True)

        else:
            st.error("Prediction failed. Please check the logs above or try a different image.")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.exception(e)
