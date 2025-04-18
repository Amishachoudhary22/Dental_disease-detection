import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import os

# --- (Keep your imports, API key, clients, class names, mapping) ---
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
        # Ensure points are integer tuples/lists for fillPoly
        points_array = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
        # Check if points form a valid polygon (at least 3 points)
        if points_array.shape[0] >= 3:
            cv2.fillPoly(mask, [points_array], 1) # Use 1 for binary mask
        else:
             st.warning(f"Not enough points ({points_array.shape[0]}) to form a polygon.")

    except (TypeError, KeyError, ValueError, OverflowError) as e: # Added OverflowError
        st.warning(f"Could not create mask from points: {points}. Error: {e}")
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
    st.write(f"Image shape: {img_shape}") # Debug: Check image dimensions

    # --- Classification ---
    predicted_class = "Unknown"
    confidence = 0
    try:
        classification_result = CLIENT.infer(img_np, model_id="sinistroodonto/1")
        if classification_result and classification_result.get('predictions'):
            # Sort predictions by confidence (descending) just in case
            sorted_predictions = sorted(classification_result['predictions'], key=lambda p: p['confidence'], reverse=True)
            top_prediction = sorted_predictions[0]
            predicted_class_raw = top_prediction['class']
            confidence = round(top_prediction['confidence'] * 100, 2)
            predicted_class = ROBOFLOW_CLASS_MAPPING.get(predicted_class_raw.lower(), "Unknown")
            st.write(f"Raw Classification: {predicted_class_raw}, Mapped: {predicted_class}, Confidence: {confidence}%") # Debug
        else:
            st.warning("No classification predictions found or 'predictions' key missing.")
            return "Unknown", 0, None, None, 0
    except Exception as e:
        st.error(f"Error during classification API call: {e}")
        return "Error", 0, None, None, 0

    # --- !!! IMPORTANT: Review and Update these IDs !!! ---
    # Using 'data_teeth/3' for Calculus/Discoloration WILL cause 100% infection rate for those.
    # You NEED specific segmentation models for them.
    disease_segmentation_model_ids = {
        'Calculus': 'data_teeth/3',             # FIXME: Needs specific Calculus segmentation model ID
        'Data caries': 'caries-sfptw/1',
        'Gingivitis': 'dental_project-ee1ur/2',
        'Mouth Ulcer': 'mouth-ulser/1',
        'Tooth Discoloration': 'data_teeth/3',  # FIXME: Needs specific Discoloration segmentation model ID
        'Hypodontia': None
    }

    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # --- Disease Area Segmentation ---
    disease_model_id = disease_segmentation_model_ids.get(predicted_class)
    if disease_model_id:
        st.write(f"Attempting disease segmentation for '{predicted_class}' using model ID: {disease_model_id}") # Debug
        # Check if using the problematic ID
        if disease_model_id == 'data_teeth/3' and predicted_class in ['Calculus', 'Tooth Discoloration']:
             st.warning(f"Warning: Using the general 'data_teeth/3' model for '{predicted_class}' segmentation. This will likely result in a 100% infected area if total area is also segmented.")

        try:
            segmentation_result = CLIENT.infer(img_np, model_id=disease_model_id)
            if 'predictions' in segmentation_result:
                st.write(f"Found {len(segmentation_result['predictions'])} predictions for disease segmentation.") # Debug
                for i, seg_pred in enumerate(segmentation_result['predictions']):
                    pred_conf = seg_pred.get('confidence', 0)
                    if pred_conf > 0.4 and 'points' in seg_pred: # Maybe adjust confidence threshold?
                        points = seg_pred['points']
                        # st.write(f"  Pred {i+1} (Conf: {pred_conf:.2f}): Points count = {len(points)}") # Debug points
                        if points:
                            single_mask = create_mask_from_points(img_shape, points)
                            infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)
                    # else: # Debug low confidence predictions
                        # st.write(f"  Pred {i+1} skipped (Conf: {pred_conf:.2f} <= 0.4 or no points)")

            else:
                 st.warning(f"No 'predictions' key in disease segmentation result for {predicted_class}.")

        except Exception as e:
            st.error(f"Error during '{predicted_class}' segmentation API call (Model: {disease_model_id}): {e}")
            st.exception(e) # Print full traceback
    elif predicted_class != 'Hypodontia':
        st.warning(f"No specific segmentation model ID found for {predicted_class}. Infected area mask will be empty.")

    # --- Total Mouth/Dental Area Segmentation ---
    total_area_model_id = "data_teeth/3"
    st.write(f"Attempting total area segmentation using model ID: {total_area_model_id}") # Debug
    try:
        mouth_segmentation_result = CLIENT2.infer(img_np, model_id=total_area_model_id)
        if 'predictions' in mouth_segmentation_result:
            st.write(f"Found {len(mouth_segmentation_result['predictions'])} predictions for total area segmentation.") # Debug
            for i, mouth_seg_pred in enumerate(mouth_segmentation_result.get('predictions', [])):
                pred_conf = mouth_seg_pred.get('confidence', 0)
                if pred_conf > 0.4 and 'points' in mouth_seg_pred: # Maybe adjust confidence threshold?
                    mouth_points = mouth_seg_pred['points']
                    # st.write(f"  Total Area Pred {i+1} (Conf: {pred_conf:.2f}): Points count = {len(mouth_points)}") # Debug points
                    if mouth_points:
                        mouth_single_mask = create_mask_from_points(img_shape, mouth_points)
                        total_area_mask = cv2.bitwise_or(total_area_mask, mouth_single_mask)
                # else: # Debug low confidence predictions
                    # st.write(f"  Total Area Pred {i+1} skipped (Conf: {pred_conf:.2f} <= 0.4 or no points)")
        else:
             st.warning(f"No 'predictions' key in total area segmentation result.")

    except Exception as e:
        st.error(f"Error during total area segmentation API call (Model: {total_area_model_id}): {e}")
        st.exception(e) # Print full traceback


    # --- **** DEBUG: Visualize Individual Masks **** ---
    if np.any(infected_area_mask): # Only show if not empty
        st.image(infected_area_mask * 255, caption=f"DEBUG: Infected Area Mask ({predicted_class})", clamp=True, channels="GRAY")
    else:
        st.write("DEBUG: Infected Area Mask is empty.")

    if np.any(total_area_mask): # Only show if not empty
        st.image(total_area_mask * 255, caption="DEBUG: Total Area Mask (data_teeth/3)", clamp=True, channels="GRAY")
    else:
        st.write("DEBUG: Total Area Mask is empty.")
    # --- **** END DEBUG **** ---


    # --- Calculate Percentage ---
    infected_area_pixels = np.count_nonzero(infected_area_mask)
    total_area_pixels = np.count_nonzero(total_area_mask)

    # --- **** DEBUG: Print Pixel Counts **** ---
    st.write(f"DEBUG: Infected Pixels Count = {infected_area_pixels}")
    st.write(f"DEBUG: Total Area Pixels Count = {total_area_pixels}")
    # --- **** END DEBUG **** ---

    infected_area_percentage = 0
    if total_area_pixels > 0 and predicted_class != 'Hypodontia':
        # Ensure calculation is done using float division
        infected_area_percentage = (float(infected_area_pixels) / float(total_area_pixels)) * 100.0
        # Clamp value between 0 and 100
        infected_area_percentage = max(0.0, min(infected_area_percentage, 100.0))
    elif predicted_class == 'Hypodontia':
        infected_area_percentage = 0
    elif total_area_pixels == 0 and predicted_class != 'Hypodontia':
         st.warning("Total dental area segmentation resulted in zero pixels. Cannot calculate percentage.")
         infected_area_percentage = 0 # Or indicate error/unknown state differently

    st.write(f"DEBUG: Calculated Percentage = {infected_area_percentage:.4f}%") # Debug final percentage

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
        img = img.convert("RGB") # Ensure RGB
        img_np = np.array(img)

        st.image(img_np, caption="Uploaded Dental Image", use_container_width=True) # Use img_np directly

        # Make sure img_np is passed, not the PIL image 'img'
        prediction_result = predict(img_np)

        if prediction_result and all(val is not None for val in prediction_result):
            predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage = prediction_result

            st.subheader(f"Prediction: {predicted_class}")
            st.subheader(f"Confidence: {confidence}%")
            st.subheader(f"Infected Area: {infected_area_percentage:.2f}%") # Display final calculated %

            st.subheader("Infection and Dental Area Segmentation")

            # Ensure masks are boolean for indexing
            infected_mask_bool = infected_area_mask.astype(bool)
            total_mask_bool = total_area_mask.astype(bool)

            # Create colored overlays
            color_mask_combined = np.zeros_like(img_np, dtype=np.uint8)
            # Apply green where total mask is true
            color_mask_combined[total_mask_bool] = [0, 255, 0] # Green
            # Apply red where infected mask is true (this will overwrite green in infected areas)
            color_mask_combined[infected_mask_bool] = [255, 0, 0] # Red

            # Blend the original image and the combined color mask
            alpha = 0.4 # Transparency of the color overlay
            beta = 1.0 - alpha # Transparency of the original image
            # Ensure dimensions match before blending
            if color_mask_combined.shape == img_np.shape:
                 combined_display = cv2.addWeighted(img_np, beta, color_mask_combined, alpha, 0.0)
                 st.image(combined_display, caption="Segmentation Overlay (Red=Infected, Green=Total)", use_container_width=True)
            else:
                 st.error(f"Shape mismatch: Image ({img_np.shape}), Combined Mask ({color_mask_combined.shape})")
                 st.image(img_np, caption="Original Image (Overlay Failed)", use_container_width=True)

        else:
            st.error("Prediction failed or returned None values. Please check the logs above or try a different image.")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.exception(e) # Show detailed error in Streamlit
