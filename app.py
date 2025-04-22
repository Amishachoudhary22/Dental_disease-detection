import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient

# Secure API Key
api_key = st.secrets["ROBOFLOW_API_KEY"]

# Initialize Roboflow Clients
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
CLIENT2 = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Class Names
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
    if points:
        try:
            pts = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 1)
        except Exception as e:
            st.warning(f"Mask creation error: {e}")
    return mask

def encode_image_to_jpg_bytes(img_np):
    success, buffer = cv2.imencode(".jpg", img_np)
    if not success:
        raise ValueError("Image encoding failed.")
    return io.BytesIO(buffer.tobytes())

def predict(img):
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        img_np = np.array(img)
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        st.error("Unsupported image format.")
        return None, None, None, None, None

    img_shape = img_np.shape
    img_bytes = encode_image_to_jpg_bytes(img_np)

    # Classification
    predicted_class = "Unknown"
    confidence = 0
    try:
        result = CLIENT.infer(img_bytes, model_id="sinistroodonto/1")
        if result and result.get("predictions"):
            sorted_preds = sorted(result["predictions"], key=lambda p: p['confidence'], reverse=True)
            top_pred = sorted_preds[0]
            predicted_class_raw = top_pred['class']
            confidence = round(top_pred['confidence'] * 100, 2)
            predicted_class = ROBOFLOW_CLASS_MAPPING.get(predicted_class_raw.lower(), "Unknown")
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "Error", 0, None, None, 0

    # Map class to segmentation model
    seg_models = {
        'Calculus': 'data_teeth/3',
        'Data caries': 'caries-sfptw/1',
        'Gingivitis': 'data_teeth/3',
        'Mouth Ulcer': 'dental_project-xcawb/1',
        'Tooth Discoloration': 'data_teeth/3',
        'Hypodontia': None
    }

    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    disease_model_id = seg_models.get(predicted_class)

    # Disease Segmentation
    if disease_model_id:
        try:
            seg_result = CLIENT.infer(img_bytes, model_id=disease_model_id)
            for pred in seg_result.get('predictions', []):
                if pred.get("confidence", 0) > 0.1 and 'points' in pred:
                    mask = create_mask_from_points(img_shape, pred['points'])
                    infected_area_mask = cv2.bitwise_or(infected_area_mask, mask)
        except Exception as e:
            st.error(f"{predicted_class} segmentation failed: {e}")
    elif predicted_class != 'Hypodontia':
        st.warning(f"No segmentation model for: {predicted_class}")

    # Total Area Segmentation
    try:
        mouth_result = CLIENT2.infer(img_bytes, model_id="dental-ai-yerxe/3")
        for pred in mouth_result.get('predictions', []):
            if pred.get("confidence", 0) > 0.4 and 'points' in pred:
                mask = create_mask_from_points(img_shape, pred['points'])
                total_area_mask = cv2.bitwise_or(total_area_mask, mask)
    except Exception as e:
        st.error(f"Total mouth area segmentation failed: {e}")

    # Area Calculation
    infected_and_total = cv2.bitwise_and(infected_area_mask, total_area_mask)
    infected_px = np.count_nonzero(infected_and_total)
    total_px = np.count_nonzero(total_area_mask)

    infected_percent = 0
    if predicted_class == 'Hypodontia':
        infected_percent = 0
    elif total_px > 0:
        infected_percent = (infected_px / total_px) * 100
        infected_percent = min(max(infected_percent, 0.0), 100.0)
    else:
        st.warning("Total area is 0, cannot compute percentage.")

    return predicted_class, confidence, infected_area_mask, total_area_mask, infected_percent


# ----------------------- STREAMLIT UI -----------------------
st.title("Automated Dental and Gum Health Detection WebApp")

st.write(
    "Upload a dental or gum image. The app will classify the disease, "
    "segment infected areas, and calculate infection severity."
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img_np = np.array(img.convert("RGB"))
        st.image(img_np, caption="Uploaded Image", use_container_width=True)

        predicted_class, confidence, infected_mask, total_mask, percent = predict(img)

        if predicted_class and infected_mask is not None:
            st.subheader(f"Disease: {predicted_class}")
            st.subheader(f"Confidence: {confidence}%")
            st.subheader(f"Infected Area: {percent:.2f}%")

            # Overlay Visualization
            infected_bool = infected_mask.astype(bool)
            total_bool = total_mask.astype(bool)

            color_overlay = np.zeros_like(img_np)
            color_overlay[total_bool] = [0, 255, 0]       # Green
            color_overlay[infected_bool] = [255, 0, 0]    # Red

            combined = cv2.addWeighted(img_np, 0.6, color_overlay, 0.4, 0)
            st.image(combined, caption="Overlay: Red=Infected, Green=Dental Area", use_container_width=True)
        else:
            st.error("Prediction failed.")
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.exception(e)
