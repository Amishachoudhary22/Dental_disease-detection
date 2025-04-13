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
    if not points:
        return mask
    try:
        points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
    except (TypeError, KeyError, ValueError) as e:
        st.warning(f"Could not create mask from points: {points}. Error: {e}")
    return mask

def predict(model, img):
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        st.error("Invalid image type passed to predict function.")
        return None, None, None, None, None

    img_shape = img_np.shape
    img_array = tf.keras.preprocessing.image.img_to_array(img_np)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    st.write("Raw Predictions:", predictions)  # Add this line for debugging

    predicted_class_index = np.argmax(predictions[0])
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        st.error(f"Prediction index {predicted_class_index} out of bounds for class_names.")
        predicted_class = "Unknown"
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence, None, None, 0 # Temporarily return None for masks and 0 for percentage

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

        @st.cache_resource
        def load_tf_model(model_path):
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                return None
            try:
                loaded_model = tf.keras.models.load_model(model_path, compile=False)
                st.success("Model loaded successfully!")
                return loaded_model
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

        model_path = os.path.join(os.getcwd(), "dental_problems-2.h5")
        model = load_tf_model(model_path)

        if model:
            prediction_result = predict(model, img_np)

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
        else:
            st.error("Model could not be loaded. Cannot proceed with prediction.")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.exception(e)
