import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import os

# Retrieve API key securely
# Ensure you have defined ROBOFLOW_API_KEY in Streamlit secrets
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
class_names = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'Hypodontia'] # Assuming 'Healthy' is implicitly handled or not in this list

def create_mask_from_points(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # Ensure points is not empty
    if not points:
        return mask
    try:
        points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
    except (TypeError, KeyError, ValueError) as e:
        st.warning(f"Could not create mask from points: {points}. Error: {e}") # Log or warn about bad points data
    return mask

def predict(model, img):
    # Ensure img is a NumPy array for shape access and TF processing
    if isinstance(img, Image.Image):
         img_np = np.array(img.convert("RGB")) # Convert PIL Image to numpy
    elif isinstance(img, np.ndarray):
         img_np = img # Already numpy
    else:
         st.error("Invalid image type passed to predict function.")
         return None, None, None, None, None # Or raise an error

    img_shape = img_np.shape

    img_array = tf.keras.preprocessing.image.img_to_array(img_np)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    # Make sure the index is within the bounds of your class_names list
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        st.error(f"Prediction index {predicted_class_index} out of bounds for class_names.")
        predicted_class = "Unknown" # Handle error case
    confidence = round(100 * (np.max(predictions[0])), 2)

    # Initialize masks
    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # --- Infected Area Segmentation ---
    # Add error handling for the API call
    try:
        # Pass the numpy array or PIL image as required by the SDK
        infected_area_model_id = "data_teeth/3" # Replace if needed
        print(f"DEBUG: Calling Infected Area Model: {infected_area_model_id}")
        st.write(f"DEBUG: Calling Infected Area Model: {infected_area_model_id}")# Use consistent image format
        if predicted_class != 'Healthy' and 'predictions' in segmentation_result: # Check if 'predictions' key exists
            segmentation_predictions = segmentation_result['predictions']
            for seg_pred in segmentation_predictions:
                # **** ADD KEY CHECK HERE ****
                if seg_pred.get('confidence', 0) > 0.4 and 'points' in seg_pred: # Use .get for confidence too for safety
                    points = seg_pred['points']
                    if points: # Ensure points list is not empty
                        single_mask = create_mask_from_points(img_shape, points)
                        infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)
    except Exception as e:
        st.error(f"Error during infected area segmentation API call: {e}")


    # --- Total Mouth/Dental Area Segmentation ---
    # Add error handling for the API call
    try:
        # Pass the numpy array or PIL image as required by the SDK
        mouth_segmentation_result = CLIENT2.infer(img_np, model_id="data_teeth/3") # Use consistent image format
        if 'predictions' in mouth_segmentation_result: # Check if 'predictions' key exists
            mouth_segmentation_predictions = mouth_segmentation_result['predictions']
            for mouth_seg_pred in mouth_segmentation_predictions:
                 # **** ADD KEY CHECK HERE ****
                if mouth_seg_pred.get('confidence', 0) > 0.4 and 'points' in mouth_seg_pred: # Use .get for confidence too for safety
                    mouth_points = mouth_seg_pred['points']
                    if mouth_points: # Ensure points list is not empty
                         mouth_single_mask = create_mask_from_points(img_shape, mouth_points)
                         total_area_mask = cv2.bitwise_or(total_area_mask, mouth_single_mask)
    except Exception as e:
        st.error(f"Error during total area segmentation API call: {e}")


    infected_area_pixels = np.count_nonzero(infected_area_mask)
    total_area_pixels = np.count_nonzero(total_area_mask)

    infected_area_percentage = 0 # Default to 0
    if total_area_pixels > 0 and predicted_class != 'Healthy':
        infected_area_percentage = (infected_area_pixels / total_area_pixels) * 100
        # Remove the arbitrary +5, calculate the actual percentage
    # else: # Redundant, already defaulted to 0
    #     infected_area_percentage = 0

    # Consider if capping at 90 is really desired or if >100% is possible due to mask overlaps/issues
    infected_area_percentage = min(infected_area_percentage, 100) # Cap at 100 typically makes more sense

    return predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage

# --- Rest of your Streamlit App Code ---
# (Make sure the main app part correctly handles the returned values from predict)

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
        img_np = np.array(img) # Convert to NumPy array for processing

        st.image(img, caption="Uploaded Dental Image", use_container_width=True)

        # Consider loading the model only once using st.cache_resource
        @st.cache_resource
        def load_tf_model(model_path):
            if not os.path.exists(model_path):
                 st.error(f"Model file not found at: {model_path}")
                 return None
            try:
                # Add compile=False if you aren't retraining/evaluating in this app
                loaded_model = tf.keras.models.load_model(model_path, compile=False)
                st.success("Model loaded successfully!")
                return loaded_model
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

        model_path = os.path.join(os.getcwd(), "dental_problems-2.h5") # Make sure this path is correct in your deployment
        model = load_tf_model(model_path)

        if model: # Proceed only if model loaded successfully
            # Call predict with the numpy array
            prediction_result = predict(model, img_np)

            # Check if prediction was successful
            if prediction_result and all(val is not None for val in prediction_result):
                 predicted_class, confidence, infected_area_mask, total_area_mask, infected_area_percentage = prediction_result

                 st.subheader(f"Prediction: {predicted_class}")
                 st.subheader(f"Confidence: {confidence}%")
                 st.subheader(f"Infected Area: {infected_area_percentage:.2f}%")

                 st.subheader("Infection and Dental Area Segmentation")

                 # Ensure masks are boolean or 0/1 for indexing
                 infected_mask_bool = infected_area_mask > 0
                 total_mask_bool = total_area_mask > 0

                 # Red mask for infected area
                 color_mask_infected = np.zeros_like(img_np, dtype=np.uint8)
                 color_mask_infected[infected_mask_bool] = [255, 0, 0] # Red

                 # Green mask for total dental area (will be overwritten by red where infected)
                 color_mask_combined = np.zeros_like(img_np, dtype=np.uint8)
                 color_mask_combined[total_mask_bool] = [0, 255, 0] # Green
                 color_mask_combined[infected_mask_bool] = [255, 0, 0] # Red (priority)


                 # Blend the original image with the colored mask
                 alpha = 0.3 # Transparency of the mask overlay
                 beta = 1.0 - alpha
                 combined_display = cv2.addWeighted(img_np, beta, color_mask_combined, alpha, 0.0)

                 st.image(combined_display, caption="Segmentation Overlay", use_container_width=True)
            else:
                 st.error("Prediction failed. Please check the logs or try a different image.")
        else:
             st.error("Model could not be loaded. Cannot proceed with prediction.")

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.exception(e) # Show full traceback in Streamlit for debugging
