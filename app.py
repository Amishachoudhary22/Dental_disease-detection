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
        # Replace with your actual Roboflow Model ID and Version for classification
        classification_result = CLIENT.classify(
            img_np,
            model_id="dental_disease_detection/2",
            model_version=YOUR_CLASSIFICATION_MODEL_VERSION_NUMBER
        )

        if classification_result and classification_result['predictions']:
            top_prediction = classification_result['predictions'][0] # Get the top prediction
            predicted_class = top_prediction['class']
            confidence = round(top_prediction['confidence'] * 100, 2)
        else:
            st.warning("No classification predictions found.")
            return "Unknown", 0, None, None, 0

    except Exception as e:
        st.error(f"Error during classification API call: {e}")
        return "Error", 0, None, None, 0

    disease_segmentation_model_ids = {
        'Calculus': 'tooth_seg-yqnzk/3',
        'Data caries': 'data_teeth/3',
        'Gingivitis': 'gingivitis-n2cjt/3',
        'Mouth Ulcer': 'gingivitis-n2cjt/3',
        'Tooth Discoloration': 'gingivitis-n2cjt/3',
        'Hypodontia': None
    }

    infected_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    total_area_mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # --- Infected Area Segmentation (Disease-Specific) ---
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
            for mouth_seg_pred in mouth_segmentation_predictions:
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
