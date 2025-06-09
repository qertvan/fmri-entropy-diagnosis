# ==============================================================================
# === ml_predictor.py (Revised with Data Scaling Fix) ==========================
# ==============================================================================
import joblib
import numpy as np

# This dictionary will hold all loaded models and their configurations
LOADED_MODELS = {}

# --- Central configuration for all diseases (No changes here) ---
DISEASE_CONFIG = {
    'scz': {
        'model_path': 'model_healthy_vs_scz.joblib',
        'class_names': ['Healthy', 'Schizophrenia'],
        'feature_names': ["ROI_86_SaEn", "ROI_144_SaEn", "ROI_188_SaEn", "ROI_129_FuEn", "ROI_168_DiffEn",
                          "ROI_264_FuEn", "ROI_253_RaEn", "ROI_214_SaEn", "ROI_258_DiffEn", "ROI_209_RaEn",
                          "ROI_232_RaEn", "ROI_209_SaEn", "ROI_181_RaEn", "ROI_24_SaEn", "ROI_21_FuEn", "ROI_150_SaEn",
                          "ROI_36_SaEn", "ROI_115_DiffEn", "ROI_104_SaEn", "ROI_55_DiffEn", "ROI_256_SaEn",
                          "ROI_247_SaEn", "ROI_104_RaEn", "ROI_170_DiffEn", "ROI_257_SaEn", "ROI_55_SaEn",
                          "ROI_55_RaEn", "ROI_7_SaEn", "ROI_155_SaEn", "ROI_191_SaEn", "ROI_127_SaEn", "ROI_255_SaEn",
                          "ROI_20_SaEn", "ROI_258_FuEn", "ROI_177_FuEn", "ROI_112_FuEn", "ROI_176_DiffEn",
                          "ROI_242_DiffEn", "ROI_25_SaEn", "ROI_73_DiffEn", "ROI_127_RaEn", "ROI_96_DiffEn",
                          "ROI_6_DiffEn"]
    },
    'adhd': {'model_path': 'model_healthy_vs_adhd.joblib', 'class_names': ['Healthy', 'ADHD'],
             'feature_names': ["ROI_86_SaEn", "ROI_144_SaEn", "ROI_188_SaEn", "ROI_129_FuEn", "ROI_168_DiffEn",
                               "ROI_264_FuEn", "ROI_253_RaEn", "ROI_214_SaEn", "ROI_258_DiffEn", "ROI_209_RaEn",
                               "ROI_232_RaEn", "ROI_209_SaEn", "ROI_181_RaEn", "ROI_24_SaEn", "ROI_21_FuEn"]},
    'bpd': {'model_path': 'model_healthy_vs_bpd.joblib', 'class_names': ['Healthy', 'Bipolar'], 'feature_names': 'all'}
}


def _get_feature_indices_from_names(feature_names_list):
    N_ROIS = 264;
    TOTAL_FEATURES = N_ROIS * 4
    if feature_names_list == 'all': return list(range(TOTAL_FEATURES))
    type_offsets = {'SaEn': 0, 'DiffEn': N_ROIS, 'FuEn': N_ROIS * 2, 'RaEn': N_ROIS * 3}
    indices = []
    for name in feature_names_list:
        try:
            parts = name.split('_');
            roi_num, entropy_type = int(parts[1]), parts[2]
            indices.append(type_offsets[entropy_type] + (roi_num - 1))
        except Exception:
            print(f"🚨 WARNING: Could not parse feature name '{name}'. Skipping."); continue
    return indices


def load_all_models():
    """
    Loads model packages (model and scaler) for all diseases.
    This version now checks that the loaded file is in the correct format.
    """
    global LOADED_MODELS
    print("--- Loading all available ML models and scalers ---")
    for key, config in DISEASE_CONFIG.items():
        try:
            prediction_package = joblib.load(config['model_path'])

            # --- NEW: Check if the loaded file is the correct package format ---
            if not isinstance(prediction_package,
                              dict) or 'model' not in prediction_package or 'scaler' not in prediction_package:
                print(f"🚨 ERROR: Model file '{config['model_path']}' is in the WRONG format.")
                print("   -> It must be a dictionary containing both 'model' and 'scaler'.")
                print("   -> Please re-save your model using the correct package format from your training script.")
                continue  # Skip to the next model

            model = prediction_package['model']
            scaler = prediction_package['scaler']
            feature_indices = _get_feature_indices_from_names(config['feature_names'])

            LOADED_MODELS[key] = {
                'model': model,
                'scaler': scaler,  # Store the scaler
                'feature_indices': feature_indices,
                'class_names': config['class_names']
            }
            print(f"✅ Package for '{key.upper()}' loaded successfully from '{config['model_path']}'.")
            print(f"   - Uses {len(feature_indices)} features and a saved scaler.")
        except FileNotFoundError:
            print(
                f"🚨 INFO: Model package for '{key.upper()}' not found at '{config['model_path']}'. This option will not be available.")
        except Exception as e:
            # A general catch for other unexpected errors during loading
            print(f"🚨 ERROR loading package for '{key.upper()}': {e}")


def run_ml_prediction(all_features, disease_key):
    """
    Selects features, SCALES them, and then runs the prediction.
    """
    if disease_key not in LOADED_MODELS:
        raise RuntimeError(f"Model package for '{disease_key}' is not loaded. Please check model file and format.")

    model_config = LOADED_MODELS[disease_key]
    model = model_config['model']
    scaler = model_config['scaler']  # Get the scaler
    required_indices = model_config['feature_indices']
    class_names = model_config['class_names']

    # Step 1: Select the correct feature subset
    selected_features = all_features[required_indices]

    # Reshape features to 2D array for the scaler and model
    features_2d = selected_features.reshape(1, -1)

    # --- Step 2: Apply the loaded scaler ---
    # We use .transform() ONLY. We DO NOT use .fit() or .fit_transform() here.
    scaled_features = scaler.transform(features_2d)

    # --- UPDATED DEBUGGING LOGS ---
    print("\n" + "=" * 50)
    print("--- DEBUGGING ML PREDICTION STEP ---")
    print(f"Model selected: For disease '{disease_key.upper()}'")
    print("First 5 RAW selected feature values (before scaling):")
    print(features_2d[0, :5])
    print("\nFirst 5 SCALED feature values being sent to the model:")
    print(scaled_features[0, :5])
    print("--- END DEBUG LOG ---")
    print("=" * 50 + "\n")
    # --- END OF DEBUGGING LOGS ---

    # Step 3: Get prediction probabilities using the SCALED data
    probabilities = model.predict_proba(scaled_features)[0]
    primary_diagnosis = class_names[np.argmax(probabilities)]

    # Step 4: Return the results
    return {
        'primary_diagnosis': primary_diagnosis,
        'class_names': class_names,
        'probabilities': {
            class_names[0].lower(): probabilities[0] * 100,
            class_names[1].lower(): probabilities[1] * 100
        }
    }
