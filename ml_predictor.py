# ==============================================================================
# === ml_predictor.py (Hata Düzeltmesi Uygulandı) ==============================
# ==============================================================================
import joblib
import numpy as np
import pandas as pd  # DataFrame işlemleri için gerekli

# Bu dictionary, yüklenen tüm model paketlerini tutacak
LOADED_MODELS = {}

# --- Model Konfigürasyonu ---
DISEASE_CONFIG = {
    'scz': {
        'model_path': 'final_model_SCZ.joblib',
    },
    'adhd': {
        'model_path': 'final_model_ADHD.joblib',
    },
    'bpd': {
        'model_path': 'final_model_BPD.joblib',
    }
}

def load_all_models():
    """
    Tüm hastalıklar için model paketlerini (model, scaler, encoder, columns) yükler.
    """
    global LOADED_MODELS
    print("--- Tüm ML model paketleri yükleniyor ---")
    for key, config in DISEASE_CONFIG.items():
        try:
            prediction_package = joblib.load(config['model_path'])
            required_keys = ['model', 'scaler', 'labels_to_filter', 'FEATURES_TO_USE']
            if not isinstance(prediction_package, dict) or not all(k in prediction_package for k in required_keys):
                print(f"🚨 HATA: Model dosyası '{config['model_path']}' YANLIŞ formatta.")
                continue
            LOADED_MODELS[key] = prediction_package
            print(f"✅ Paket '{key.upper()}' için başarıyla yüklendi: '{config['model_path']}'.")
        except FileNotFoundError:
            print(f"🚨 BİLGİ: '{key.upper()}' için model paketi bulunamadı: '{config['model_path']}'.")
        except Exception as e:
            print(f"🚨 HATA: '{key.upper()}' için paket yüklenirken hata oluştu: {e}")

def run_ml_prediction(entropy_file_path, disease_key):
    """
    GÜNCELLENDİ:
    Fonksiyona gelen dosya yolunu önce bir DataFrame'e okur, sonra tahmin yapar.
    'entropy_file_path' bir dosya yolu string'i olmalıdır.
    """
    if disease_key not in LOADED_MODELS:
        raise RuntimeError(f"'{disease_key}' için model paketi yüklenemedi. Dosya yolunu ve formatını kontrol edin.")

    # Adım 1: Fonksiyona gelen dosya yolunu (string) bir csv oku.
    try:
        all_features_df = pd.read_csv(entropy_file_path)
    except FileNotFoundError:
        raise RuntimeError(f"Entropi sonuç dosyası bulunamadı: {entropy_file_path}")

    # İlgili model paketini hafızadan al
    model_config = LOADED_MODELS[disease_key]
    model = model_config['model']
    scaler = model_config['scaler']
    label_encoder = model_config['labels_to_filter']
    required_columns = model_config['FEATURES_TO_USE']

    # Adım 2: Gerekli özellikleri isimleriyle ve doğru sırada seç
    try:
        if required_columns != None:
            selected_features_df = all_features_df[required_columns]
        else:
            selected_features_df = all_features_df
    except KeyError as e:
        raise RuntimeError(f"Tahmin verisinde gerekli özellikler eksik: {e}. Sütun isimlerini kontrol edin.")

    # Adım 3: Kaydedilmiş scaler'ı uygula
    scaled_features = scaler.transform(selected_features_df)

    # Adım 4: Tahmin ve olasılıkları al
    probabilities = model.predict_proba(scaled_features)[0]
    class_names = label_encoder
    primary_diagnosis = class_names[np.argmax(probabilities)]

    # Adım 5: Sonuçları formatla
    result_probabilities = {class_name.lower(): prob * 100 for class_name, prob in zip(class_names, probabilities)}

    return {
        'primary_diagnosis': primary_diagnosis,
        'class_names': list(class_names),
        'probabilities': result_probabilities
    }