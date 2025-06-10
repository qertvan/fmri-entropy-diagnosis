# run_ml_models.py

import joblib
import numpy as np
import os

# --- Model ve Özellik Bilgilerini Tanımlama ---

# Used_Features.docx dosyasından alınan özellik listeleri
# SCZ = Schizophrenia, BPD = Bipolar Disorder, ADHD = Attention-Deficit/Hyperactivity Disorder
FEATURE_SETS = {
    'SCZ': [
        "ROI_86_SaEn", "ROI_144_SaEn", "ROI_188_SaEn", "ROI_129_FuEn", "ROI_168_DiffEn", "ROI_264_FuEn",
        "ROI_253_RaEn", "ROI_214_SaEn", "ROI_258_DiffEn", "ROI_209_RaEn", "ROI_232_RaEn", "ROI_209_SaEn",
        "ROI_181_RaEn", "ROI_24_SaEn", "ROI_21_FuEn", "ROI_150_SaEn", "ROI_36_SaEn", "ROI_115_DiffEn",
        "ROI_104_SaEn", "ROI_55_DiffEn", "ROI_256_SaEn", "ROI_247_SaEn", "ROI_104_RaEn", "ROI_170_DiffEn",
        "ROI_257_SaEn", "ROI_55_SaEn", "ROI_55_RaEn", "ROI_7_SaEn", "ROI_155_SaEn", "ROI_191_SaEn",
        "ROI_127_SaEn", "ROI_255_SaEn", "ROI_20_SaEn", "ROI_258_FuEn", "ROI_177_FuEn", "ROI_112_FuEn",
        "ROI_176_DiffEn", "ROI_242_DiffEn", "ROI_25_SaEn", "ROI_73_DiffEn", "ROI_127_RaEn", "ROI_96_DiffEn",
        "ROI_6_DiffEn"
    ],
    'ADHD': [
        "ROI_86_SaEn", "ROI_144_SaEn", "ROI_188_SaEn", "ROI_129_FuEn", "ROI_168_DiffEn", "ROI_264_FuEn",
        "ROI_253_RaEn", "ROI_214_SaEn", "ROI_258_DiffEn", "ROI_209_RaEn", "ROI_232_RaEn", "ROI_209_SaEn",
        "ROI_181_RaEn", "ROI_24_SaEn", "ROI_21_FuEn"
    ],
    'BPD': 'all'  # BPD için tüm özellikler kullanılıyor
}

# Modellerin bulunduğu klasör
MODEL_DIR = 'ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)  # Klasör yoksa oluştur

MODEL_PATHS = {
    'SCZ': os.path.join(MODEL_DIR, 'SCZ_vs_Healthy.joblib'),
    'ADHD': os.path.join(MODEL_DIR, 'ADHD_vs_Healthy.joblib'),
    'BPD': os.path.join(MODEL_DIR, 'BPD_vs_Healthy.joblib')
}

# ÖNEMLİ: entropy_calculator.py'nin ürettiği sırayla tam özellik listesi
# Power2011 atlasında 264 ROI var. 4 entropi tipi: SaEn, DiffEn, FuEn, RaEn
# Toplam 264 * 4 = 1056 özellik.
N_ROIS = 264
ENTROPY_TYPES = ["SaEn", "DiffEn", "FuEn", "RaEn"]

ALL_FEATURE_NAMES = []
for etype in ENTROPY_TYPES:
    for i in range(1, N_ROIS + 1):
        ALL_FEATURE_NAMES.append(f"ROI_{i}_{etype}")


# run_ml_models.py

def run_prediction(all_entropy_features, disease_choice):
    """
    Verilen entropi özelliklerini alır, DOĞRU SIRAYLA ön işler (özellik seçimi -> scaler/pca)
    ve ML modelini çalıştırır.
    """
    print(f"\n🧠 ML Prediction started for: {disease_choice}")

    # 1. Gerekli tüm nesneleri içeren payload'u yükle
    model_path = MODEL_PATHS.get(disease_choice)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {disease_choice} at {model_path}.")

    payload = joblib.load(model_path)
    model = payload['model']
    label_encoder = payload['label_encoder']
    scaler = payload['scaler']
    pca = payload.get('pca', None)

    print(f"✅ Payload loaded from: {model_path}")
    print(f"   - Label mapping: {list(label_encoder.classes_)}")

    # 2. ÖZELLİK SEÇİMİ (Ön İşlemeden ÖNCE)
    # PCA kullanılmadıysa, modele özel özellikleri şimdi seçmeliyiz.
    features_for_preprocessing = all_entropy_features

    if not pca:
        print("   - Selecting specific features (No PCA)...")
        selected_features_names = FEATURE_SETS.get(disease_choice)

        if selected_features_names != 'all':
            feature_index_map = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
            indices_to_select = [feature_index_map[name] for name in selected_features_names]

            # Ana diziden doğru özellikleri seç
            features_for_preprocessing = all_entropy_features[indices_to_select]
            print(f"   - Selected {len(features_for_preprocessing)} features for preprocessing.")

    # Gelen veriyi 2D array'e çevir
    features_reshaped = features_for_preprocessing.reshape(1, -1)

    # 3. ÖN İŞLEME ADIMLARI (Seçilmiş Veriye Uygulanır)
    # Adım 3a: Scaler'ı uygula
    print(f"   - Applying StandardScaler to {features_reshaped.shape[1]} features...")
    features_scaled = scaler.transform(features_reshaped)

    # Adım 3b: PCA'yı uygula (eğer model PCA ile eğitildiyse)
    features_for_model = features_scaled
    if pca:
        print("   - Applying PCA...")
        features_for_model = pca.transform(features_scaled)

    # 4. Tahmin yap
    prediction_encoded = model.predict(features_for_model)[0]
    probabilities = model.predict_proba(features_for_model)[0]

    # 5. Sonuçları formatla (Bu kısım aynı)
    primary_diagnosis = label_encoder.inverse_transform([prediction_encoded])[0]
    prob_dict = {class_name.lower(): prob * 100 for i, (class_name, prob) in
                 enumerate(zip(label_encoder.classes_, probabilities))}
    confidence = prob_dict[primary_diagnosis.lower()]
    final_probs = {'healthy': prob_dict.get('hlt', 0.0), disease_choice: prob_dict.get(disease_choice.lower(),
                                                                                       prob_dict.get('bpd',
                                                                                                     prob_dict.get(
                                                                                                         'scz',
                                                                                                         prob_dict.get(
                                                                                                             'adhd',
                                                                                                             0.0))))}

    result = {'primary_diagnosis': primary_diagnosis, 'confidence': confidence, 'probabilities': final_probs}

    print(f"🎯 Prediction complete: {result}")
    return result