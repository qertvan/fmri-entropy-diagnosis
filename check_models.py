# check_models.py
import joblib
import os

# Modellerin bulunduğu klasör ve isimleri
MODEL_DIR = 'ml_models'
model_filenames = {
    'SCZ': 'SCZ_vs_Healthy.joblib',
    'ADHD': 'ADHD_vs_Healthy.joblib',
    'BPD': 'BPD_vs_Healthy.joblib',
}

print("--- Model ve Label Encoder Kontrolü ---")

for disease, filename in model_filenames.items():
    path = os.path.join(MODEL_DIR, filename)
    print(f"\nkontrol ediliyor: {path}")

    if not os.path.exists(path):
        print(">> HATA: Model dosyası bulunamadı!")
        continue

    try:
        payload = joblib.load(path)
        model = payload['model']
        label_encoder = payload['label_encoder']

        print(f"  ✅ Model Yüklendi: {type(model).__name__}")
        print(f"  ✅ Label Encoder Sınıfları: {list(label_encoder.classes_)}")

        # Beklenen sonucu kontrol et
        if disease in list(label_encoder.classes_):
            print("  >> DURUM: Label Encoder doğru görünüyor.")
        else:
            print("  >> HATA: Label Encoder bu hastalık için yanlış etiketler içeriyor!")

    except Exception as e:
        print(f">> HATA: Model yüklenirken bir sorun oluştu: {e}")

print("\n--- Kontrol Tamamlandı ---")