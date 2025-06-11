import numpy as np
import pandas as pd
from nilearn import datasets, input_data
import antropy as ant
import os

# === Atlas ve sabitleri modül yüklendiğinde bir kez yükle (Performans için) ===
try:
    print("🧠 Power2011 Atlas yükleniyor...")
    POWER_ATLAS = datasets.fetch_coords_power_2011()
    ATLAS_COORDS = list(zip(POWER_ATLAS.rois['x'], POWER_ATLAS.rois['y'], POWER_ATLAS.rois['z']))
    N_ROIS = len(ATLAS_COORDS)
    print("✅ Atlas yüklendi.")
except Exception as e:
    print(f"❌ Atlas yüklenirken hata oluştu: {e}")
    ATLAS_COORDS = None
    N_ROIS = 0


# === FinalEntropy.py dosyasından gelen özel Entropi Fonksiyonları ===
def sample_entropy_custom(ts):
    ts = np.ascontiguousarray(ts, dtype=np.float64)
    r = 0.2 * ts.std()
    return ant.sample_entropy(ts, 2, r)


def differential_entropy_custom(ts):
    ts = np.ascontiguousarray(ts, dtype=np.float64)
    std = ts.std()
    if std == 0: return 0.0  # Standart sapma sıfırsa entropi de sıfırdır
    return 0.5 * np.log(2 * np.pi * np.e * std ** 2)


def fuzzy_entropy(x, m=2, r_ratio=0.2, n=2):
    """Antropy'de bulunmayan özel bir fuzzy entropy implementasyonu."""
    r = r_ratio * np.std(x)
    if r == 0: return 0.0

    def _phi(m_val):
        N = len(x) - m_val + 1
        if N <= 1: return 0
        X = np.array([x[i:i + m_val] for i in range(N)])
        C = np.zeros(N)
        for i in range(N):
            dist = np.max(np.abs(X - X[i]), axis=1)
            C[i] = np.sum(np.exp(-np.power(dist, n) / r))
        return np.sum(C) / (N * N)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0: return 0.0

    return -np.log(phi_m1 / phi_m)


def compute_range_entropy(ts, m=2, r_ratio=0.2):
    """Antropy'de bulunmayan özel bir range entropy implementasyonu."""

    def range_distance(x, y):
        return np.max(np.abs(x - y)) - np.min(np.abs(x - y))

    def _count_similar(template_vectors, r):
        count = 0
        N = len(template_vectors)
        for i in range(N):
            for j in range(i + 1, N):
                if range_distance(template_vectors[i], template_vectors[j]) < r:
                    count += 1
        return count

    ts = np.ascontiguousarray(ts, dtype=np.float64)
    N = len(ts)
    r = r_ratio * np.std(ts)
    if r == 0: return 0.0

    Xm = np.array([ts[i:i + m] for i in range(N - m + 1)])
    Xm1 = np.array([ts[i:i + m + 1] for i in range(N - m)])
    B = _count_similar(Xm, r)
    A = _count_similar(Xm1, r)

    if B == 0 or A == 0:
        return 0.0
    return -np.log(A / B)


# === ANA FONKSİYON ===
def calculate_entropy_features(nilearn_processed_path, t_r=2.0):
    """
    Nilearn ile işlenmiş tek bir NIfTI dosyasını alır,
    ROI zaman serilerini çıkarır ve tüm entropi özelliklerini hesaplar.

    Args:
        nilearn_processed_path (str): Nilearn pipeline'ından gelen son .nii.gz dosyasının yolu.
        t_r (float): Repetition time.

    Returns:
        np.ndarray: Makine öğrenmesi modeli için girdi olabilecek 1D bir özellik vektörü.
    """
    if ATLAS_COORDS is None:
        raise RuntimeError("Power2011 atlası yüklenemedi, entropi hesaplanamaz.")

    print(f"✅ Entropi hesaplaması başlatıldı: {nilearn_processed_path}")

    # ... (The first part of the function that extracts time series is unchanged) ...
    masker_std = input_data.NiftiSpheresMasker(
        seeds=ATLAS_COORDS, radius=5, detrend=True, standardize=True,
        low_pass=0.08, high_pass=0.009, t_r=t_r
    )
    timeseries_std = masker_std.fit_transform(nilearn_processed_path)

    masker_raw = input_data.NiftiSpheresMasker(
        seeds=ATLAS_COORDS, radius=5, detrend=False, standardize=False,
        low_pass=0.08, high_pass=0.009, t_r=t_r
    )
    timeseries_raw = masker_raw.fit_transform(nilearn_processed_path)

    saen_values = [sample_entropy_custom(timeseries_std[:, i]) for i in range(N_ROIS)]
    diffen_values = [differential_entropy_custom(timeseries_raw[:, i]) for i in range(N_ROIS)]
    fuen_values = [fuzzy_entropy(timeseries_raw[:, i]) for i in range(N_ROIS)]
    rangeen_values = [compute_range_entropy(timeseries_raw[:, i]) for i in range(N_ROIS)]

    all_features = np.concatenate([
        saen_values, diffen_values, fuen_values, rangeen_values
    ])

    # --- THIS IS THE NEW PART THAT SAVES THE CSV ---
    print("💾 Entropi özellikleri CSV dosyasına kaydediliyor...")

    # Create column headers for the CSV file
    n_rois = len(saen_values)
    headers = (
            [f"ROI_{i + 1}_SaEn" for i in range(n_rois)] +
            [f"ROI_{i + 1}_DiffEn" for i in range(n_rois)] +
            [f"ROI_{i + 1}_FuEn" for i in range(n_rois)] +
            [f"ROI_{i + 1}_RaEn" for i in range(n_rois)]
    )

    # Convert the numpy array to a pandas DataFrame
    # We need to reshape the 1D array to a 2D array with one row
    features_df = pd.DataFrame(all_features.reshape(1, -1), columns=headers)

    # Define where to save the CSV file
    output_dir = os.path.dirname(nilearn_processed_path)
    csv_path = os.path.join(output_dir, 'yeni_hasta_DUZELTILMIS.csv')

    # Save the DataFrame to a CSV file
    features_df.to_csv(csv_path, index=False)
    print(f"✅ CSV dosyası başarıyla kaydedildi: {csv_path}")
    # -----------------------------------------------

    # Return the features as before
    return csv_path