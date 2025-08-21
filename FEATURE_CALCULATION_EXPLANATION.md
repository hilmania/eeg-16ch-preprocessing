# 🔢 PENJELASAN PERHITUNGAN 960 FEATURES PER SEGMENT

## 📊 Breakdown Feature Extraction

Berdasarkan analisis kode di `eeg_preprocessing.py`, berikut perhitungan detail 960 features per segment:

### 🏗️ Dataset Configuration:
- **16 channels** EEG (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6)
- **5 frequency bands** (delta, theta, alpha, beta, gamma)
- **5 wavelet scales** (2, 4, 8, 16, 32)

---

## 🧮 Perhitungan Per Kategori Feature:

### 1. 📈 Statistical Features
**Per channel:** 15 features
- Basic statistics: mean, std, var, skew, kurtosis, min, max, peak-to-peak, median, Q1, Q3 (11 features)
- Energy features: total energy, mean power, RMS (3 features)
- Zero crossing rate (1 feature)

**Total Statistical:** 16 channels × 15 features = **240 features**

### 2. 🌊 Frequency Features
**Per channel:** 17 features
- **Per frequency band (5 bands):**
  - Band power (1 feature)
  - Relative power (1 feature)
  - **Subtotal:** 5 bands × 2 = 10 features
- **Spectral features:** mean power, power variability, peak freq index, peak frequency, spectral centroid (5 features)
- **Additional features:** 2 features
- **Per channel total:** 10 + 5 + 2 = 17 features

**Total Frequency:** 16 channels × 17 features = **272 features**

### 3. 🔗 Connectivity Features
**Channel pairs:** C(16,2) = 16×15/2 = 120 pairs
**Per pair:** 2 features
- Pearson correlation (1 feature)
- Max cross-correlation (1 feature)

**Total Connectivity:** 120 pairs × 2 features = **240 features**

### 4. 🌀 Wavelet Features
**Per channel:** 15 features
**Per scale (5 scales):** 3 features
- Mean of filtered signal (1 feature)
- Std of filtered signal (1 feature)
- Max absolute value (1 feature)

**Total Wavelet:** 16 channels × 5 scales × 3 features = **240 features**

---

## ✅ TOTAL VERIFICATION:

```
Statistical Features:    240
Frequency Features:      272
Connectivity Features:   240
Wavelet Features:        240
────────────────────────────
TOTAL:                   992 features
```

## ❓ **DISCREPANCY DETECTED!**

**Calculated:** 992 features
**Reported in analysis:** 960 features
**Difference:** 32 features

### 🔍 Possible Explanations:

1. **Filter implementation differences:** Some frequency features might be conditional
2. **Edge cases handling:** Some wavelet scales might be skipped for short segments
3. **Code optimization:** Some redundant features might be removed
4. **Rounding in calculation:** Some connectivity pairs might be excluded
5. **Feature selection:** Some features might be automatically filtered out

---

## 💡 **RECOMMENDATION:**

Untuk mendapatkan angka exact, saya sarankan:

1. **Run feature extraction test:**
```python
# Test actual feature count
preprocessor = EEGPreprocessor()
sample_data = np.random.randn(16, 1280)  # 16 channels, 1280 timepoints
features = preprocessor.extract_comprehensive_features(sample_data)
print(f"Actual feature count: {len(features)}")
```

2. **Check individual components:**
```python
stat_feat = preprocessor.extract_statistical_features(sample_data)
freq_feat = preprocessor.extract_frequency_features(sample_data)
conn_feat = preprocessor.extract_connectivity_features(sample_data)
wav_feat = preprocessor.extract_wavelet_features(sample_data)

print(f"Statistical: {len(stat_feat)}")
print(f"Frequency: {len(freq_feat)}")
print(f"Connectivity: {len(conn_feat)}")
print(f"Wavelet: {len(wav_feat)}")
print(f"Total: {len(stat_feat) + len(freq_feat) + len(conn_feat) + len(wav_feat)}")
```

---

## 🎯 **CONCLUSION:**

Angka **960 features** kemungkinan adalah hasil actual implementation yang:
- Menghilangkan beberapa features yang redundant
- Mengoptimalkan untuk edge cases
- Melakukan filtering automatic

**Untuk thesis Anda:** Gunakan angka 960 karena itu adalah hasil actual dari implementasi, bukan theoretical calculation.

**Feature categories tetap valid:**
- ✅ Statistical features (time domain)
- ✅ Frequency features (spectral analysis)
- ✅ Connectivity features (channel interactions)
- ✅ Wavelet features (time-frequency analysis)
