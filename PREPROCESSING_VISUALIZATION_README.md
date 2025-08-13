# Visualisasi Perbandingan Preprocessing EEG

Saya telah menambahkan fitur komprehensif untuk visualisasi perbandingan data EEG sebelum dan setelah preprocessing. Berikut adalah penjelasan lengkap:

## ✨ Fitur Utama yang Ditambahkan

### 1. **Preprocessing Comparison Visualization**
- **Fungsi**: `plot_preprocessing_comparison()`
- **Output**: Perbandingan sinyal raw vs processed untuk semua 16 channel
- **Format**: Grid 4x4 menampilkan semua channel (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6)
- **Visualisasi**: Time series plot untuk setiap channel dengan overlay raw dan processed signal

### 2. **Frequency Domain Analysis**
- **Fungsi**: `plot_frequency_comparison()`
- **Output**: Power Spectral Density (PSD) sebelum dan setelah filtering
- **Analisis**: Menunjukkan efek bandpass filter (0.5-50 Hz) dan notch filter (50 Hz)
- **Benefit**: Memvalidasi pengurangan noise pada frekuensi tinggi

### 3. **Noise Reduction Analysis**
- **Fungsi**: `plot_noise_reduction_analysis()`
- **Metrics**:
  - Signal-to-Noise Ratio (SNR) improvement
  - RMS reduction pada frekuensi noise
  - Power reduction pada frekuensi di luar band yang diinginkan
- **Visualisasi**: Before/after comparison dengan quantitative metrics

### 4. **Comprehensive Analysis**
- **Fungsi**: `analyze_preprocessing_effects()`
- **Fitur**:
  - Automatically matches raw and processed files
  - Analyzes multiple samples from each split (train/eval/dev)
  - Generates summary report dalam format Markdown
  - Creates quantitative comparison tables

## 🚀 Cara Penggunaan

### Option 1: Menggunakan run_pipeline.py
```bash
# Hanya membuat visualisasi preprocessing
python run_pipeline.py --preprocessing-viz

# Atau menggunakan step-specific command
python run_pipeline.py --step preprocessing-viz
```

### Option 2: Script demo
```bash
# Menjalankan demo dengan penjelasan lengkap
python demo_preprocessing_visualization.py
```

### Option 3: Programmatic usage
```python
from eeg_visualization import EEGVisualizer

visualizer = EEGVisualizer()
visualizer.analyze_preprocessing_effects(
    dataset_path=".",
    processed_path="processed_data",
    output_dir="preprocessing_analysis"
)
```

## 📊 Output yang Dihasilkan

### Visualisasi Files:
1. **`preprocessing_comparison_*.png`** - Perbandingan sinyal 16 channel
2. **`frequency_comparison_*.png`** - Analisis spektrum frekuensi
3. **`noise_reduction_analysis_*.png`** - Evaluasi pengurangan noise

### Report Files:
1. **`preprocessing_analysis_summary.md`** - Ringkasan lengkap analisis
2. **Quantitative metrics** - SNR improvement, RMS reduction, dll

## 🔧 Preprocessing Pipeline yang Dianalisis

1. **Bandpass Filtering**: 0.5-50 Hz untuk menghilangkan drift dan high-frequency noise
2. **Notch Filtering**: 50 Hz untuk menghilangkan power line interference
3. **Z-score Normalization**: Standardisasi amplitudo sinyal
4. **Feature Extraction**: 960 features per segment termasuk:
   - Statistical features (mean, std, skewness, kurtosis)
   - Frequency domain features (PSD in multiple bands)
   - Connectivity features (correlation between channels)
   - Wavelet features (time-frequency analysis)

## 💡 Insight yang Diperoleh

### Expected Results:
- **Reduced noise**: High-frequency artifacts berkurang drastis
- **Preserved neural signals**: Informasi penting pada 0.5-50 Hz tetap utuh
- **Improved SNR**: Signal-to-Noise Ratio meningkat signifikan
- **Better feature quality**: Features menjadi lebih discriminative untuk classification

### Validation Metrics:
- **Power reduction** pada frekuensi > 50 Hz
- **SNR improvement** across all channels
- **Baseline drift removal** pada frekuensi < 0.5 Hz
- **Line noise suppression** pada 50 Hz

## 🎯 Manfaat untuk Klasifikasi

1. **Improved Signal Quality**: Preprocessing menghasilkan sinyal yang lebih bersih
2. **Better Feature Extraction**: Features yang diekstrak lebih representatif
3. **Enhanced Separability**: Perbedaan seizure vs normal lebih jelas
4. **Robust Classification**: Model klasifikasi menjadi lebih akurat dan stabil

## 📝 Troubleshooting

### Common Issues:
1. **"Processed data not found"**: Jalankan preprocessing dulu dengan `python run_pipeline.py --step preprocessing`
2. **"Import error"**: Install dependencies dengan `pip install matplotlib numpy scipy pandas`
3. **"No matching files"**: Pastikan struktur directory sesuai (raw: `./train/`, processed: `./processed_data/train/`)

### Directory Structure Expected:
```
./
├── train/           # Raw data files
├── eval/
├── dev/
├── processed_data/  # Processed data files
│   ├── train/
│   ├── eval/
│   └── dev/
└── preprocessing_analysis/  # Generated visualizations
```

## 🎉 Kesimpulan

Fitur visualisasi preprocessing ini memberikan insight mendalam tentang efektivitas preprocessing pipeline. Dengan 16-channel comparison, frequency analysis, dan noise reduction metrics, Anda dapat:

1. **Validate preprocessing effectiveness**
2. **Identify potential issues** dalam pipeline
3. **Optimize parameters** untuk hasil yang lebih baik
4. **Document preprocessing effects** untuk paper/thesis
5. **Compare different preprocessing approaches**

Semua visualisasi dibuat dengan matplotlib dan dapat diintegrasikan ke dalam workflow machine learning untuk epileptic seizure detection.
