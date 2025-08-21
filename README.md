# EEG Seizure Detection Pipeline

Comprehensive pipeline untuk pre-processing dan klasifikasi data EEG untuk deteksi epileptic seizure menggunakan dataset TUSZ yang telah dimodifikasi.

## 📋 Deskripsi

Pipeline ini menyediakan solusi end-to-end untuk:
- Analisis dataset EEG TUSZ
- Pre-processing data EEG (denoising dan ekstraksi fitur)
- Klasifikasi seizure menggunakan multiple machine learning algorithms
- Visualisasi dan analisis hasil

## 🗂️ Struktur Dataset

```
EEG_NEW_16CHS/
├── train/          # Data training
├── dev/            # Data development/validation
├── eval/           # Data evaluasi/testing
└── processed/      # Data hasil preprocessing (akan dibuat)
```

Setiap file data memiliki format:
- `*_X.npy`: Data EEG dengan shape (36, 16, 1280)
  - 36 segments per file
  - 16 channels EEG
  - 1280 time points per segment (5.12 detik @ 250Hz)
- `*_y.npy`: Labels dengan shape (36,)
  - 0: Normal/non-seizure
  - 1: Seizure

## 🚀 Quick Start

### 1. Instalasi Dependencies

```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# atau .venv\Scripts\activate  # Windows

# Install packages
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
```

### 2. Jalankan Pipeline Lengkap

```bash
# Jalankan seluruh pipeline
python run_pipeline.py

# Atau jalankan step tertentu
python run_pipeline.py --step analysis
python run_pipeline.py --step preprocessing
python run_pipeline.py --step classification
python run_pipeline.py --step visualization
```

### 3. Jalankan Script Individual

```bash
# Analisis dataset
python analyze_dataset.py

# Preprocessing
python eeg_preprocessing.py

# Klasifikasi
python eeg_classification.py

# Visualisasi
python eeg_visualization.py
```

## 📁 File Scripts

### 1. `analyze_dataset.py`
Menganalisis struktur dan karakteristik dataset:
- Scan seluruh dataset untuk metadata
- Analisis dimensi dan distribusi data
- Generate summary report

### 2. `eeg_preprocessing.py`
Pipeline preprocessing comprehensive:
- **Filtering**: Bandpass (0.5-50Hz) dan Notch (50Hz) filters
- **Denoising**: Optional ICA untuk artifact removal
- **Feature Extraction**:
  - Statistical features (mean, std, skewness, kurtosis, dll.)
  - Frequency domain features (power spectral density per band)
  - Connectivity features (cross-correlation antar channel)
  - Wavelet-based features (multi-scale analysis)
- **Normalization**: RobustScaler untuk mengatasi outliers

### 3. `eeg_classification.py`
Sistem klasifikasi multi-model:
- **Models**: Random Forest, SVM, Logistic Regression, MLP, dll.
- **Feature Selection**: SelectKBest, RFE, atau PCA
- **Evaluation**: Cross-validation dan testing pada multiple splits
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### 4. `eeg_visualization.py`
Comprehensive visualization system:
- **Raw EEG Plotting**: Time series visualization untuk multiple channels
- **Frequency Analysis**: Power spectral density plots
- **Spectrograms**: Time-frequency analysis
- **Feature Visualization**: Feature importance dan distribution plots
- **Classification Results**: Confusion matrix, ROC curves
- **Preprocessing Comparison**: Raw vs processed signal comparison

### 5. `comprehensive_preprocessing_analysis.py` 🆕
Advanced analysis untuk seluruh dataset:
- **Global Statistics**: Aggregate analysis untuk semua segmen
- **Frequency Domain**: Band power analysis across entire dataset
- **Channel-wise Metrics**: SNR, noise reduction per channel
- **Sample Visualization**: Representative seizure vs normal samples
- **Quantitative Reports**: Comprehensive metrics dan improvements

### 6. `run_pipeline.py`
Script orchestrator utama dengan berbagai options:
- Full pipeline execution
- Individual step execution
- Skip options untuk debugging
- Preprocessing visualization
- **Comprehensive analysis** 🆕

## � Advanced Analysis Options 🆕

### Preprocessing Visualization
Untuk melihat efek preprocessing pada sample data:

```bash
# Simple preprocessing comparison
python run_pipeline.py --preprocessing-viz

# Atau menggunakan step
python run_pipeline.py --step preprocessing-viz
```

### Comprehensive Dataset Analysis 🌟
Untuk analisis mendalam seluruh dataset:

```bash
# Comprehensive analysis untuk seluruh dataset
python run_pipeline.py --comprehensive-analysis

# Atau menggunakan step
python run_pipeline.py --step comprehensive-analysis

# Atau jalankan demo dengan penjelasan
python demo_comprehensive_analysis.py
```

#### Output Comprehensive Analysis:
1. **Statistical Analysis**: Global statistics raw vs processed
2. **Frequency Analysis**: Band power comparison across all channels
3. **Channel-wise Metrics**: SNR improvement, noise reduction per channel
4. **Sample Comparisons**: Representative seizure vs normal time series
5. **Noise Reduction Analysis**: Quantitative artifact removal metrics
6. **Condition-specific Effects**: Preprocessing impact on seizure vs normal
7. **Comprehensive Report**: Detailed markdown report with all metrics

## �🔧 Konfigurasi

### Preprocessing Options
Dalam `eeg_preprocessing.py`, Anda dapat mengatur:

```python
create_preprocessing_pipeline(
    dataset_path=dataset_path,
    output_path=output_path,
    apply_filters=True,      # Bandpass + Notch filtering
    apply_ica=False,         # ICA artifact removal (computationally expensive)
    extract_features=True,   # Feature extraction
    normalize_features=True  # Feature normalization
)
```

### Classification Options
Dalam `eeg_classification.py`, Anda dapat mengatur:

```python
classifier.run_complete_pipeline(
    use_normalized=True,        # Gunakan features yang dinormalisasi
    feature_selection='selectk', # 'selectk', 'rfe', atau 'pca'
    n_features=500              # Jumlah fitur yang dipilih
)
```

## � Command Line Options

### run_pipeline.py Options

```bash
# Basic usage
python run_pipeline.py                    # Run full pipeline

# Step-specific execution
python run_pipeline.py --step analysis                    # Dataset analysis only
python run_pipeline.py --step preprocessing               # Preprocessing only
python run_pipeline.py --step classification              # Classification only
python run_pipeline.py --step visualization               # Visualization only
python run_pipeline.py --step preprocessing-viz           # Preprocessing comparison
python run_pipeline.py --step comprehensive-analysis      # Full dataset analysis

# Skip options
python run_pipeline.py --skip-analysis          # Skip dataset analysis
python run_pipeline.py --skip-preprocessing     # Skip preprocessing
python run_pipeline.py --skip-classification    # Skip classification
python run_pipeline.py --skip-visualization     # Skip visualization

# Special analysis options
python run_pipeline.py --preprocessing-viz         # Quick preprocessing comparison
python run_pipeline.py --comprehensive-analysis    # Full dataset analysis
```

### Demo Scripts

```bash
# Interactive demos with explanations
python demo_preprocessing_visualization.py      # Basic preprocessing demo
python demo_comprehensive_analysis.py           # Advanced analysis demo
```

## �📊 Output Files

### Data Files
- `dataset_metadata.csv`: Metadata lengkap dataset
- `processed/train_features.npy`: Training features
- `processed/train_labels.npy`: Training labels
- `processed/train_features_normalized.npy`: Normalized features
- `processed/feature_scaler.pkl`: Trained scaler untuk normalisasi

### Results
- `classification_results.csv`: Perbandingan performa model
- `preprocessing_summary.csv`: Summary hasil preprocessing
- `models/`: Direktori berisi trained models (.pkl files)

### Visualizations
- `model_comparison_*.png`: Perbandingan metrik antar model
- `confusion_matrices_*.png`: Confusion matrices untuk setiap model
- `raw_eeg_sample_*.png`: Contoh sinyal EEG mentah
- `frequency_spectrum_*.png`: Analisis spektrum frekuensi
- `seizure_vs_normal_comparison.png`: Perbandingan pola seizure vs normal

### Preprocessing Analysis 🆕
- `preprocessing_analysis/`: Basic preprocessing comparisons
- `comprehensive_preprocessing_analysis/`: Advanced dataset-wide analysis
  - `statistical_comparison_aggregate.png`: Global statistical comparison
  - `frequency_comparison_aggregate.png`: Frequency band analysis
  - `channel_wise_comparison.png`: Per-channel metrics (SNR, correlation)
  - `timeseries_comparison_*.png`: Sample time series comparisons
  - `noise_reduction_analysis_aggregate.png`: Noise reduction metrics
  - `seizure_vs_normal_preprocessing.png`: Condition-specific analysis
  - `comprehensive_preprocessing_report.md`: Detailed quantitative report

## 🧠 Feature Engineering

Pipeline ini mengekstrak berbagai jenis fitur dari sinyal EEG:

### 1. Statistical Features (per channel)
- Mean, Standard deviation, Variance
- Skewness, Kurtosis
- Min, Max, Peak-to-peak, Median, Quartiles
- Total energy, Mean power, RMS
- Zero crossing rate

### 2. Frequency Domain Features (per channel per band)
- Power dalam frequency bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-50Hz)
- Relative power (power relatif terhadap total power)
- Spectral centroid, Peak frequency

### 3. Connectivity Features
- Pearson correlation antar pasangan channel
- Cross-correlation dengan delay

### 4. Wavelet Features (simplified)
- Multi-scale analysis menggunakan filtering
- Statistical features pada berbagai skala temporal

## 📈 Model Performance

Pipeline ini mengevaluasi multiple algoritma:

1. **Random Forest**: Ensemble method yang robust
2. **Gradient Boosting**: Boosting algorithm untuk high performance
3. **SVM**: Support Vector Machine dengan RBF kernel
4. **Logistic Regression**: Linear model sebagai baseline
5. **MLP**: Multi-layer Perceptron (neural network)
6. **Naive Bayes**: Probabilistic classifier
7. **KNN**: K-Nearest Neighbors

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision dan recall
- **AUC-ROC**: Area Under ROC Curve

## 🔬 Pipeline Details

### 1. Dataset Analysis
- Scan struktur direktori dan file
- Analisis dimensi data (segments, channels, time points)
- Distribusi label (seizure vs non-seizure)
- Generate metadata untuk tracking

### 2. Preprocessing
- **Bandpass Filtering**: Menghilangkan noise di luar range 0.5-50Hz
- **Notch Filtering**: Menghilangkan interference dari power line (50Hz)
- **Optional ICA**: Independent Component Analysis untuk artifact removal
- **Feature Extraction**: Multiple domain features untuk representasi yang kaya
- **Normalization**: RobustScaler untuk handling outliers

### 3. Classification
- **Feature Selection**: Mengurangi dimensi fitur untuk efisiensi
- **Model Training**: Train multiple models dengan cross-validation
- **Evaluation**: Test pada development dan evaluation sets
- **Model Persistence**: Save trained models untuk future use

### 4. Visualization & Analysis
- **Signal Visualization**: Plot time series, spectrum, spectrogram
- **Comparison Analysis**: Seizure vs normal patterns
- **Performance Visualization**: Model comparison charts
- **Feature Analysis**: Feature importance plots

## 🚨 Notes & Tips

1. **Memory Usage**: Dataset cukup besar, pastikan RAM mencukupi (minimal 8GB recommended)

2. **Processing Time**:
   - Preprocessing: ~10-30 menit tergantung dataset size
   - Classification: ~5-15 menit tergantung number of features
   - ICA (jika diaktifkan): Bisa menambah waktu signifikan

3. **Feature Selection**:
   - Gunakan feature selection untuk mengurangi overfitting
   - SelectKBest biasanya paling cepat
   - RFE memberikan hasil yang baik tapi lebih lambat

4. **Model Selection**:
   - Random Forest biasanya memberikan baseline yang baik
   - SVM dengan RBF kernel sering memberikan performa terbaik
   - Coba multiple models untuk comparison

5. **Hyperparameter Tuning**:
   - Scripts menggunakan default parameters yang reasonable
   - Untuk hasil optimal, lakukan grid search pada model terbaik

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error**:
   ```bash
   # Reduce batch size atau gunakan feature selection dengan n_features yang lebih kecil
   ```

2. **Import Error**:
   ```bash
   # Pastikan semua packages terinstall
   pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
   ```

3. **File Not Found**:
   ```bash
   # Pastikan struktur direktori sesuai dan file .npy ada
   python analyze_dataset.py  # untuk check dataset
   ```

4. **Performance Issues**:
   ```bash
   # Disable ICA jika terlalu lambat
   # Gunakan feature selection untuk mengurangi dimensi
   # Reduce number of models yang di-train
   ```

## 📚 References

- **TUSZ Dataset**: Temple University Hospital Seizure Database
- **EEG Signal Processing**: Niedermeyer's Electroencephalography
- **Machine Learning**: scikit-learn documentation
- **Signal Processing**: SciPy signal processing library

## 📄 License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Submit pull request

---

**Happy EEG Analysis! 🧠⚡**
