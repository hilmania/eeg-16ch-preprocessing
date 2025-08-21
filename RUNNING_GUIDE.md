# Panduan Menjalankan EEG Preprocessing Pipeline

## 🧠 EEG Seizure Detection Pipeline
Pipeline lengkap untuk preprocessing dan klasifikasi data EEG untuk deteksi kejang epilepsi.

---

## 📋 Daftar Isi
1. [Channel Selection Results](#channel-selection-results)
2. [Step-by-Step Running Guide](#step-by-step-running-guide)
3. [Channel Selection Methods](#channel-selection-methods)
4. [Advanced Options](#advanced-options)

---

## 📁 Channel Selection Results

Hasil channel selection **DISIMPAN** ke dalam file berikut:

### 1. Wrapper Method
```bash
processed/channel_selection_wrapper.json
```

**Format file:**
```json
{
  "indices": [2, 8, 13],           // Indeks channel yang dipilih
  "names": ["F3", "O1", "T4"],    // Nama channel yang dipilih
  "params": {                     // Parameter yang digunakan
    "max_k": 8,
    "model": "logreg",
    "scoring": "roc_auc",
    "cv": 3,
    "max_files": 2,
    "max_segments_per_file": 50
  }
}
```

### 2. Variance Top-K Method
Channel yang dipilih akan ditampilkan di log output dan disimpan dalam summary.

### 3. By Name Method
Channel yang dipilih akan ditampilkan di log output.

---

## 🚀 Step-by-Step Running Guide

### **STEP 1: Dataset Analysis**
Menganalisis struktur dataset dan membuat metadata.

```bash
# Jalankan analysis saja
python run_pipeline.py --step analysis

# Atau manual
python analyze_dataset.py
```

**Output:**
- `dataset_metadata.csv` - metadata lengkap dataset
- Log informasi tentang jumlah file per split

---

### **STEP 2: Data Preprocessing**
Preprocessing data EEG dengan berbagai opsi channel selection.

#### A. Preprocessing Standar (Tanpa Channel Selection)
```bash
python run_pipeline.py --step preprocessing
```

#### B. Preprocessing dengan Channel Selection by Name
```bash
# Keep specific channels
python run_pipeline.py --step preprocessing \
  --channel-selection-method by_name \
  --channels-keep "F3,F4,C3,C4,O1,O2"

# Drop specific channels
python run_pipeline.py --step preprocessing \
  --channel-selection-method by_name \
  --channels-drop "T3,T4,T5,T6"
```

#### C. Preprocessing dengan Variance Top-K Selection
```bash
# Select top 8 channels by variance
python run_pipeline.py --step preprocessing \
  --channel-selection-method variance_topk \
  --channel-topk 8 \
  --channel-selection-metric variance
```

#### D. Preprocessing dengan Wrapper Selection
```bash
# Basic wrapper selection
python run_pipeline.py --step preprocessing \
  --channel-selection-method wrapper

# Advanced wrapper selection
python run_pipeline.py --step preprocessing \
  --channel-selection-method wrapper \
  --wrapper-max-k 5 \
  --wrapper-model rf \
  --wrapper-scoring roc_auc \
  --wrapper-cv 5 \
  --wrapper-max-files 20 \
  --wrapper-max-segments-per-file 100
```

**Output:**
- `processed/train_features.npy` - training features
- `processed/dev_features.npy` - development features
- `processed/eval_features.npy` - evaluation features
- `processed/*_features_normalized.npy` - normalized features
- `processed/feature_scaler.pkl` - scaler object
- `processed/channel_selection_wrapper.json` - wrapper results (jika menggunakan wrapper)
- `processed/preprocessing_summary.csv` - summary preprocessing

---

### **STEP 3: Model Training and Classification**
Melatih model untuk klasifikasi seizure detection.

```bash
# Jalankan classification saja (setelah preprocessing)
python run_pipeline.py --step classification

# Skip preprocessing jika sudah ada
python run_pipeline.py --step classification --skip-preprocessing
```

**Output:**
- Model yang sudah dilatih
- Hasil evaluasi model
- Confusion matrix dan metrics

---

### **STEP 4: Visualization and Analysis**
Membuat visualisasi hasil preprocessing dan analysis.

```bash
# Jalankan visualization saja
python run_pipeline.py --step visualization

# Preprocessing comparison visualization
python run_pipeline.py --preprocessing-viz

# Comprehensive analysis (dataset-wide)
python run_pipeline.py --comprehensive-analysis
```

**Output:**
- Plot perbandingan raw vs processed
- Analisis frekuensi dan statistik
- Visualisasi per pasien

---

### **STEP 5: Full Pipeline**
Menjalankan semua step secara berurutan.

```bash
# Full pipeline standar
python run_pipeline.py

# Full pipeline dengan wrapper selection
python run_pipeline.py \
  --channel-selection-method wrapper \
  --wrapper-max-k 6 \
  --wrapper-model logreg

# Skip step tertentu
python run_pipeline.py \
  --skip-analysis \
  --channel-selection-method variance_topk \
  --channel-topk 10
```

---

## 🔧 Channel Selection Methods

### 1. **None** (Default)
Tidak ada channel selection, menggunakan semua 16 channel.
```bash
--channel-selection-method none
```

### 2. **By Name**
Memilih channel berdasarkan nama channel.

**Keep specific channels:**
```bash
--channel-selection-method by_name \
--channels-keep "Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2"
```

**Drop specific channels:**
```bash
--channel-selection-method by_name \
--channels-drop "F7,F8,T3,T4,T5,T6"
```

### 3. **Variance Top-K**
Memilih K channel dengan variance tertinggi.
```bash
--channel-selection-method variance_topk \
--channel-topk 8 \
--channel-selection-metric variance
```

### 4. **Wrapper Selection** ⭐
Memilih channel terbaik menggunakan machine learning (greedy forward selection).

```bash
--channel-selection-method wrapper \
--wrapper-max-k 8 \                      # Max channels to select
--wrapper-model logreg \                  # Model: logreg atau rf
--wrapper-scoring roc_auc \               # Scoring: roc_auc, accuracy, f1
--wrapper-cv 3 \                         # Cross-validation folds
--wrapper-max-files 10 \                 # Max training files to sample
--wrapper-max-segments-per-file 200      # Max segments per file
```

**Hasil wrapper disimpan di:**
- `processed/channel_selection_wrapper.json`

---

## ⚙️ Advanced Options

### Channel Selection pada Raw Data
Simpan processed raw data hanya dengan channel terpilih:
```bash
--apply-channel-selection-to-raw
```

### Skip Steps
```bash
--skip-analysis          # Skip dataset analysis
--skip-preprocessing     # Skip preprocessing
--skip-classification    # Skip model training
--skip-visualization     # Skip visualization
```

### Standalone Steps
```bash
--step analysis                    # Hanya analysis
--step preprocessing              # Hanya preprocessing
--step classification            # Hanya classification
--step visualization            # Hanya visualization
--preprocessing-viz            # Hanya preprocessing visualization
--comprehensive-analysis      # Hanya comprehensive analysis
```

---

## 📊 Output Files

### Preprocessing Results
```
processed/
├── train_features.npy              # Training features
├── dev_features.npy                # Development features
├── eval_features.npy               # Evaluation features
├── train_features_normalized.npy   # Normalized training features
├── dev_features_normalized.npy     # Normalized dev features
├── eval_features_normalized.npy    # Normalized eval features
├── feature_scaler.pkl              # Feature scaler object
├── channel_selection_wrapper.json  # Wrapper selection results
├── preprocessing_summary.csv       # Preprocessing summary
└── {split}_processed/              # Processed raw data per split
    ├── patient_file_processed.npy
    └── ...
```

### Analysis Results
```
comprehensive_preprocessing_analysis/
├── dataset_overview.png
├── frequency_analysis.png
├── channel_comparison.png
├── time_series_samples.png
└── comprehensive_report.md
```

---

## 🎯 Example Commands

### 1. Quick Start (Wrapper Selection)
```bash
python run_pipeline.py \
  --channel-selection-method wrapper \
  --wrapper-max-k 5 \
  --wrapper-model logreg
```

### 2. Production Run (Best Performance)
```bash
python run_pipeline.py \
  --channel-selection-method wrapper \
  --wrapper-max-k 8 \
  --wrapper-model rf \
  --wrapper-scoring roc_auc \
  --wrapper-cv 5 \
  --wrapper-max-files 50 \
  --wrapper-max-segments-per-file 200 \
  --apply-channel-selection-to-raw
```

### 3. Preprocessing Only (Specific Channels)
```bash
python run_pipeline.py --step preprocessing \
  --channel-selection-method by_name \
  --channels-keep "F3,F4,C3,C4,P3,P4,O1,O2" \
  --apply-channel-selection-to-raw
```

### 4. Classification Only (After Preprocessing)
```bash
python run_pipeline.py --step classification --skip-preprocessing
```

---

## ❓ FAQ

**Q: Di mana hasil channel selection disimpan?**
A: Untuk wrapper method di `processed/channel_selection_wrapper.json`, untuk method lain di log output dan preprocessing summary.

**Q: Bagaimana cara melihat channel yang terpilih?**
A: Check file JSON untuk wrapper, atau lihat log output saat preprocessing.

**Q: Bisa menggunakan custom channel names?**
A: Ya, edit `default_channel_names` di `eeg_preprocessing.py` atau pass via parameter.

**Q: Wrapper method lambat?**
A: Reduce `--wrapper-max-files` dan `--wrapper-max-segments-per-file` untuk testing.

---

## 🔗 Files Reference

- **Main Script**: `run_pipeline.py`
- **Preprocessing**: `eeg_preprocessing.py`
- **Classification**: `eeg_classification.py`
- **Visualization**: `eeg_visualization.py`
- **Analysis**: `comprehensive_preprocessing_analysis.py`

---

**💡 Tip:** Gunakan wrapper method untuk hasil terbaik, atau variance_topk untuk speed vs performance balance.
