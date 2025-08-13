"""
Preprocessing Visualization Demo

Script ini mendemonstrasikan visualisasi perbandingan data sebelum dan setelah preprocessing
untuk dataset EEG TUSZ yang telah dimodifikasi.

Fitur yang tersedia:
1. plot_preprocessing_comparison() - Membandingkan sinyal raw vs processed
2. plot_frequency_comparison() - Membandingkan spektrum frekuensi
3. plot_noise_reduction_analysis() - Analisis pengurangan noise
4. analyze_preprocessing_effects() - Analisis komprehensif efek preprocessing

Usage:
    from eeg_visualization import EEGVisualizer

    visualizer = EEGVisualizer()

    # Membuat visualisasi perbandingan preprocessing
    visualizer.analyze_preprocessing_effects(
        dataset_path=".",
        processed_path="processed_data",
        output_dir="visualization_output"
    )

Output yang dihasilkan:
- preprocessing_comparison_*.png - Perbandingan sinyal raw vs processed
- frequency_comparison_*.png - Perbandingan spektrum frekuensi
- noise_reduction_analysis_*.png - Analisis pengurangan noise
- preprocessing_analysis_summary.md - Ringkasan analisis

Penjelasan Visualisasi:

1. Preprocessing Comparison:
   - Menampilkan 16 channel EEG sebelum dan setelah filtering
   - Bandpass filter: 0.5-50 Hz
   - Notch filter: 50 Hz (untuk menghilangkan line noise)
   - Normalisasi z-score

2. Frequency Analysis:
   - Power Spectral Density (PSD) sebelum dan setelah filtering
   - Menunjukkan pengurangan noise pada frekuensi tinggi
   - Preservasi sinyal pada frekuensi yang relevan (0.5-50 Hz)

3. Noise Reduction:
   - SNR (Signal-to-Noise Ratio) improvement
   - RMS reduction pada frekuensi noise
   - Preservation of neural signals

Preprocessing Pipeline yang dianalisis:
1. Bandpass filtering (0.5-50 Hz)
2. Notch filtering (50 Hz)
3. Z-score normalization
4. Feature extraction (960 features per segment):
   - Statistical features (mean, std, skewness, kurtosis)
   - Frequency domain features (PSD in multiple bands)
   - Connectivity features (correlation between channels)
   - Wavelet features (time-frequency analysis)

Expected Results:
- Reduced high-frequency noise
- Improved signal clarity
- Better feature separability for classification
- Enhanced seizure vs normal distinction

Troubleshooting:
- Pastikan processed_data directory tersedia
- Jalankan preprocessing terlebih dahulu: python run_pipeline.py --preprocess-only
- Pastikan dependencies terinstall: numpy, matplotlib, scipy, pandas
"""

def demo_usage():
    """
    Demonstrasi penggunaan visualisasi preprocessing
    """

    print("📊 EEG Preprocessing Visualization Demo")
    print("=" * 50)

    print("\n🔧 Setup:")
    print("from eeg_visualization import EEGVisualizer")
    print("visualizer = EEGVisualizer()")

    print("\n📈 Create Preprocessing Comparison:")
    print("visualizer.analyze_preprocessing_effects(")
    print("    dataset_path='.',")
    print("    processed_path='processed_data',")
    print("    output_dir='visualization_output'")
    print(")")

    print("\n📋 Expected Output Files:")
    output_files = [
        "preprocessing_comparison_train_sample_1.png",
        "preprocessing_comparison_eval_sample_1.png",
        "frequency_comparison_train_sample_1.png",
        "noise_reduction_analysis_train_sample_1.png",
        "preprocessing_analysis_summary.md"
    ]

    for file in output_files:
        print(f"  📄 {file}")

    print("\n💡 Tips:")
    print("- Data harus sudah dipreprocess terlebih dahulu")
    print("- Gunakan matplotlib backend 'Agg' untuk server tanpa display")
    print("- Script akan mencocokkan file raw dan processed berdasarkan nama")
    print("- Analisis dilakukan pada beberapa sample dari setiap split")

if __name__ == "__main__":
    demo_usage()
