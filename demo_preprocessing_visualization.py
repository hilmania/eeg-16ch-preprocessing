#!/usr/bin/env python3
"""
Demo Script: Visualisasi Perbandingan Preprocessing EEG

Script ini mendemonstrasikan cara menggunakan fitur visualisasi untuk membandingkan
data EEG sebelum dan setelah preprocessing.

Fungsi Utama:
1. Membandingkan sinyal raw vs processed pada 16 channel
2. Analisis spektrum frekuensi sebelum dan setelah filtering
3. Evaluasi pengurangan noise
4. Ringkasan kuantitatif efek preprocessing

Usage:
    python demo_preprocessing_visualization.py

Atau menggunakan run_pipeline.py:
    python run_pipeline.py --preprocessing-viz
    python run_pipeline.py --step preprocessing-viz
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    print("🎨 Demo: Visualisasi Perbandingan Preprocessing EEG")
    print("=" * 60)

    # Check if processed data exists
    processed_path = Path('processed_data')
    if not processed_path.exists():
        print("❌ Directory processed_data tidak ditemukan!")
        print("💡 Jalankan preprocessing terlebih dahulu:")
        print("   python run_pipeline.py --step preprocessing")
        return False

    # Check available data
    splits = ['train', 'eval', 'dev']
    available_data = {}

    print("\n📊 Checking available data...")
    for split in splits:
        split_path = processed_path / split
        if split_path.exists():
            files = list(split_path.glob('*.npy'))
            available_data[split] = len(files)
            print(f"   {split}: {len(files)} files")
        else:
            available_data[split] = 0
            print(f"   {split}: 0 files (directory not found)")

    total_files = sum(available_data.values())
    if total_files == 0:
        print("❌ Tidak ada file processed data ditemukan!")
        return False

    print(f"✅ Total {total_files} processed files tersedia")

    # Import visualization module
    try:
        from eeg_visualization import EEGVisualizer
        print("✅ EEG Visualization module loaded")
    except ImportError as e:
        print(f"❌ Error importing visualization module: {e}")
        return False

    # Create visualizer
    visualizer = EEGVisualizer()
    output_dir = Path('demo_preprocessing_analysis')
    output_dir.mkdir(exist_ok=True)

    print(f"\n📈 Creating preprocessing visualizations...")
    print(f"   Output directory: {output_dir.absolute()}")

    try:
        # Run preprocessing analysis
        visualizer.analyze_preprocessing_effects(
            dataset_path=".",
            processed_path=str(processed_path),
            output_dir=str(output_dir)
        )

        print("\n✅ Visualisasi berhasil dibuat!")

        # List generated files
        output_files = list(output_dir.glob('*'))
        if output_files:
            print(f"\n📁 Generated files ({len(output_files)}):")
            for file in sorted(output_files):
                print(f"   📄 {file.name}")

        print(f"\n💡 Tips:")
        print(f"   - Buka file PNG untuk melihat visualisasi")
        print(f"   - Baca file .md untuk ringkasan analisis")
        print(f"   - Preprocessing comparison menunjukkan efek filtering")
        print(f"   - Frequency analysis menunjukkan pengurangan noise")

        return True

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Buka file PNG yang dihasilkan untuk melihat visualisasi")
        print("   2. Jalankan full pipeline: python run_pipeline.py")
        print("   3. Eksperimen dengan parameter preprocessing")
    else:
        print("\n❌ Demo failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Pastikan data sudah dipreprocess")
        print("   2. Check dependencies: matplotlib, numpy, scipy")
        print("   3. Jalankan: python run_pipeline.py --step preprocessing")

    sys.exit(0 if success else 1)
