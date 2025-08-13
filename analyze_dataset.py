#!/usr/bin/env python3
"""
Dataset Analysis Script for TUSZ EEG Dataset
Menganalisis struktur dan konten dataset EEG untuk klasifikasi epileptic seizure
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TUSZDatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'dev', 'eval']
        self.data_info = defaultdict(list)

    def scan_dataset(self):
        """Scan seluruh dataset untuk mendapatkan informasi struktur dan metadata"""
        print("📊 Scanning dataset structure...")

        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue

            print(f"\n🔍 Analyzing {split} split...")
            patient_count = 0
            session_count = 0
            file_count = 0

            for patient_dir in split_path.iterdir():
                if not patient_dir.is_dir():
                    continue

                patient_count += 1
                patient_id = patient_dir.name

                for session_dir in patient_dir.iterdir():
                    if not session_dir.is_dir():
                        continue

                    session_count += 1
                    session_id = session_dir.name

                    tcp_ar_path = session_dir / "01_tcp_ar"
                    if tcp_ar_path.exists():
                        for file in tcp_ar_path.iterdir():
                            if file.suffix == '.npy':
                                file_count += 1
                                self.data_info[split].append({
                                    'patient_id': patient_id,
                                    'session_id': session_id,
                                    'file_path': str(file),
                                    'file_type': 'X' if '_X.npy' in file.name else 'y'
                                })

            print(f"  - Patients: {patient_count}")
            print(f"  - Sessions: {session_count}")
            print(f"  - Files: {file_count}")

    def analyze_data_samples(self, max_samples=5):
        """Analisis sampel data untuk memahami format dan dimensi"""
        print("\n🔬 Analyzing data samples...")

        sample_info = {}

        for split in self.splits:
            if split not in self.data_info:
                continue

            print(f"\n📈 {split.upper()} Split Analysis:")

            # Ambil sampel file X dan y
            x_files = [item for item in self.data_info[split] if item['file_type'] == 'X']
            y_files = [item for item in self.data_info[split] if item['file_type'] == 'y']

            if not x_files or not y_files:
                print("  No X or y files found!")
                continue

            sample_info[split] = {'X': [], 'y': []}

            # Analisis beberapa sampel
            for i, (x_file, y_file) in enumerate(zip(x_files[:max_samples], y_files[:max_samples])):
                try:
                    # Load data
                    X_data = np.load(x_file['file_path'])
                    y_data = np.load(y_file['file_path'])

                    sample_info[split]['X'].append({
                        'shape': X_data.shape,
                        'dtype': X_data.dtype,
                        'min': float(np.min(X_data)),
                        'max': float(np.max(X_data)),
                        'mean': float(np.mean(X_data)),
                        'std': float(np.std(X_data))
                    })

                    sample_info[split]['y'].append({
                        'shape': y_data.shape,
                        'dtype': y_data.dtype,
                        'unique_values': np.unique(y_data).tolist(),
                        'label_counts': dict(zip(*np.unique(y_data, return_counts=True)))
                    })

                    if i == 0:  # Print details untuk sampel pertama
                        print(f"  Sample {i+1}:")
                        print(f"    X shape: {X_data.shape}, dtype: {X_data.dtype}")
                        print(f"    X range: [{np.min(X_data):.4f}, {np.max(X_data):.4f}]")
                        print(f"    X mean±std: {np.mean(X_data):.4f}±{np.std(X_data):.4f}")
                        print(f"    y shape: {y_data.shape}, dtype: {y_data.dtype}")
                        print(f"    y labels: {np.unique(y_data)}")
                        print(f"    y distribution: {dict(zip(*np.unique(y_data, return_counts=True)))}")

                except Exception as e:
                    print(f"    Error loading sample {i+1}: {e}")

        return sample_info

    def create_summary_report(self, sample_info):
        """Buat laporan ringkasan dataset"""
        print("\n📋 Dataset Summary Report")
        print("=" * 50)

        total_patients = 0
        total_sessions = 0
        total_files = 0

        for split in self.splits:
            if split in self.data_info:
                patients = len(set([item['patient_id'] for item in self.data_info[split]]))
                sessions = len(set([f"{item['patient_id']}_{item['session_id']}" for item in self.data_info[split]]))
                files = len(self.data_info[split])

                total_patients += patients
                total_sessions += sessions
                total_files += files

                print(f"\n{split.upper()}:")
                print(f"  - Unique patients: {patients}")
                print(f"  - Total sessions: {sessions}")
                print(f"  - Total files: {files}")

                if split in sample_info and sample_info[split]['X']:
                    x_shapes = [info['shape'] for info in sample_info[split]['X']]
                    print(f"  - X data shapes: {set(x_shapes)}")

                    if sample_info[split]['y']:
                        all_labels = set()
                        for y_info in sample_info[split]['y']:
                            all_labels.update(y_info['unique_values'])
                        print(f"  - Unique labels: {sorted(all_labels)}")

        print(f"\nTOTAL DATASET:")
        print(f"  - Total patients: {total_patients}")
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Total files: {total_files}")

    def save_metadata(self):
        """Simpan metadata dataset ke file CSV"""
        print("\n💾 Saving metadata...")

        all_data = []
        for split in self.splits:
            if split in self.data_info:
                for item in self.data_info[split]:
                    all_data.append({
                        'split': split,
                        'patient_id': item['patient_id'],
                        'session_id': item['session_id'],
                        'file_path': item['file_path'],
                        'file_type': item['file_type']
                    })

        df = pd.DataFrame(all_data)
        output_path = self.dataset_path / 'dataset_metadata.csv'
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to: {output_path}")

        return df

def main():
    dataset_path = "/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS"

    print("🧠 TUSZ EEG Dataset Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = TUSZDatasetAnalyzer(dataset_path)

    # Scan dataset
    analyzer.scan_dataset()

    # Analyze samples
    sample_info = analyzer.analyze_data_samples(max_samples=3)

    # Create summary
    analyzer.create_summary_report(sample_info)

    # Save metadata
    metadata_df = analyzer.save_metadata()

    print("\n✅ Analysis complete!")
    print(f"📁 Check the dataset directory for metadata.csv")

if __name__ == "__main__":
    main()
