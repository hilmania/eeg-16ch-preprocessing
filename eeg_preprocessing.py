#!/usr/bin/env python3
"""
EEG Pre-processing Script for TUSZ Dataset
Comprehensive preprocessing pipeline including denoising and feature extraction
for epileptic seizure classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Signal processing
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

# Machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing class for TUSZ dataset
    """

    def __init__(self,
                 sampling_rate: int = 250,
                 channels: int = 16,
                 segment_length: int = 1280):
        """
        Initialize preprocessor

        Args:
            sampling_rate: EEG sampling rate in Hz
            channels: Number of EEG channels
            segment_length: Length of each segment in samples
        """
        self.fs = sampling_rate
        self.channels = channels
        self.segment_length = segment_length
        self.duration = segment_length / sampling_rate  # Duration in seconds

        # Frequency bands for feature extraction
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        print(f"🧠 EEG Preprocessor initialized:")
        print(f"   - Sampling rate: {self.fs} Hz")
        print(f"   - Channels: {self.channels}")
        print(f"   - Segment length: {self.segment_length} samples ({self.duration:.1f}s)")

    def apply_bandpass_filter(self,
                            data: np.ndarray,
                            low_freq: float = 0.5,
                            high_freq: float = 50.0,
                            filter_order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to remove noise outside frequency range of interest

        Args:
            data: EEG data (channels, time_points)
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            filter_order: Filter order

        Returns:
            Filtered EEG data
        """
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = signal.butter(filter_order, [low, high], btype='band')

        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = signal.filtfilt(b, a, data[ch])

        return filtered_data

    def apply_notch_filter(self,
                          data: np.ndarray,
                          notch_freq: float = 50.0,
                          quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference

        Args:
            data: EEG data (channels, time_points)
            notch_freq: Frequency to remove (50Hz for EU, 60Hz for US)
            quality_factor: Quality factor of the notch filter

        Returns:
            Filtered EEG data
        """
        b, a = signal.iirnotch(notch_freq, quality_factor, self.fs)

        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = signal.filtfilt(b, a, data[ch])

        return filtered_data

    def remove_artifacts_ica(self,
                           data: np.ndarray,
                           n_components: Optional[int] = None) -> np.ndarray:
        """
        Remove artifacts using Independent Component Analysis

        Args:
            data: EEG data (segments, channels, time_points)
            n_components: Number of ICA components

        Returns:
            Cleaned EEG data
        """
        if n_components is None:
            n_components = min(self.channels, data.shape[0])

        # Reshape data for ICA (time_points, channels)
        original_shape = data.shape
        reshaped_data = data.reshape(-1, self.channels)

        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42)
        components = ica.fit_transform(reshaped_data)

        # Remove components with high variance (likely artifacts)
        component_vars = np.var(components, axis=0)
        threshold = np.percentile(component_vars, 90)  # Remove top 10% high variance components

        # Reconstruct without high-variance components
        for i, var in enumerate(component_vars):
            if var > threshold:
                components[:, i] = 0

        cleaned_data = ica.inverse_transform(components)

        return cleaned_data.reshape(original_shape)

    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from EEG data

        Args:
            data: EEG data (channels, time_points)

        Returns:
            Statistical features array
        """
        features = []

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            # Basic statistics
            features.extend([
                np.mean(channel_data),          # Mean
                np.std(channel_data),           # Standard deviation
                np.var(channel_data),           # Variance
                skew(channel_data),             # Skewness
                kurtosis(channel_data),         # Kurtosis
                np.min(channel_data),           # Minimum
                np.max(channel_data),           # Maximum
                np.ptp(channel_data),           # Peak-to-peak
                np.median(channel_data),        # Median
                np.percentile(channel_data, 25), # Q1
                np.percentile(channel_data, 75), # Q3
            ])

            # Energy features
            features.extend([
                np.sum(channel_data**2),        # Total energy
                np.mean(channel_data**2),       # Mean power
                np.sqrt(np.mean(channel_data**2)), # RMS
            ])

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(channel_data)))
            features.append(zero_crossings / len(channel_data))

        return np.array(features)

    def extract_frequency_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract frequency domain features

        Args:
            data: EEG data (channels, time_points)

        Returns:
            Frequency features array
        """
        features = []

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            # Compute FFT
            fft_vals = fft(channel_data)
            freqs = fftfreq(len(channel_data), 1/self.fs)
            psd = np.abs(fft_vals)**2

            # Keep only positive frequencies
            positive_freqs = freqs > 0
            freqs = freqs[positive_freqs]
            psd = psd[positive_freqs]

            # Extract power in different frequency bands
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(psd[band_mask])
                features.append(band_power)

                # Relative power
                total_power = np.sum(psd)
                relative_power = band_power / total_power if total_power > 0 else 0
                features.append(relative_power)

            # Spectral features
            features.extend([
                np.mean(psd),                   # Mean power
                np.std(psd),                    # Power variability
                np.argmax(psd),                 # Peak frequency index
                freqs[np.argmax(psd)],          # Peak frequency
                np.sum(psd * freqs) / np.sum(psd), # Spectral centroid
            ])

        return np.array(features)

    def extract_connectivity_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract connectivity features between channels

        Args:
            data: EEG data (channels, time_points)

        Returns:
            Connectivity features array
        """
        features = []

        # Cross-correlation between channel pairs
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                # Pearson correlation
                corr = np.corrcoef(data[i], data[j])[0, 1]
                features.append(corr)

                # Cross-correlation with delay
                cross_corr = signal.correlate(data[i], data[j], mode='full')
                max_corr = np.max(np.abs(cross_corr))
                features.append(max_corr)

        return np.array(features)

    def extract_wavelet_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract wavelet-based features (simplified version)

        Args:
            data: EEG data (channels, time_points)

        Returns:
            Wavelet features array
        """
        features = []

        # Simple approximation using filtering for different scales
        scales = [2, 4, 8, 16, 32]  # Different time scales

        for ch in range(data.shape[0]):
            channel_data = data[ch]

            for scale in scales:
                # Simple moving average as approximation to wavelet
                if scale < len(channel_data):
                    filtered = signal.savgol_filter(channel_data,
                                                  min(scale*2+1, len(channel_data)//4),
                                                  3)
                    features.extend([
                        np.mean(filtered),
                        np.std(filtered),
                        np.max(np.abs(filtered))
                    ])
                else:
                    features.extend([0, 0, 0])

        return np.array(features)

    def extract_comprehensive_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract all types of features

        Args:
            data: EEG data (channels, time_points)

        Returns:
            Combined feature vector
        """
        # Extract different types of features
        stat_features = self.extract_statistical_features(data)
        freq_features = self.extract_frequency_features(data)
        conn_features = self.extract_connectivity_features(data)
        wavelet_features = self.extract_wavelet_features(data)

        # Combine all features
        all_features = np.concatenate([
            stat_features,
            freq_features,
            conn_features,
            wavelet_features
        ])

        return all_features

    def preprocess_single_file(self,
                             X_data: np.ndarray,
                             apply_filters: bool = True,
                             apply_ica: bool = False,
                             extract_features: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a single EEG file

        Args:
            X_data: EEG data (segments, channels, time_points)
            apply_filters: Whether to apply filtering
            apply_ica: Whether to apply ICA for artifact removal
            extract_features: Whether to extract features

        Returns:
            Tuple of (processed_data, features_if_extracted)
        """
        processed_data = X_data.copy()
        all_features = None

        if apply_filters or apply_ica or extract_features:
            print(f"  Processing {processed_data.shape[0]} segments...")

            segment_features = []

            for i in range(processed_data.shape[0]):
                segment = processed_data[i]  # (channels, time_points)

                # Apply filtering
                if apply_filters:
                    # Bandpass filter
                    segment = self.apply_bandpass_filter(segment)
                    # Notch filter for power line interference
                    segment = self.apply_notch_filter(segment)

                processed_data[i] = segment

                # Extract features if requested
                if extract_features:
                    features = self.extract_comprehensive_features(segment)
                    segment_features.append(features)

            # Apply ICA if requested (on all segments together)
            if apply_ica:
                print("  Applying ICA artifact removal...")
                processed_data = self.remove_artifacts_ica(processed_data)

            # Combine features
            if extract_features and segment_features:
                all_features = np.array(segment_features)

        return processed_data, all_features

def create_preprocessing_pipeline(dataset_path: str,
                                output_path: str,
                                apply_filters: bool = True,
                                apply_ica: bool = False,
                                extract_features: bool = True,
                                normalize_features: bool = True):
    """
    Create complete preprocessing pipeline for TUSZ dataset

    Args:
        dataset_path: Path to dataset
        output_path: Path to save processed data
        apply_filters: Whether to apply filtering
        apply_ica: Whether to apply ICA
        extract_features: Whether to extract features
        normalize_features: Whether to normalize features
    """

    print("🔧 Starting EEG Preprocessing Pipeline")
    print("=" * 50)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Initialize preprocessor
    preprocessor = EEGPreprocessor()

    # Load metadata
    metadata_path = dataset_path / 'dataset_metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        print("❌ Metadata file not found. Please run analyze_dataset.py first.")
        return

    # Process each split
    splits = ['train', 'dev', 'eval']

    all_features = {}
    all_labels = {}

    for split in splits:
        print(f"\n📊 Processing {split.upper()} split...")

        split_data = metadata[metadata['split'] == split]
        X_files = split_data[split_data['file_type'] == 'X']['file_path'].tolist()
        y_files = split_data[split_data['file_type'] == 'y']['file_path'].tolist()

        if not X_files or not y_files:
            print(f"  No files found for {split}")
            continue

        # Match X and y files
        file_pairs = []
        for x_file in X_files:
            y_file = x_file.replace('_X.npy', '_y.npy')
            if y_file in y_files:
                file_pairs.append((x_file, y_file))

        print(f"  Found {len(file_pairs)} file pairs")

        split_features = []
        split_labels = []
        split_processed_data = []

        for i, (x_file, y_file) in enumerate(file_pairs):
            print(f"  Processing file {i+1}/{len(file_pairs)}: {Path(x_file).name}")

            try:
                # Load data
                X_data = np.load(x_file)
                y_data = np.load(y_file)

                # Preprocess
                processed_X, features = preprocessor.preprocess_single_file(
                    X_data,
                    apply_filters=apply_filters,
                    apply_ica=apply_ica,
                    extract_features=extract_features
                )

                # Store results
                if extract_features and features is not None:
                    split_features.append(features)
                    split_labels.append(y_data)

                split_processed_data.append(processed_X)

                # Save processed raw data
                output_x_file = output_path / f"{split}_processed" / f"{Path(x_file).stem}_processed.npy"
                output_x_file.parent.mkdir(exist_ok=True)
                np.save(output_x_file, processed_X)

                output_y_file = output_path / f"{split}_processed" / f"{Path(y_file).stem}.npy"
                np.save(output_y_file, y_data)

            except Exception as e:
                print(f"    ❌ Error processing {x_file}: {e}")
                continue

        # Combine and save features
        if extract_features and split_features:
            # Combine all features and labels
            combined_features = np.vstack(split_features)
            combined_labels = np.concatenate(split_labels)

            print(f"  Combined features shape: {combined_features.shape}")
            print(f"  Combined labels shape: {combined_labels.shape}")

            # Save features
            np.save(output_path / f"{split}_features.npy", combined_features)
            np.save(output_path / f"{split}_labels.npy", combined_labels)

            all_features[split] = combined_features
            all_labels[split] = combined_labels

    # Normalize features across all splits
    if extract_features and normalize_features and all_features:
        print(f"\n🔄 Normalizing features...")

        # Fit scaler on training data
        if 'train' in all_features:
            scaler = RobustScaler()  # Less sensitive to outliers
            scaler.fit(all_features['train'])

            # Transform all splits
            for split in all_features.keys():
                normalized_features = scaler.transform(all_features[split])
                np.save(output_path / f"{split}_features_normalized.npy", normalized_features)
                print(f"  Saved normalized {split} features: {normalized_features.shape}")

            # Save scaler
            import joblib
            joblib.dump(scaler, output_path / 'feature_scaler.pkl')

    # Create summary report
    create_processing_summary(output_path, all_features, all_labels)

    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Output saved to: {output_path}")

def create_processing_summary(output_path: Path,
                            all_features: Dict,
                            all_labels: Dict):
    """Create summary report of preprocessing results"""

    print(f"\n📋 Preprocessing Summary")
    print("=" * 30)

    summary_data = []

    for split in all_features.keys():
        features = all_features[split]
        labels = all_labels[split]

        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))

        print(f"\n{split.upper()}:")
        print(f"  - Samples: {features.shape[0]}")
        print(f"  - Features: {features.shape[1]}")
        print(f"  - Label distribution: {label_dist}")
        print(f"  - Seizure ratio: {label_dist.get(1, 0) / features.shape[0]:.3f}")

        summary_data.append({
            'split': split,
            'samples': features.shape[0],
            'features': features.shape[1],
            'non_seizure': label_dist.get(0, 0),
            'seizure': label_dist.get(1, 0),
            'seizure_ratio': label_dist.get(1, 0) / features.shape[0]
        })

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'preprocessing_summary.csv', index=False)
    print(f"\n💾 Summary saved to: {output_path / 'preprocessing_summary.csv'}")

def main():
    # Configuration
    dataset_path = "/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS"
    output_path = "/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS/processed"

    # Run preprocessing pipeline
    create_preprocessing_pipeline(
        dataset_path=dataset_path,
        output_path=output_path,
        apply_filters=True,      # Apply bandpass and notch filters
        apply_ica=False,         # ICA can be computationally expensive
        extract_features=True,   # Extract comprehensive features
        normalize_features=True  # Normalize features for ML
    )

if __name__ == "__main__":
    main()
