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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
                 segment_length: int = 1280,
                 # Channel selection config
                 channel_names: Optional[List[str]] = None,
                 channel_selection_method: str = 'none',  # 'none' | 'by_name' | 'variance_topk'
                 selected_channels: Optional[List[str]] = None,
                 drop_channels: Optional[List[str]] = None,
                 channel_selection_k: Optional[int] = None,
                 channel_selection_metric: str = 'variance',  # currently only 'variance'
                 apply_channel_selection_to_raw: bool = False,
                 # Wrapper selection config (optional)
                 precomputed_channel_indices: Optional[np.ndarray] = None):
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

        # Channels meta
        default_channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                                 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
        self.channel_names = channel_names if channel_names is not None else default_channel_names[:channels]

        # Channel selection configuration
        self.channel_selection_method = channel_selection_method
        self.selected_channels_cfg = selected_channels
        self.drop_channels_cfg = drop_channels
        self.channel_selection_k = channel_selection_k
        self.channel_selection_metric = channel_selection_metric
        self.apply_channel_selection_to_raw = apply_channel_selection_to_raw
        self._last_selected_indices = None
        # For wrapper method: indices selected offline
        self.precomputed_channel_indices = precomputed_channel_indices

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
        # Channel selection summary
        if self.channel_selection_method != 'none':
            print(f"   - Channel selection: {self.channel_selection_method}")
            if self.channel_selection_method == 'by_name':
                print(f"     · Keep: {self.selected_channels_cfg} | Drop: {self.drop_channels_cfg}")
            if self.channel_selection_method == 'variance_topk':
                print(f"     · Metric: {self.channel_selection_metric} | Top-k: {self.channel_selection_k}")
            if self.channel_selection_method == 'wrapper':
                print(f"     · Wrapper selection active (indices provided: {self.precomputed_channel_indices is not None})")
            print(f"     · Apply to saved raw: {self.apply_channel_selection_to_raw}")

    # ---------------------- Channel selection helpers ----------------------
    def _indices_from_channel_names(self, keep: Optional[List[str]], drop: Optional[List[str]]) -> Optional[np.ndarray]:
        """
        Build channel indices from names to keep/drop based on self.channel_names.

        Priority: if keep provided -> use keep; else if drop provided -> use all except drop; else None
        """
        if not self.channel_names:
            return None
        name_to_idx = {name: i for i, name in enumerate(self.channel_names)}

        if keep:
            valid = [name for name in keep if name in name_to_idx]
            if len(valid) == 0:
                print("⚠️  Channel selection by_name: no valid channel names found. Skipping selection.")
                return None
            return np.array([name_to_idx[name] for name in valid], dtype=int)

        if drop:
            drop_set = {name for name in drop if name in name_to_idx}
            indices = [i for i, n in enumerate(self.channel_names) if n not in drop_set]
            if len(indices) == 0:
                print("⚠️  Channel selection by_name: dropping all channels is not allowed. Skipping selection.")
                return None
            return np.array(indices, dtype=int)

        return None

    def _select_channels_for_file(self, X_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Decide which channel indices to keep for the entire file (consistent across all segments).

        X_data shape: (segments, channels, time_points)
        Returns indices array or None if no selection.
        """
        method = (self.channel_selection_method or 'none').lower()
        if method == 'none':
            return None

        if method == 'by_name':
            idx = self._indices_from_channel_names(self.selected_channels_cfg, self.drop_channels_cfg)
            return idx

        if method == 'variance_topk':
            k = self.channel_selection_k or self.channels
            k = max(1, min(k, X_data.shape[1]))
            # Compute per-channel metric across all segments/time
            if self.channel_selection_metric == 'variance':
                ch_var_per_segment = np.var(X_data, axis=2)  # (segments, channels)
                ch_metric = np.mean(ch_var_per_segment, axis=0)  # (channels,)
            else:
                ch_var_per_segment = np.var(X_data, axis=2)
                ch_metric = np.mean(ch_var_per_segment, axis=0)

            indices = np.argsort(ch_metric)[-k:]
            indices = np.sort(indices)
            if self.channel_names and len(self.channel_names) >= indices.max() + 1:
                sel_names = [self.channel_names[i] for i in indices]
                print(f"   · Selected channels (top-{k} by {self.channel_selection_metric}): {sel_names}")
            else:
                print(f"   · Selected channel indices (top-{k}): {indices.tolist()}")
            return indices

        if method == 'wrapper':
            # Use precomputed indices if provided
            if self.precomputed_channel_indices is not None and len(self.precomputed_channel_indices) > 0:
                return np.array(self.precomputed_channel_indices, dtype=int)
            print("⚠️  Wrapper selection requested but no precomputed indices provided. Skipping selection.")
            return None

        print(f"⚠️  Unknown channel_selection_method: {self.channel_selection_method}. Skipping selection.")
        return None

    # ---------------------- Wrapper selection utilities ----------------------
    def _compute_simple_channel_features(self, X_data: np.ndarray) -> np.ndarray:
        """
        Compute fast per-channel features for wrapper selection.

        X_data: (segments, channels, time)
        Returns: (segments, channels, f_dim)
        """
        segs, chs, T = X_data.shape
        # Features: variance, mean abs, band powers (delta..gamma)
        fdim = 2 + 5  # var, mean_abs, 5 bandpowers
        out = np.zeros((segs, chs, fdim), dtype=np.float32)
        # Welch params
        nperseg = min(256, T)
        for s in range(segs):
            for c in range(chs):
                x = X_data[s, c]
                var = np.var(x)
                mabs = np.mean(np.abs(x))
                # Bandpowers via Welch
                freqs, psd = signal.welch(x, fs=self.fs, nperseg=nperseg)
                def band_power(blo, bhi):
                    mask = (freqs >= blo) & (freqs < bhi)
                    return np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0
                bp = [
                    band_power(0.5, 4),
                    band_power(4, 8),
                    band_power(8, 13),
                    band_power(13, 30),
                    band_power(30, 50)
                ]
                out[s, c, :] = np.array([var, mabs] + bp, dtype=np.float32)
        return out

    def _evaluate_channel_subset(self, F_chan: np.ndarray, y: np.ndarray, subset: List[int],
                                 model: str = 'logreg', scoring: str = 'roc_auc', cv: int = 3) -> float:
        """
        Evaluate a subset of channels using cross-validation on simple features.
        F_chan: (n_samples, channels, f_dim) aggregated across files
        y: (n_samples,)
        subset: list of channel indices
        Returns mean CV score.
        """
        if len(subset) == 0:
            return 0.0
        X = F_chan[:, subset, :].reshape(F_chan.shape[0], -1)
        # Choose model
        if model == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            pipe = clf  # RF ok without scaling
        else:
            clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
            pipe = make_pipeline(StandardScaler(with_mean=False), clf)
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.nanmean(scores))
        except Exception:
            # Fallback to accuracy if scoring fails
            try:
                scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                return float(np.nanmean(scores))
            except Exception:
                return 0.0

    def select_channels_wrapper(self,
                                train_files: List[Tuple[str, str]],
                                max_k: int = 8,
                                model: str = 'logreg',
                                scoring: str = 'roc_auc',
                                cv: int = 3,
                                max_files: int = 10,
                                max_segments_per_file: int = 200) -> np.ndarray:
        """
        Greedy forward wrapper selection on a subset of training data.
        Returns selected channel indices.
        """
        sel_files = train_files[:max_files] if max_files else train_files
        F_list = []
        y_list = []
        for x_path, y_path in sel_files:
            try:
                X = np.load(x_path)
                y = np.load(y_path)
                # Ensure shape (segments, channels, time)
                if X.ndim == 3 and X.shape[0] == 36:
                    pass
                elif X.ndim == 2:
                    X = X[np.newaxis, ...]
                else:
                    # Try to reshape (channels, time) repeated 36
                    if X.shape[-1] == self.segment_length and X.shape[-2] == self.channels:
                        X = X[np.newaxis, ...]
                # Subsample segments
                if max_segments_per_file and X.shape[0] > max_segments_per_file:
                    X = X[:max_segments_per_file]
                    y = y[:max_segments_per_file]
                F = self._compute_simple_channel_features(X)  # (segs, ch, f)
                F_list.append(F)
                y_list.append(y)
            except Exception as e:
                print(f"    ⚠️  Skipping file for wrapper selection ({x_path}): {e}")
                continue
        if not F_list:
            print("⚠️  No data collected for wrapper selection. Falling back to no selection.")
            return np.array([], dtype=int)
        F_all = np.concatenate(F_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        # Ensure both classes exist
        if len(np.unique(y_all)) < 2:
            print("⚠️  Wrapper selection requires at least two classes. Skipping selection.")
            return np.array([], dtype=int)

        remaining = list(range(F_all.shape[1]))
        selected: List[int] = []
        best_score = -np.inf
        while len(selected) < max_k and remaining:
            best_candidate = None
            best_candidate_score = best_score
            for ch in remaining:
                subset = selected + [ch]
                score = self._evaluate_channel_subset(F_all, y_all, subset, model=model, scoring=scoring, cv=cv)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = ch
            if best_candidate is None or best_candidate_score <= best_score:
                break  # No improvement
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            if self.channel_names and best_candidate < len(self.channel_names):
                print(f"    ✓ Added channel {self.channel_names[best_candidate]} -> CV {best_score:.4f}")
            else:
                print(f"    ✓ Added channel idx {best_candidate} -> CV {best_score:.4f}")

        print(f"  Wrapper selected {len(selected)} channels. Best CV score: {best_score if best_score>-np.inf else 0:.4f}")
        return np.array(sorted(selected), dtype=int)

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
        # Infer channel count from data to support channel selection
        # data shape: (segments, channels, time_points)
        channels_local = data.shape[1]
        segments_local = data.shape[0]
        if n_components is None:
            n_components = min(channels_local, segments_local)

        # Reshape data for ICA (time_points, channels)
        original_shape = data.shape
        reshaped_data = data.reshape(-1, channels_local)

        # Apply ICA
        ica = FastICA(n_components=min(n_components, channels_local), random_state=42)
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

            # Decide channel indices once per file (consistent across all segments)
            ch_indices = self._select_channels_for_file(processed_data) if self.channel_selection_method != 'none' else None
            if ch_indices is not None:
                self._last_selected_indices = ch_indices
                if self.channel_names and (ch_indices.max() < len(self.channel_names)):
                    names = [self.channel_names[i] for i in ch_indices]
                    print(f"  🔎 Channel selection active: keeping {len(ch_indices)} channels -> {names}")
                else:
                    print(f"  🔎 Channel selection active: keeping indices {ch_indices.tolist()}")

            for i in range(processed_data.shape[0]):
                segment = processed_data[i]  # (channels, time_points)

                # Apply filtering
                if apply_filters:
                    # Bandpass filter
                    segment = self.apply_bandpass_filter(segment)
                    # Notch filter for power line interference
                    segment = self.apply_notch_filter(segment)

                # Apply channel selection to segment for features and optionally for saved raw
                seg_for_features = segment
                if ch_indices is not None:
                    seg_for_features = segment[ch_indices, :]
                    if self.apply_channel_selection_to_raw:
                        segment = seg_for_features

                processed_data[i] = segment

                # Extract features if requested
                if extract_features:
                    features = self.extract_comprehensive_features(seg_for_features)
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
                                normalize_features: bool = True,
                                # Channel selection options
                                channel_selection_method: str = 'none',
                                selected_channels: Optional[List[str]] = None,
                                drop_channels: Optional[List[str]] = None,
                                channel_selection_k: Optional[int] = None,
                                channel_selection_metric: str = 'variance',
                                apply_channel_selection_to_raw: bool = False,
                                channel_names: Optional[List[str]] = None,
                                channels: int = 16,
                                sampling_rate: int = 250,
                                segment_length: int = 1280,
                                # Wrapper config
                                wrapper_max_k: int = 8,
                                wrapper_model: str = 'logreg',
                                wrapper_scoring: str = 'roc_auc',
                                wrapper_cv: int = 3,
                                wrapper_max_files: int = 10,
                                wrapper_max_segments_per_file: int = 200):
    """
    Create complete preprocessing pipeline for TUSZ dataset

    Args:
        dataset_path: Path to dataset
        output_path: Path to save processed data
        apply_filters: Whether to apply filtering
        apply_ica: Whether to apply ICA
        extract_features: Whether to extract features
    normalize_features: Whether to normalize features
    channel_selection_method: 'none' | 'by_name' | 'variance_topk'
    selected_channels: list of channel names to keep (by_name)
    drop_channels: list of channel names to drop (by_name)
    channel_selection_k: top-k channels to select (variance_topk)
    channel_selection_metric: metric for top-k (currently only 'variance')
    apply_channel_selection_to_raw: if True, saved processed raw will have only selected channels
    channel_names: channel name list (length == channels)
    channels/sampling_rate/segment_length: preprocessor basics
    Wrapper: wrapper_max_k, wrapper_model [logreg|rf], wrapper_scoring, wrapper_cv, wrapper_max_files, wrapper_max_segments_per_file
    """

    print("🔧 Starting EEG Preprocessing Pipeline")
    print("=" * 50)

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Initialize preprocessor (may update precomputed indices later for wrapper)
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        channels=channels,
        segment_length=segment_length,
        channel_names=channel_names,
        channel_selection_method=channel_selection_method,
        selected_channels=selected_channels,
        drop_channels=drop_channels,
        channel_selection_k=channel_selection_k,
        channel_selection_metric=channel_selection_metric,
        apply_channel_selection_to_raw=apply_channel_selection_to_raw
    )

    # Load metadata
    metadata_path = dataset_path / 'dataset_metadata.csv'
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        print("❌ Metadata file not found. Please run analyze_dataset.py first.")
        return

    # If wrapper selection: compute once on TRAIN and reuse
    wrapper_indices: Optional[np.ndarray] = None
    if channel_selection_method == 'wrapper':
        print("🧩 Running wrapper-based channel selection on TRAIN split...")
        # Gather train file pairs
        md = pd.read_csv(dataset_path / 'dataset_metadata.csv') if (dataset_path / 'dataset_metadata.csv').exists() else None
        if md is None:
            print("❌ Metadata file not found for wrapper selection.")
        else:
            split_data = md[md['split'] == 'train']
            X_train_files = split_data[split_data['file_type'] == 'X']['file_path'].tolist()
            y_train_files = split_data[split_data['file_type'] == 'y']['file_path'].tolist()
            train_pairs = []
            for xfp in X_train_files:
                yfp = xfp.replace('_X.npy', '_y.npy')
                if yfp in y_train_files:
                    train_pairs.append((xfp, yfp))
            if not train_pairs:
                print("⚠️  No train file pairs for wrapper selection.")
            else:
                wrapper_indices = preprocessor.select_channels_wrapper(
                    train_pairs,
                    max_k=wrapper_max_k,
                    model=wrapper_model,
                    scoring=wrapper_scoring,
                    cv=wrapper_cv,
                    max_files=wrapper_max_files,
                    max_segments_per_file=wrapper_max_segments_per_file,
                )
                if wrapper_indices is not None and len(wrapper_indices) > 0:
                    preprocessor.precomputed_channel_indices = wrapper_indices
                    if preprocessor.channel_names and (wrapper_indices.max() < len(preprocessor.channel_names)):
                        names = [preprocessor.channel_names[i] for i in wrapper_indices]
                        print(f"   → Wrapper selected channels: {names}")
                    else:
                        print(f"   → Wrapper selected channel indices: {wrapper_indices.tolist()}")
                    # Save selection
                    import json
                    sel_out = output_path / 'channel_selection_wrapper.json'
                    output_path.mkdir(exist_ok=True)
                    with open(sel_out, 'w') as f:
                        json.dump({
                            'indices': [int(i) for i in wrapper_indices.tolist()],
                            'names': [preprocessor.channel_names[i] for i in wrapper_indices] if preprocessor.channel_names and (wrapper_indices.max() < len(preprocessor.channel_names)) else None,
                            'params': {
                                'max_k': wrapper_max_k,
                                'model': wrapper_model,
                                'scoring': wrapper_scoring,
                                'cv': wrapper_cv,
                                'max_files': wrapper_max_files,
                                'max_segments_per_file': wrapper_max_segments_per_file
                            }
                        }, f, indent=2)
                        print(f"💾 Saved wrapper selection to: {sel_out}")

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
                # Ensure wrapper indices propagated (for non-train splits or after computation)
                if channel_selection_method == 'wrapper' and wrapper_indices is not None:
                    preprocessor.precomputed_channel_indices = wrapper_indices

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
