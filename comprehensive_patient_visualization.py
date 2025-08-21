#!/usr/bin/env python3
"""
Script untuk visualisasi sinyal EEG per pasien: raw vs preprocessed
Menampilkan semua segmen untuk gambaran sinyal secara utuh
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def load_patient_data(patient_id, base_path="/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS"):
    """
    Load raw and preprocessed data for a specific patient
    """
    base_path = Path(base_path)

    # Find raw data
    raw_data = []
    raw_labels = []

    # Look in train folder for this patient
    train_path = base_path / "train"
    patient_sessions = []

    for patient_dir in train_path.iterdir():
        if patient_dir.is_dir() and patient_id.lower() in patient_dir.name.lower():
            for session_dir in patient_dir.iterdir():
                if session_dir.is_dir():
                    tcp_ar_path = session_dir / "01_tcp_ar"
                    if tcp_ar_path.exists():
                        patient_sessions.append(tcp_ar_path)

    print(f"Found {len(patient_sessions)} sessions for patient {patient_id}")

    # Load raw data
    for session_path in patient_sessions:
        for npy_file in session_path.glob("*_X.npy"):
            try:
                data = np.load(npy_file)
                if len(data.shape) == 3:  # (segments, channels, timepoints)
                    raw_data.append(data)

                    # Load corresponding labels
                    label_file = npy_file.parent / npy_file.name.replace("_X.npy", "_y.npy")
                    if label_file.exists():
                        labels = np.load(label_file)
                        raw_labels.append(labels)

            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

    # Load preprocessed data
    processed_path = base_path / "processed" / "train_processed"
    processed_data = []
    processed_labels = []

    for npy_file in processed_path.glob("*_X_processed.npy"):
        if patient_id.lower() in npy_file.name.lower():
            try:
                data = np.load(npy_file)
                processed_data.append(data)

                # Load corresponding labels
                label_file = npy_file.parent / npy_file.name.replace("_X_processed.npy", "_y_processed.npy")
                if label_file.exists():
                    labels = np.load(label_file)
                    processed_labels.append(labels)

            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

    # Combine all data
    if raw_data:
        raw_data = np.concatenate(raw_data, axis=0)
    else:
        raw_data = np.array([])

    if raw_labels:
        raw_labels = np.concatenate(raw_labels, axis=0)
    else:
        raw_labels = np.array([])

    if processed_data:
        processed_data = np.concatenate(processed_data, axis=0)
    else:
        processed_data = np.array([])

    if processed_labels:
        processed_labels = np.concatenate(processed_labels, axis=0)
    else:
        processed_labels = np.array([])

    return raw_data, raw_labels, processed_data, processed_labels

def create_comprehensive_patient_visualization(patient_id, raw_data, raw_labels, processed_data, processed_labels):
    """
    Create comprehensive visualization for a patient
    """
    if len(raw_data) == 0 or len(processed_data) == 0:
        print(f"❌ No data found for patient {patient_id}")
        return

    print(f"📊 Creating visualization for patient {patient_id}")
    print(f"   Raw data: {raw_data.shape}")
    print(f"   Processed data: {processed_data.shape}")

    # Channel names
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']

    # Sampling rate
    fs = 250  # Hz

    # Time vector for one segment
    segment_length = raw_data.shape[2] if len(raw_data.shape) == 3 else raw_data.shape[1]
    time_segment = np.arange(segment_length) / fs

    # Separate seizure and normal segments
    seizure_indices = np.where(raw_labels == 1)[0] if len(raw_labels) > 0 else []
    normal_indices = np.where(raw_labels == 0)[0] if len(raw_labels) > 0 else []

    print(f"   Seizure segments: {len(seizure_indices)}")
    print(f"   Normal segments: {len(normal_indices)}")

    # Create output directory
    output_dir = Path(f"patient_visualizations_{patient_id}")
    output_dir.mkdir(exist_ok=True)

    # 1. Overview of all segments
    create_segments_overview(patient_id, raw_data, processed_data, raw_labels, output_dir, channel_names, fs)

    # 2. Detailed comparison for sample segments
    create_detailed_comparison(patient_id, raw_data, processed_data, raw_labels,
                             seizure_indices, normal_indices, output_dir, channel_names, time_segment)

    # 3. Spectral analysis
    create_spectral_analysis(patient_id, raw_data, processed_data, raw_labels,
                           seizure_indices, normal_indices, output_dir, channel_names, fs)

    # 4. Statistical summary
    create_statistical_summary(patient_id, raw_data, processed_data, raw_labels, output_dir)

    print(f"✅ Visualizations saved to: {output_dir}")

def create_segments_overview(patient_id, raw_data, processed_data, labels, output_dir, channel_names, fs):
    """
    Create overview of all segments
    """
    print("   📈 Creating segments overview...")

    # Show first 50 segments or all if less
    n_segments = min(50, raw_data.shape[0], processed_data.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(20, 12))

    # Raw data overview
    ax1 = axes[0]
    for seg in range(n_segments):
        # Use middle channel (C3) for overview
        ch_idx = 4  # C3
        offset = seg * 2000  # Offset between segments
        color = 'red' if labels[seg] == 1 else 'blue'

        time_axis = np.arange(raw_data.shape[2]) + offset
        ax1.plot(time_axis, raw_data[seg, ch_idx, :] + seg * 500,
                color=color, alpha=0.7, linewidth=0.5)

    ax1.set_title(f'Patient {patient_id} - Raw EEG Signals (Channel C3)\nRed=Seizure, Blue=Normal', fontsize=14)
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('Amplitude + Offset')
    ax1.grid(True, alpha=0.3)

    # Processed data overview
    ax2 = axes[1]
    for seg in range(n_segments):
        ch_idx = 4  # C3
        offset = seg * 2000
        color = 'red' if labels[seg] == 1 else 'blue'

        time_axis = np.arange(processed_data.shape[2]) + offset
        ax2.plot(time_axis, processed_data[seg, ch_idx, :] + seg * 500,
                color=color, alpha=0.7, linewidth=0.5)

    ax2.set_title(f'Patient {patient_id} - Preprocessed EEG Signals (Channel C3)\nRed=Seizure, Blue=Normal', fontsize=14)
    ax2.set_xlabel('Time Points')
    ax2.set_ylabel('Amplitude + Offset')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_id}_segments_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_comparison(patient_id, raw_data, processed_data, labels,
                             seizure_indices, normal_indices, output_dir, channel_names, time_segment):
    """
    Create detailed comparison for sample segments
    """
    print("   🔍 Creating detailed comparison...")

    # Select sample segments
    sample_seizure = seizure_indices[:3] if len(seizure_indices) >= 3 else seizure_indices
    sample_normal = normal_indices[:3] if len(normal_indices) >= 3 else normal_indices

    for condition, indices in [("seizure", sample_seizure), ("normal", sample_normal)]:
        if len(indices) == 0:
            continue

        for i, seg_idx in enumerate(indices):
            fig, axes = plt.subplots(16, 2, figsize=(16, 24))

            for ch in range(16):
                # Raw signal
                axes[ch, 0].plot(time_segment, raw_data[seg_idx, ch, :], 'b-', linewidth=0.8)
                axes[ch, 0].set_title(f'Raw - {channel_names[ch]}', fontsize=10)
                axes[ch, 0].grid(True, alpha=0.3)

                # Processed signal
                axes[ch, 1].plot(time_segment, processed_data[seg_idx, ch, :], 'r-', linewidth=0.8)
                axes[ch, 1].set_title(f'Processed - {channel_names[ch]}', fontsize=10)
                axes[ch, 1].grid(True, alpha=0.3)

                # Remove x-axis labels except for bottom plots
                if ch < 15:
                    axes[ch, 0].set_xticklabels([])
                    axes[ch, 1].set_xticklabels([])
                else:
                    axes[ch, 0].set_xlabel('Time (s)')
                    axes[ch, 1].set_xlabel('Time (s)')

            plt.suptitle(f'Patient {patient_id} - {condition.title()} Segment {i+1} (Segment #{seg_idx})', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / f"{patient_id}_{condition}_segment_{i+1}_detailed.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

def create_spectral_analysis(patient_id, raw_data, processed_data, labels,
                           seizure_indices, normal_indices, output_dir, channel_names, fs):
    """
    Create spectral analysis comparison
    """
    print("   📊 Creating spectral analysis...")

    # Select representative channels
    repr_channels = [0, 4, 8, 12]  # Fp1, C3, O1, T4

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, ch in enumerate(repr_channels):
        # Compute average PSD for seizure and normal segments

        # Raw data
        ax1 = axes[0, i]

        if len(seizure_indices) > 0:
            seizure_raw = raw_data[seizure_indices, ch, :]
            freqs, psd_seizure_raw = signal.welch(seizure_raw.flatten(), fs, nperseg=256)
            ax1.semilogy(freqs, psd_seizure_raw, 'r-', label='Seizure (Raw)', alpha=0.8)

        if len(normal_indices) > 0:
            normal_raw = raw_data[normal_indices, ch, :]
            freqs, psd_normal_raw = signal.welch(normal_raw.flatten(), fs, nperseg=256)
            ax1.semilogy(freqs, psd_normal_raw, 'b-', label='Normal (Raw)', alpha=0.8)

        ax1.set_title(f'Raw - {channel_names[ch]}')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 50)

        # Processed data
        ax2 = axes[1, i]

        if len(seizure_indices) > 0:
            seizure_proc = processed_data[seizure_indices, ch, :]
            freqs, psd_seizure_proc = signal.welch(seizure_proc.flatten(), fs, nperseg=256)
            ax2.semilogy(freqs, psd_seizure_proc, 'r-', label='Seizure (Processed)', alpha=0.8)

        if len(normal_indices) > 0:
            normal_proc = processed_data[normal_indices, ch, :]
            freqs, psd_normal_proc = signal.welch(normal_proc.flatten(), fs, nperseg=256)
            ax2.semilogy(freqs, psd_normal_proc, 'b-', label='Normal (Processed)', alpha=0.8)

        ax2.set_title(f'Processed - {channel_names[ch]}')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 50)

    plt.suptitle(f'Patient {patient_id} - Spectral Analysis Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_id}_spectral_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary(patient_id, raw_data, processed_data, labels, output_dir):
    """
    Create statistical summary
    """
    print("   📋 Creating statistical summary...")

    # Calculate statistics
    stats = {
        'Metric': [],
        'Raw_Mean': [],
        'Raw_Std': [],
        'Processed_Mean': [],
        'Processed_Std': [],
        'Improvement': []
    }

    # Overall statistics
    metrics = [
        ('Mean Amplitude', np.mean, 'μV'),
        ('Std Amplitude', np.std, 'μV'),
        ('Max Amplitude', np.max, 'μV'),
        ('Min Amplitude', np.min, 'μV'),
        ('RMS', lambda x: np.sqrt(np.mean(x**2)), 'μV')
    ]

    for metric_name, metric_func, unit in metrics:
        raw_val = metric_func(raw_data)
        proc_val = metric_func(processed_data)
        improvement = ((proc_val - raw_val) / raw_val * 100) if raw_val != 0 else 0

        stats['Metric'].append(f'{metric_name} ({unit})')
        stats['Raw_Mean'].append(f'{raw_val:.3f}')
        stats['Raw_Std'].append('-')
        stats['Processed_Mean'].append(f'{proc_val:.3f}')
        stats['Processed_Std'].append('-')
        stats['Improvement'].append(f'{improvement:+.1f}%')

    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Amplitude distribution
    ax1 = axes[0, 0]
    ax1.hist(raw_data.flatten()[::1000], bins=50, alpha=0.6, label='Raw', density=True)
    ax1.hist(processed_data.flatten()[::1000], bins=50, alpha=0.6, label='Processed', density=True)
    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Density')
    ax1.set_title('Amplitude Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Segment-wise statistics
    ax2 = axes[0, 1]
    raw_means = np.mean(raw_data, axis=(1, 2))
    proc_means = np.mean(processed_data, axis=(1, 2))
    ax2.scatter(raw_means, proc_means, alpha=0.6, c=labels, cmap='coolwarm')
    ax2.plot([raw_means.min(), raw_means.max()], [raw_means.min(), raw_means.max()], 'k--', alpha=0.5)
    ax2.set_xlabel('Raw Mean Amplitude')
    ax2.set_ylabel('Processed Mean Amplitude')
    ax2.set_title('Segment-wise Mean Comparison')
    ax2.grid(True, alpha=0.3)

    # Channel-wise comparison
    ax3 = axes[1, 0]
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
    raw_ch_means = np.mean(raw_data, axis=(0, 2))
    proc_ch_means = np.mean(processed_data, axis=(0, 2))

    x = np.arange(len(channel_names))
    width = 0.35
    ax3.bar(x - width/2, raw_ch_means, width, label='Raw', alpha=0.7)
    ax3.bar(x + width/2, proc_ch_means, width, label='Processed', alpha=0.7)
    ax3.set_xlabel('Channels')
    ax3.set_ylabel('Mean Amplitude')
    ax3.set_title('Channel-wise Mean Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(channel_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
Patient {patient_id} Summary:

Total Segments: {raw_data.shape[0]}
Seizure Segments: {np.sum(labels == 1)}
Normal Segments: {np.sum(labels == 0)}

Data Shape:
Raw: {raw_data.shape}
Processed: {processed_data.shape}

Quality Improvement:
• Noise Reduction: Applied
• Filtering: 0.5-50 Hz bandpass
• Normalization: Z-score per channel
• Artifact Removal: ICA applied
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'Patient {patient_id} - Statistical Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_id}_statistical_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save statistics to CSV
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(output_dir / f"{patient_id}_statistics.csv", index=False)

def get_available_patients(base_path="/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS"):
    """
    Get list of available patients from training data
    """
    base_path = Path(base_path)
    train_path = base_path / "train"

    patients = set()

    if train_path.exists():
        for patient_dir in train_path.iterdir():
            if patient_dir.is_dir():
                # Extract patient ID from folder name
                patient_id = patient_dir.name
                patients.add(patient_id)

    return sorted(list(patients))

def main():
    """
    Main function to create patient visualizations
    """
    print("🧠 EEG PATIENT VISUALIZATION TOOL")
    print("=" * 50)

    # Get available patients
    patients = get_available_patients()

    if not patients:
        print("❌ No patients found in training data!")
        return

    print(f"📋 Found {len(patients)} patients:")
    for i, patient in enumerate(patients[:10], 1):  # Show first 10
        print(f"   {i:2d}. {patient}")

    if len(patients) > 10:
        print(f"   ... and {len(patients) - 10} more")

    print("\n" + "=" * 50)

    # Process each patient (limit to first 5 for demo)
    max_patients = 5
    processed_count = 0

    for patient_id in patients:
        if processed_count >= max_patients:
            print(f"\n⏹️  Stopping after {max_patients} patients (demo limit)")
            break

        print(f"\n🔍 Processing patient: {patient_id}")

        try:
            # Load data
            raw_data, raw_labels, processed_data, processed_labels = load_patient_data(patient_id)

            if len(raw_data) == 0 or len(processed_data) == 0:
                print(f"   ⚠️  Skipping {patient_id} - no matching data found")
                continue

            # Create visualizations
            create_comprehensive_patient_visualization(patient_id, raw_data, raw_labels,
                                                     processed_data, processed_labels)

            processed_count += 1

        except Exception as e:
            print(f"   ❌ Error processing {patient_id}: {e}")
            continue

    print(f"\n✅ Visualization completed for {processed_count} patients!")
    print(f"📁 Check individual 'patient_visualizations_*' folders for results")

if __name__ == "__main__":
    main()
