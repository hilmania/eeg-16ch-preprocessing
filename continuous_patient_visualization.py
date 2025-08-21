#!/usr/bin/env python3
"""
Visualisasi kontinyu EEG per pasien - semua 36 segmen digabung
Membandingkan sinyal mentah vs preprocessed secara utuh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# EEG channel names (16 channels)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']

def load_patient_data(patient_id, base_path):
    """
    Load all segments for a patient and concatenate them

    Args:
        patient_id: Patient identifier
        base_path: Base path to dataset

    Returns:
        raw_continuous: Concatenated raw data (16, total_timepoints)
        processed_continuous: Concatenated processed data (16, total_timepoints)
        success: Boolean indicating if data was loaded successfully
    """
    train_path = Path(base_path) / "train"
    processed_path = Path(base_path) / "processed" / "train_processed"

    # Find patient directory in train folder
    patient_dirs = [d for d in train_path.iterdir() if d.is_dir() and patient_id in d.name]

    if not patient_dirs:
        print(f"❌ Patient {patient_id} not found in train directory")
        return None, None, False

    patient_dir = patient_dirs[0]
    print(f"📁 Found patient directory: {patient_dir.name}")

    # Load raw data - find all sessions for this patient
    raw_segments = []
    session_info = []

    for session_dir in patient_dir.iterdir():
        if session_dir.is_dir():
            # Find tcp_ar folder (could be 01_tcp_ar, 03_tcp_ar_a, etc.)
            tcp_ar_folders = [d for d in session_dir.iterdir() if d.is_dir() and 'tcp_ar' in d.name]

            for tcp_ar_path in tcp_ar_folders:
                # Load all X files (segments) from this session
                x_files = sorted([f for f in tcp_ar_path.glob("*_X.npy") if patient_id in f.name])
                print(f"📊 Session {session_dir.name}/{tcp_ar_path.name}: Found {len(x_files)} segments")

                for x_file in x_files:
                    try:
                        data = np.load(x_file)
                        print(f"   📊 Loaded {x_file.name} - shape: {data.shape}")

                        if data.shape == (36, 16, 1280):  # Multiple segments in one file
                            # Split into individual segments
                            for seg_idx in range(36):
                                segment = data[seg_idx]  # Shape: (16, 1280)
                                raw_segments.append(segment)
                                session_info.append(f"{session_dir.name}_{tcp_ar_path.name}_{x_file.stem}_seg{seg_idx:02d}")
                            print(f"   ✅ Extracted 36 segments from {x_file.name}")

                        elif data.shape == (16, 1280):  # Single segment
                            raw_segments.append(data)
                            session_info.append(f"{session_dir.name}_{tcp_ar_path.name}_{x_file.stem}")
                            print(f"   ✅ Loaded single segment from {x_file.name}")

                        else:
                            print(f"⚠️  Skipping {x_file.name}: unexpected shape {data.shape}")

                    except Exception as e:
                        print(f"❌ Error loading {x_file}: {e}")

    if not raw_segments:
        print(f"❌ No valid raw segments found for patient {patient_id}")
        return None, None, False

    print(f"✅ Loaded {len(raw_segments)} raw segments")

    # Load processed data
    processed_files = sorted(processed_path.glob(f"*{patient_id}*_X_processed.npy"))
    processed_segments = []

    print(f"🔍 Looking for processed files with pattern: *{patient_id}*_X_processed.npy")
    print(f"📁 Processed path: {processed_path}")

    for proc_file in processed_files:
        try:
            data = np.load(proc_file)
            print(f"   📊 Loaded {proc_file.name} - shape: {data.shape}")

            if data.shape == (36, 16, 1280):  # Multiple segments in one file
                # Split into individual segments
                for seg_idx in range(36):
                    segment = data[seg_idx]  # Shape: (16, 1280)
                    processed_segments.append(segment)
                print(f"   ✅ Extracted 36 segments from {proc_file.name}")

            elif data.shape == (16, 1280):  # Single segment
                processed_segments.append(data)
                print(f"   ✅ Loaded single segment from {proc_file.name}")

            else:
                print(f"⚠️  Skipping {proc_file.name}: unexpected shape {data.shape}")

        except Exception as e:
            print(f"❌ Error loading {proc_file}: {e}")

    if not processed_segments:
        print(f"❌ No processed segments found for patient {patient_id}")
        return None, None, False

    print(f"✅ Loaded {len(processed_segments)} processed segments")

    # Ensure we have data from both raw and processed
    if len(raw_segments) == 0:
        print(f"❌ No raw segments found for patient {patient_id}")
        return None, None, False

    if len(processed_segments) == 0:
        print(f"❌ No processed segments found for patient {patient_id}")
        return None, None, False

    # Use all available segments (should be same number if data is consistent)
    min_segments = min(len(raw_segments), len(processed_segments))
    print(f"📏 Using {min_segments} segments for comparison")
    print(f"   Raw segments available: {len(raw_segments)}")
    print(f"   Processed segments available: {len(processed_segments)}")

    if min_segments == 0:
        return None, None, False

    # Concatenate segments to create continuous signal
    raw_continuous = np.concatenate(raw_segments[:min_segments], axis=1)
    processed_continuous = np.concatenate(processed_segments[:min_segments], axis=1)

    print(f"🔗 Continuous signal shape: {raw_continuous.shape}")
    print(f"   Total duration: {raw_continuous.shape[1]/250:.1f} seconds")
    print(f"   Number of segments: {min_segments}")

    return raw_continuous, processed_continuous, True

def create_continuous_visualization(patient_id, raw_data, processed_data, output_dir):
    """
    Create continuous visualization comparing raw vs processed data

    Args:
        patient_id: Patient identifier
        raw_data: Raw continuous data (16, timepoints)
        processed_data: Processed continuous data (16, timepoints)
        output_dir: Output directory for saving plots
    """
    # Create figure with subplots for each channel
    fig, axes = plt.subplots(16, 1, figsize=(20, 24))
    fig.suptitle(f'Patient {patient_id} - Continuous EEG Signal Comparison\n'
                 f'Raw vs Preprocessed ({raw_data.shape[1]/250:.1f} seconds, {raw_data.shape[1]//1280} segments)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Time axis
    time_axis = np.arange(raw_data.shape[1]) / 250.0  # Convert to seconds

    # Colors
    raw_color = '#1f77b4'      # Blue
    processed_color = '#ff7f0e' # Orange

    # Plot each channel
    for ch in range(16):
        ax = axes[ch]

        # Plot raw signal
        ax.plot(time_axis, raw_data[ch], color=raw_color, alpha=0.7,
                linewidth=0.5, label='Raw' if ch == 0 else "")

        # Plot processed signal (offset for visibility)
        offset = np.max(np.abs(raw_data[ch])) * 1.2
        ax.plot(time_axis, processed_data[ch] + offset, color=processed_color,
                alpha=0.8, linewidth=0.6, label='Processed' if ch == 0 else "")

        # Add segment boundaries
        n_segments = raw_data.shape[1] // 1280
        for seg in range(1, n_segments):
            seg_time = seg * 1280 / 250.0
            ax.axvline(seg_time, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)

        # Formatting
        ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(μV)', fontsize=10, rotation=0, ha='right', va='center')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

        # Set y-limits for better visibility
        y_range = np.max(np.abs(raw_data[ch]))
        ax.set_ylim(-y_range * 0.5, offset + y_range * 0.5)

        # Remove x-axis labels except for last subplot
        if ch < 15:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (seconds)', fontsize=12)

    # Add legend
    axes[0].legend(loc='upper right', fontsize=10)

    # Add text annotations
    fig.text(0.02, 0.5, 'Raw Signal (Blue) vs Processed Signal (Orange + Offset)',
             rotation=90, va='center', fontsize=12, fontweight='bold')

    # Add segment information
    n_segments = raw_data.shape[1] // 1280
    fig.text(0.02, 0.02, f'Segments: {n_segments} | Duration: {time_axis[-1]:.1f}s | '
                         f'Sampling Rate: 250 Hz | Channels: 16',
             fontsize=10, ha='left')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.06)

    # Save plot
    output_file = output_dir / f'patient_{patient_id}_continuous_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved visualization: {output_file}")
    return output_file

def create_overview_plot(patient_id, raw_data, processed_data, output_dir):
    """
    Create overview plot showing statistical differences
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Patient {patient_id} - Signal Analysis Overview', fontsize=16, fontweight='bold')

    # 1. Amplitude comparison across channels
    ax1 = axes[0, 0]
    raw_std = np.std(raw_data, axis=1)
    proc_std = np.std(processed_data, axis=1)

    x = np.arange(16)
    width = 0.35
    ax1.bar(x - width/2, raw_std, width, label='Raw', alpha=0.7)
    ax1.bar(x + width/2, proc_std, width, label='Processed', alpha=0.7)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Standard Deviation (μV)')
    ax1.set_title('Signal Amplitude per Channel')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CHANNEL_NAMES, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Signal power over time (windowed)
    ax2 = axes[0, 1]
    window_size = 1280  # 1 segment
    n_windows = raw_data.shape[1] // window_size

    raw_power = []
    proc_power = []
    time_windows = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        raw_window = raw_data[:, start_idx:end_idx]
        proc_window = processed_data[:, start_idx:end_idx]

        raw_power.append(np.mean(np.sum(raw_window**2, axis=0)))
        proc_power.append(np.mean(np.sum(proc_window**2, axis=0)))
        time_windows.append((start_idx + end_idx) / 2 / 250.0)

    ax2.plot(time_windows, raw_power, 'o-', label='Raw', alpha=0.7)
    ax2.plot(time_windows, proc_power, 's-', label='Processed', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Average Power')
    ax2.set_title('Signal Power Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Frequency domain comparison (sample channel)
    ax3 = axes[1, 0]
    sample_ch = 2  # C3 channel

    # FFT for a representative segment
    seg_start = raw_data.shape[1] // 2
    seg_end = seg_start + 1280

    raw_seg = raw_data[sample_ch, seg_start:seg_end]
    proc_seg = processed_data[sample_ch, seg_start:seg_end]

    freqs = np.fft.fftfreq(1280, 1/250)[:640]  # Positive frequencies only
    raw_fft = np.abs(np.fft.fft(raw_seg))[:640]
    proc_fft = np.abs(np.fft.fft(proc_seg))[:640]

    ax3.semilogy(freqs, raw_fft, label='Raw', alpha=0.7)
    ax3.semilogy(freqs, proc_fft, label='Processed', alpha=0.7)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title(f'Frequency Spectrum - {CHANNEL_NAMES[sample_ch]} Channel')
    ax3.set_xlim(0, 60)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cross-correlation between raw and processed
    ax4 = axes[1, 1]
    correlations = []

    for ch in range(16):
        # Calculate correlation for each channel
        corr = np.corrcoef(raw_data[ch], processed_data[ch])[0, 1]
        correlations.append(corr)

    ax4.bar(range(16), correlations, alpha=0.7)
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Correlation Coefficient')
    ax4.set_title('Raw vs Processed Signal Correlation')
    ax4.set_xticks(range(16))
    ax4.set_xticklabels(CHANNEL_NAMES, rotation=45)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # Add mean correlation line
    mean_corr = np.mean(correlations)
    ax4.axhline(mean_corr, color='red', linestyle='--',
                label=f'Mean: {mean_corr:.3f}')
    ax4.legend()

    plt.tight_layout()

    # Save overview plot
    overview_file = output_dir / f'patient_{patient_id}_overview.png'
    plt.savefig(overview_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved overview: {overview_file}")
    return overview_file

def main():
    """Main function"""
    base_path = Path("/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS")
    output_dir = base_path / "patient_continuous_visualizations"
    output_dir.mkdir(exist_ok=True)

    print("🧠 EEG CONTINUOUS PATIENT VISUALIZATION")
    print("=" * 50)
    print(f"📁 Base path: {base_path}")
    print(f"💾 Output directory: {output_dir}")

    # Get list of available patients from train directory
    train_path = base_path / "train"
    processed_path = base_path / "processed" / "train_processed"

    if not train_path.exists():
        print(f"❌ Train directory not found: {train_path}")
        return

    if not processed_path.exists():
        print(f"❌ Processed directory not found: {processed_path}")
        return

    # Get patients that have processed data
    processed_files = list(processed_path.glob('*_X_processed.npy'))
    processed_patient_ids = set([f.name.split('_')[0] for f in processed_files])

    # Extract patient IDs from directory names that also have processed data
    patient_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    all_patient_ids = [d.name for d in patient_dirs]

    # Filter to only patients with both raw and processed data
    patient_ids = [pid for pid in all_patient_ids if pid in processed_patient_ids]
    patient_ids = sorted(list(set(patient_ids)))

    print(f"📊 Found {len(all_patient_ids)} total patients")
    print(f"🔄 Found {len(processed_patient_ids)} patients with processed data")
    print(f"✅ Available for visualization: {len(patient_ids)} patients")
    print(f"📋 First 5 available: {patient_ids[:5]}...")

    if not patient_ids:
        print("❌ No patients found with both raw and processed data!")
        print("\n💡 Possible reasons:")
        print("   1. Preprocessing script only processed patients with '01_tcp_ar' folders")
        print("   2. Some patients have '03_tcp_ar_a' folders which weren't processed")
        print("   3. Run preprocessing pipeline first to generate more processed data")
        return

    # Show some missing patients for reference
    missing_patients = [pid for pid in all_patient_ids if pid not in processed_patient_ids]
    if missing_patients:
        print(f"\n⚠️  Note: {len(missing_patients)} patients don't have processed data:")
        print(f"📋 Examples: {missing_patients[:3]}...")
        print("💡 These patients likely have '03_tcp_ar_a' folders instead of '01_tcp_ar'")

    # Ask user which patients to visualize
    print("\nOptions:")
    print("1. Visualize first 3 patients (quick demo)")
    print("2. Visualize all patients")
    print("3. Visualize specific patient")

    choice = input("\nChoose option (1/2/3): ").strip()

    patients_to_process = []

    if choice == "1":
        patients_to_process = patient_ids[:3]
    elif choice == "2":
        patients_to_process = patient_ids
    elif choice == "3":
        patient_id = input("Enter patient ID: ").strip()
        if patient_id in patient_ids:
            patients_to_process = [patient_id]
        else:
            print(f"❌ Patient {patient_id} not found!")
            return
    else:
        print("❌ Invalid choice!")
        return

    print(f"\n🎯 Processing {len(patients_to_process)} patient(s)...")

    # Process each patient
    successful_visualizations = []

    for i, patient_id in enumerate(patients_to_process, 1):
        print(f"\n{'='*20} Patient {i}/{len(patients_to_process)}: {patient_id} {'='*20}")

        # Load patient data
        raw_data, processed_data, success = load_patient_data(patient_id, base_path)

        if not success:
            print(f"❌ Failed to load data for patient {patient_id}")
            continue

        try:
            # Create visualizations
            print(f"🎨 Creating continuous visualization...")
            continuous_file = create_continuous_visualization(patient_id, raw_data, processed_data, output_dir)

            print(f"📊 Creating overview plot...")
            overview_file = create_overview_plot(patient_id, raw_data, processed_data, output_dir)

            successful_visualizations.append({
                'patient_id': patient_id,
                'continuous_file': continuous_file,
                'overview_file': overview_file,
                'n_segments': raw_data.shape[1] // 1280,
                'duration': raw_data.shape[1] / 250.0
            })

            print(f"✅ Successfully processed patient {patient_id}")

        except Exception as e:
            print(f"❌ Error processing patient {patient_id}: {e}")
            continue

    # Summary
    print(f"\n🎉 VISUALIZATION COMPLETE!")
    print("=" * 50)
    print(f"📊 Successfully processed: {len(successful_visualizations)} patients")
    print(f"💾 Output directory: {output_dir}")

    for viz in successful_visualizations:
        print(f"\n👤 Patient {viz['patient_id']}:")
        print(f"   📈 Continuous plot: {viz['continuous_file'].name}")
        print(f"   📊 Overview plot: {viz['overview_file'].name}")
        print(f"   ⏱️  Duration: {viz['duration']:.1f}s ({viz['n_segments']} segments)")

    print(f"\n💡 Open the PNG files in {output_dir} to view the visualizations!")

if __name__ == "__main__":
    main()
