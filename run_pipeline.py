#!/usr/bin/env python3
"""
Main Pipeline Script for EEG Seizure Detection
Script utama untuk menjalankan seluruh pipeline dari analisis hingga klasifikasi
"""

import sys
import argparse
from pathlib import Path
import time

def run_analysis():
    """Run dataset analysis"""
    print("🔍 STEP 1: Dataset Analysis")
    print("=" * 40)

    try:
        from analyze_dataset import main as analyze_main
        analyze_main()
        print("✅ Dataset analysis completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error in dataset analysis: {e}\n")
        return False

def run_preprocessing():
    """Run preprocessing pipeline"""
    print("🔧 STEP 2: Data Preprocessing")
    print("=" * 40)

    try:
        from eeg_preprocessing import main as preprocess_main
        preprocess_main()
        print("✅ Preprocessing completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}\n")
        return False

def run_classification():
    """Run classification pipeline"""
    print("🎯 STEP 3: Model Training and Classification")
    print("=" * 40)

    try:
        from eeg_classification import main as classify_main
        classify_main()
        print("✅ Classification completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error in classification: {e}\n")
        return False

def run_visualization():
    """Run visualization and analysis"""
    print("📊 STEP 4: Visualization and Analysis")
    print("=" * 40)

    try:
        from eeg_visualization import main as viz_main
        viz_main()
        print("✅ Visualization completed successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error in visualization: {e}\n")
        return False

def run_preprocessing_visualization():
    """Run preprocessing comparison visualization"""
    print("📈 STEP 5: Preprocessing Comparison Visualization")
    print("=" * 50)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        from eeg_visualization import EEGVisualizer

        visualizer = EEGVisualizer()
        output_dir = Path('preprocessing_analysis')
        output_dir.mkdir(exist_ok=True)

        # Analyze preprocessing effects
        visualizer.analyze_preprocessing_effects(
            dataset_path=".",
            processed_path="processed_data",
            output_dir=str(output_dir)
        )

        print("✅ Preprocessing visualization completed successfully!")
        print(f"📊 Results saved to: {output_dir.absolute()}\n")
        return True

    except Exception as e:
        print(f"❌ Error in preprocessing visualization: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def check_requirements():
    """Check if all required packages are installed"""
    # Map package names to their import names
    package_imports = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'joblib': 'joblib'
    }

    missing_packages = []
    available_packages = []

    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            available_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)

    # Print available packages for debugging
    if len(available_packages) > 0:
        print(f"✅ Found {len(available_packages)} packages: {', '.join(available_packages)}")

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def print_banner():
    """Print banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    EEG SEIZURE DETECTION                     ║
║                   Comprehensive Pipeline                     ║
║                                                              ║
║  📊 Dataset Analysis → 🔧 Preprocessing →                    ║
║  🎯 Classification → 📈 Visualization                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_summary(results):
    """Print pipeline execution summary"""
    print("\n" + "=" * 60)
    print("📋 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    steps = [
        ("Dataset Analysis", results.get('analysis', False)),
        ("Data Preprocessing", results.get('preprocessing', False)),
        ("Model Classification", results.get('classification', False)),
        ("Visualization & Analysis", results.get('visualization', False))
    ]

    # Add preprocessing visualization if it was run
    if 'preprocessing_viz' in results and results['preprocessing_viz'] is not False:
        steps.append(("Preprocessing Comparison", results['preprocessing_viz']))

    for step_name, success in steps:
        if success is not False:  # Only show steps that were actually run
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{step_name:<25} {status}")

    executed_steps = [success for success in results.values() if success is not False]
    overall_success = all(executed_steps) if executed_steps else False

    print("\n" + "-" * 60)
    if overall_success:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n📁 Check the following directories for results:")
        print("   - dataset_metadata.csv (dataset information)")
        print("   - processed_data/ (preprocessed data and features)")
        print("   - analysis/ (visualizations and reports)")
        print("   - preprocessing_analysis/ (preprocessing comparisons)")
    else:
        print("⚠️  SOME STEPS FAILED - Check error messages above")

    print("\n" + "=" * 60)

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="EEG Seizure Detection Pipeline")
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip dataset analysis step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip classification step')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--preprocessing-viz', action='store_true',
                       help='Create preprocessing comparison visualizations')
    parser.add_argument('--step', choices=['analysis', 'preprocessing', 'classification', 'visualization', 'preprocessing-viz'],
                       help='Run only a specific step')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All requirements satisfied!\n")

    # Record start time
    start_time = time.time()

    # Initialize results
    results = {
        'analysis': False,
        'preprocessing': False,
        'classification': False,
        'visualization': False,
        'preprocessing_viz': False
    }

    # Handle special case for preprocessing visualization
    if args.preprocessing_viz:
        print("📈 Running preprocessing comparison visualization only...")
        results['preprocessing_viz'] = run_preprocessing_visualization()

        # Print results
        if results['preprocessing_viz']:
            print("🎉 Preprocessing visualization completed successfully!")
        else:
            print("❌ Preprocessing visualization failed!")

        return

    # Determine which steps to run
    if args.step:
        # Run only specific step
        if args.step == 'analysis':
            results['analysis'] = run_analysis()
        elif args.step == 'preprocessing':
            results['preprocessing'] = run_preprocessing()
        elif args.step == 'classification':
            results['classification'] = run_classification()
        elif args.step == 'visualization':
            results['visualization'] = run_visualization()
        elif args.step == 'preprocessing-viz':
            results['preprocessing_viz'] = run_preprocessing_visualization()
    else:
        # Run full pipeline with skip options
        if not args.skip_analysis:
            results['analysis'] = run_analysis()
        else:
            print("⏭️  Skipping dataset analysis")
            results['analysis'] = True  # Assume success if skipped

        if not args.skip_preprocessing and results['analysis']:
            results['preprocessing'] = run_preprocessing()
        elif args.skip_preprocessing:
            print("⏭️  Skipping preprocessing")
            results['preprocessing'] = True

        if not args.skip_classification and results['preprocessing']:
            results['classification'] = run_classification()
        elif args.skip_classification:
            print("⏭️  Skipping classification")
            results['classification'] = True

        if not args.skip_visualization:
            results['visualization'] = run_visualization()
        else:
            print("⏭️  Skipping visualization")
            results['visualization'] = True

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print summary
    print_summary(results)
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
