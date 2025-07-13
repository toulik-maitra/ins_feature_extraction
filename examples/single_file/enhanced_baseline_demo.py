"""
Enhanced Baseline Detection System - Comprehensive Demo
======================================================

This script demonstrates the complete enhanced baseline detection system with:
- Multiple baseline algorithms from pybaselines and spectrochempy
- Parameter optimization with validation
- Quality metrics and comparison
- Integration with the existing ML analysis pipeline
- Comprehensive reporting and visualization

Author: Enhanced Baseline Detection System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.enhanced_baseline_detection import (
    EnhancedBaselineDetectorFactory,
    BaselineValidationSystem,
    BaselineQualityMetrics
)
from core.enhanced_ml_peak_analyzer import EnhancedMLPeakAnalyzer, analyze_spectrum_with_enhanced_baseline

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_realistic_test_spectrum():
    """Create a realistic test spectrum with known baseline for validation."""
    
    # Energy range
    energy = np.linspace(0, 3500, 1000)
    
    # Create true baseline (complex baseline with multiple components)
    true_baseline = (
        0.05 +  # Constant offset
        0.0001 * energy +  # Linear trend
        0.0000001 * energy**2 +  # Quadratic trend
        0.1 * np.exp(-energy / 1000) +  # Exponential decay
        0.02 * np.sin(energy / 500)  # Oscillatory component
    )
    
    # Add peaks at realistic INS positions
    peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
    peak_widths = [30, 25, 35, 40, 30, 35, 25]
    
    # Create spectrum
    spectrum = true_baseline.copy()
    for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
        peak = amp * np.exp(-(energy - pos)**2 / (2 * width**2))
        spectrum += peak
    
    # Add realistic noise (different levels for different regions)
    noise_levels = np.where(energy < 1000, 0.01, 0.02)  # Higher noise at higher energy
    noise = np.random.normal(0, 1, len(energy)) * noise_levels
    spectrum += noise
    
    return energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths

def demo_available_detectors():
    """Demonstrate all available baseline detectors."""
    
    print("="*80)
    print("DEMONSTRATING AVAILABLE BASELINE DETECTORS")
    print("="*80)
    
    # Get available detectors
    available_detectors = EnhancedBaselineDetectorFactory.get_available_detectors()
    print(f"Available detectors: {available_detectors}")
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_realistic_test_spectrum()
    
    # Test each detector
    results = []
    
    for detector_type in available_detectors:
        print(f"\nTesting {detector_type}...")
        
        try:
            # Create detector
            detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
            
            # Detect baseline
            start_time = time.time()
            estimated_baseline = detector.detect_baseline(spectrum, energy)
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            corrected_spectrum = spectrum - estimated_baseline
            metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
                true_baseline, estimated_baseline, spectrum, corrected_spectrum,
                np.array(peak_positions, dtype=int)
            )
            
            result = {
                'detector': detector_type,
                'processing_time': processing_time,
                **metrics
            }
            results.append(result)
            
            print(f"  ✓ Success: RMSE = {metrics['rmse']:.6f}, "
                  f"Correlation = {metrics['correlation']:.4f}, "
                  f"Time = {processing_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'detector': detector_type,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def demo_parameter_optimization():
    """Demonstrate parameter optimization for selected detectors."""
    
    print("\n" + "="*80)
    print("DEMONSTRATING PARAMETER OPTIMIZATION")
    print("="*80)
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_realistic_test_spectrum()
    
    # Test detectors that support optimization
    test_detectors = ['pybaselines_asls', 'pybaselines_airpls', 'spectrochempy_polynomial']
    
    optimization_results = []
    
    for detector_type in test_detectors:
        print(f"\nOptimizing {detector_type}...")
        
        try:
            # Create detector
            detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
            
            # Optimize parameters
            optimization_result = detector.optimize_parameters(
                intensity=spectrum,
                energy=energy,
                true_baseline=true_baseline,
                peak_positions=np.array(peak_positions, dtype=int),
                max_iterations=20  # Reduced for demo
            )
            
            # Test with optimized parameters
            optimized_baseline = detector.detect_baseline(spectrum, energy)
            corrected_spectrum = spectrum - optimized_baseline
            
            metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
                true_baseline, optimized_baseline, spectrum, corrected_spectrum,
                np.array(peak_positions, dtype=int)
            )
            
            result = {
                'detector': detector_type,
                'best_score': optimization_result['best_score'],
                'best_parameters': optimization_result['best_parameters'],
                'optimized_rmse': metrics['rmse'],
                'optimized_correlation': metrics['correlation']
            }
            optimization_results.append(result)
            
            print(f"  ✓ Optimization completed: Best score = {optimization_result['best_score']:.6f}")
            print(f"  ✓ Best parameters: {optimization_result['best_parameters']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return optimization_results

def demo_validation_system():
    """Demonstrate the validation system."""
    
    print("\n" + "="*80)
    print("DEMONSTRATING VALIDATION SYSTEM")
    print("="*80)
    
    # Create validation system
    validation_system = BaselineValidationSystem()
    
    # Energy range
    energy = np.linspace(0, 3500, 1000)
    
    # Test parameters
    peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
    peak_widths = [30, 25, 35, 40, 30, 35, 25]
    
    baseline_types = ['polynomial', 'exponential', 'linear']
    noise_levels = [0.01, 0.05, 0.1]
    
    # Test detectors
    test_detectors = ['pybaselines_asls', 'pybaselines_airpls', 'spectrochempy_polynomial']
    detectors = [EnhancedBaselineDetectorFactory.create_detector(dt) for dt in test_detectors]
    
    # Run validation
    validation_results = validation_system.compare_detectors(
        detectors, energy, peak_positions, peak_amplitudes, peak_widths,
        baseline_types, noise_levels
    )
    
    return validation_results

def demo_enhanced_ml_analyzer():
    """Demonstrate the enhanced ML analyzer integration."""
    
    print("\n" + "="*80)
    print("DEMONSTRATING ENHANCED ML ANALYZER INTEGRATION")
    print("="*80)
    
    # Create test spectrum and save it temporarily
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_realistic_test_spectrum()
    
    # Create temporary file
    temp_file = "temp_test_spectrum.csv"
    test_data = pd.DataFrame({
        'x': energy,
        'y': spectrum
    })
    test_data.to_csv(temp_file, index=False)
    
    try:
        # Test different baseline detectors
        detectors_to_test = ['pybaselines_asls', 'pybaselines_airpls', 'spectrochempy_polynomial']
        
        results = []
        
        for detector_type in detectors_to_test:
            print(f"\nTesting Enhanced ML Analyzer with {detector_type}...")
            
            try:
                # Create validation data
                validation_data = {
                    'intensity': spectrum,
                    'energy': energy,
                    'true_baseline': true_baseline,
                    'peak_positions': np.array(peak_positions, dtype=int)
                }
                
                # Analyze with enhanced baseline detection
                analyzer = analyze_spectrum_with_enhanced_baseline(
                    filepath=temp_file,
                    baseline_detector_type=detector_type,
                    enable_parameter_optimization=True,
                    validation_data=validation_data,
                    distance=3,
                    prominence=0.01,
                    width=1
                )
                
                # Extract features
                features = analyzer.extract_enhanced_ml_features()
                
                result = {
                    'detector': detector_type,
                    'num_peaks': features['num_peaks'],
                    'baseline_rmse': features.get('baseline_rmse', 0),
                    'baseline_correlation': features.get('baseline_correlation', 0),
                    'fit_r_squared': features.get('fit_r_squared', 0),
                    'baseline_processing_time': features.get('baseline_processing_time', 0)
                }
                results.append(result)
                
                print(f"  ✓ Analysis completed:")
                print(f"    - Peaks detected: {features['num_peaks']}")
                print(f"    - Baseline RMSE: {features.get('baseline_rmse', 0):.6f}")
                print(f"    - Fit R²: {features.get('fit_r_squared', 0):.4f}")
                
                # Generate report
                report_path = f"enhanced_analysis_report_{detector_type.replace('_', '-')}.md"
                analyzer.generate_enhanced_report(save_path=report_path)
                
                # Create visualization
                plot_path = f"enhanced_analysis_plot_{detector_type.replace('_', '-')}.pdf"
                analyzer.plot_enhanced_analysis(save_path=plot_path)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'detector': detector_type,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_comprehensive_visualization(energy, spectrum, true_baseline, results_df):
    """Create comprehensive visualization of all results."""
    
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("="*80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Enhanced Baseline Detection System - Comprehensive Demo Results', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Original spectrum with true baseline
    ax1 = axes[0, 0]
    ax1.plot(energy, spectrum, 'k-', alpha=0.7, label='Original Spectrum', linewidth=1)
    ax1.plot(energy, true_baseline, 'g-', linewidth=2, label='True Baseline')
    ax1.set_xlabel('Energy (cm⁻¹)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title('Original Spectrum with True Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Baseline detection comparison
    ax2 = axes[0, 1]
    ax2.plot(energy, true_baseline, 'g-', linewidth=2, label='True Baseline')
    
    # Plot detected baselines for successful detectors
    successful_results = results_df[~results_df.get('error', False).fillna(False)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_results)))
    for i, (_, result) in enumerate(successful_results.iterrows()):
        detector_type = result['detector']
        try:
            detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
            detected_baseline = detector.detect_baseline(spectrum, energy)
            ax2.plot(energy, detected_baseline, '--', color=colors[i], 
                    linewidth=1.5, label=f"{detector_type} (RMSE: {result['rmse']:.4f})")
        except:
            continue
    
    ax2.set_xlabel('Energy (cm⁻¹)')
    ax2.set_ylabel('Baseline (a.u.)')
    ax2.set_title('Baseline Detection Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quality metrics comparison
    ax3 = axes[1, 0]
    if len(successful_results) > 0:
        metrics_to_plot = ['rmse', 'correlation', 'smoothness']
        x = np.arange(len(successful_results))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = successful_results[metric].values
            ax3.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax3.set_xlabel('Detector')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Quality Metrics Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(successful_results['detector'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Processing time comparison
    ax4 = axes[1, 1]
    if len(successful_results) > 0:
        times = successful_results['processing_time'].values
        detectors = successful_results['detector'].values
        
        bars = ax4.bar(range(len(detectors)), times, color='skyblue', alpha=0.7)
        ax4.set_xlabel('Detector')
        ax4.set_ylabel('Processing Time (s)')
        ax4.set_title('Processing Time Comparison')
        ax4.set_xticks(range(len(detectors)))
        ax4.set_xticklabels(detectors, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Plot 5: Quality score comparison
    ax5 = axes[2, 0]
    if len(successful_results) > 0:
        quality_scores = successful_results['quality_score'].values
        detectors = successful_results['detector'].values
        
        bars = ax5.bar(range(len(detectors)), quality_scores, color='lightgreen', alpha=0.7)
        ax5.set_xlabel('Detector')
        ax5.set_ylabel('Quality Score (Lower is Better)')
        ax5.set_title('Overall Quality Score Comparison')
        ax5.set_xticks(range(len(detectors)))
        ax5.set_xticklabels(detectors, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.6f}', ha='center', va='bottom')
    
    # Plot 6: Peak preservation comparison
    ax6 = axes[2, 1]
    if len(successful_results) > 0:
        peak_preservation = successful_results['peak_preservation'].values
        detectors = successful_results['detector'].values
        
        bars = ax6.bar(range(len(detectors)), peak_preservation, color='orange', alpha=0.7)
        ax6.set_xlabel('Detector')
        ax6.set_ylabel('Peak Preservation')
        ax6.set_title('Peak Preservation Comparison')
        ax6.set_xticks(range(len(detectors)))
        ax6.set_xticklabels(detectors, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, preservation in zip(bars, peak_preservation):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{preservation:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_baseline_detection_comprehensive_demo.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comprehensive visualization saved: enhanced_baseline_detection_comprehensive_demo.pdf")

def generate_demo_report(results_df, optimization_results, validation_results, ml_results):
    """Generate a comprehensive demo report."""
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE DEMO REPORT")
    print("="*80)
    
    # Create report
    report = []
    report.append("# Enhanced Baseline Detection System - Comprehensive Demo Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("This demo showcases the enhanced baseline detection system with:")
    report.append("- Multiple baseline algorithms from pybaselines and spectrochempy")
    report.append("- Systematic parameter optimization with validation")
    report.append("- Comprehensive quality metrics and comparison")
    report.append("- Integration with the existing ML analysis pipeline")
    report.append("- Detailed reporting and visualization capabilities")
    report.append("")
    
    # Overall statistics
    successful_detectors = results_df[~results_df.get('error', False).fillna(False)]
    report.append(f"## Detector Performance Summary")
    report.append(f"- Total detectors tested: {len(results_df)}")
    report.append(f"- Successful detectors: {len(successful_detectors)}")
    report.append(f"- Failed detectors: {len(results_df) - len(successful_detectors)}")
    report.append("")
    
    if len(successful_detectors) > 0:
        # Best performing detector
        best_detector = successful_detectors.loc[successful_detectors['quality_score'].idxmin()]
        report.append(f"### Best Overall Detector")
        report.append(f"- **{best_detector['detector']}**")
        report.append(f"  - Quality score: {best_detector['quality_score']:.6f}")
        report.append(f"  - RMSE: {best_detector['rmse']:.6f}")
        report.append(f"  - Correlation: {best_detector['correlation']:.4f}")
        report.append(f"  - Processing time: {best_detector['processing_time']:.3f}s")
        report.append("")
        
        # Fastest detector
        fastest_detector = successful_detectors.loc[successful_detectors['processing_time'].idxmin()]
        report.append(f"### Fastest Detector")
        report.append(f"- **{fastest_detector['detector']}**")
        report.append(f"  - Processing time: {fastest_detector['processing_time']:.3f}s")
        report.append(f"  - Quality score: {fastest_detector['quality_score']:.6f}")
        report.append("")
        
        # Most accurate detector
        most_accurate = successful_detectors.loc[successful_detectors['rmse'].idxmin()]
        report.append(f"### Most Accurate Detector")
        report.append(f"- **{most_accurate['detector']}**")
        report.append(f"  - RMSE: {most_accurate['rmse']:.6f}")
        report.append(f"  - Quality score: {most_accurate['quality_score']:.6f}")
        report.append("")
    
    # Detailed results table
    report.append("## Detailed Detector Results")
    report.append("")
    report.append("| Detector | RMSE | Correlation | Smoothness | Processing Time (s) | Quality Score | Status |")
    report.append("|----------|------|-------------|------------|-------------------|---------------|--------|")
    
    for _, result in results_df.iterrows():
        if 'error' in result and result['error']:
            report.append(f"| {result['detector']} | - | - | - | - | - | Failed: {result['error']} |")
        else:
            report.append(f"| {result['detector']} | {result['rmse']:.6f} | {result['correlation']:.4f} | "
                         f"{result['smoothness']:.6f} | {result['processing_time']:.3f} | {result['quality_score']:.6f} | Success |")
    
    report.append("")
    
    # Parameter optimization results
    if optimization_results:
        report.append("## Parameter Optimization Results")
        report.append("")
        report.append("| Detector | Best Score | Best Parameters | Optimized RMSE | Optimized Correlation |")
        report.append("|----------|------------|-----------------|----------------|---------------------|")
        
        for result in optimization_results:
            params_str = ", ".join([f"{k}={v:.4f}" for k, v in result['best_parameters'].items()])
            report.append(f"| {result['detector']} | {result['best_score']:.6f} | {params_str} | "
                         f"{result['optimized_rmse']:.6f} | {result['optimized_correlation']:.4f} |")
        
        report.append("")
    
    # ML integration results
    if ml_results is not None and len(ml_results) > 0:
        report.append("## ML Integration Results")
        report.append("")
        report.append("| Detector | Peaks Detected | Baseline RMSE | Baseline Correlation | Fit R² | Processing Time (s) |")
        report.append("|----------|---------------|---------------|---------------------|--------|-------------------|")
        
        for _, result in ml_results.iterrows():
            if 'error' in result and result['error']:
                report.append(f"| {result['detector']} | - | - | - | - | - | Failed: {result['error']} |")
            else:
                report.append(f"| {result['detector']} | {result['num_peaks']} | {result['baseline_rmse']:.6f} | "
                             f"{result['baseline_correlation']:.4f} | {result['fit_r_squared']:.4f} | {result['baseline_processing_time']:.3f} |")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if len(successful_detectors) > 0:
        report.append("### For High Accuracy Applications:")
        best_accuracy = successful_detectors.loc[successful_detectors['rmse'].idxmin()]
        report.append(f"- Use **{best_accuracy['detector']}** (RMSE: {best_accuracy['rmse']:.6f})")
        report.append("")
        
        report.append("### For Fast Processing:")
        fastest = successful_detectors.loc[successful_detectors['processing_time'].idxmin()]
        report.append(f"- Use **{fastest['detector']}** (Time: {fastest['processing_time']:.3f}s)")
        report.append("")
        
        report.append("### For Balanced Performance:")
        best_balanced = successful_detectors.loc[successful_detectors['quality_score'].idxmin()]
        report.append(f"- Use **{best_balanced['detector']}** (Quality score: {best_balanced['quality_score']:.6f})")
        report.append("")
        
        report.append("### For ML Integration:")
        if ml_results is not None and len(ml_results) > 0:
            successful_ml = ml_results[~ml_results.get('error', False).fillna(False)]
            if len(successful_ml) > 0:
                best_ml = successful_ml.loc[successful_ml['baseline_rmse'].idxmin()]
                report.append(f"- Use **{best_ml['detector']}** for ML analysis (Baseline RMSE: {best_ml['baseline_rmse']:.6f})")
        report.append("")
    
    # System capabilities
    report.append("## System Capabilities")
    report.append("")
    report.append("### Available Detectors:")
    available_detectors = EnhancedBaselineDetectorFactory.get_available_detectors()
    for detector in available_detectors:
        report.append(f"- {detector}")
    report.append("")
    
    report.append("### Quality Metrics:")
    report.append("- **RMSE**: Root Mean Square Error between true and estimated baseline")
    report.append("- **Correlation**: Pearson correlation coefficient")
    report.append("- **Smoothness**: Average absolute difference between consecutive baseline points")
    report.append("- **Peak Preservation**: How well peaks are preserved after baseline correction")
    report.append("- **Quality Score**: Combined metric (lower is better)")
    report.append("")
    
    report.append("### Parameter Optimization:")
    report.append("- Systematic parameter search with validation")
    report.append("- Quality-based optimization using simulated data")
    report.append("- Comprehensive parameter ranges for each algorithm")
    report.append("")
    
    # Save report
    with open('enhanced_baseline_detection_demo_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✓ Comprehensive demo report saved: enhanced_baseline_detection_demo_report.md")

def main():
    """Main function to run the comprehensive demo."""
    
    print("Enhanced Baseline Detection System - Comprehensive Demo")
    print("="*80)
    
    # Demo 1: Available detectors
    results_df = demo_available_detectors()
    
    # Demo 2: Parameter optimization
    optimization_results = demo_parameter_optimization()
    
    # Demo 3: Validation system
    validation_results = demo_validation_system()
    
    # Demo 4: Enhanced ML analyzer integration
    ml_results = demo_enhanced_ml_analyzer()
    
    # Create comprehensive visualization
    energy, spectrum, true_baseline, _, _, _ = create_realistic_test_spectrum()
    create_comprehensive_visualization(energy, spectrum, true_baseline, results_df)
    
    # Generate comprehensive report
    generate_demo_report(results_df, optimization_results, validation_results, ml_results)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("- enhanced_baseline_detection_comprehensive_demo.pdf")
    print("- enhanced_baseline_detection_demo_report.md")
    print("- enhanced_analysis_report_*.md (for each detector)")
    print("- enhanced_analysis_plot_*.pdf (for each detector)")
    print("\nCheck the demo report for detailed results and recommendations.")
    print("\nThe enhanced baseline detection system is now ready for use!")

if __name__ == "__main__":
    main() 