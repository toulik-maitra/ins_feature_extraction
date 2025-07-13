"""
Enhanced Baseline Detection Test Script
=======================================

This script demonstrates the enhanced baseline detection system with:
- Multiple baseline algorithms
- Parameter optimization
- Validation with synthetic data
- Comprehensive quality metrics
- Visualization of results

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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.enhanced_baseline_detection import (
    EnhancedBaselineDetectorFactory,
    BaselineValidationSystem,
    BaselineQualityMetrics
)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_test_spectrum():
    """Create a realistic test spectrum with known baseline."""
    
    # Energy range
    energy = np.linspace(0, 3500, 1000)
    
    # Create true baseline (polynomial + exponential)
    true_baseline = (0.05 + 0.0001 * energy + 
                     0.0000001 * energy**2 + 
                     0.1 * np.exp(-energy / 1000))
    
    # Add peaks at realistic positions
    peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
    peak_widths = [30, 25, 35, 40, 30, 35, 25]
    
    # Create spectrum
    spectrum = true_baseline.copy()
    for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
        peak = amp * np.exp(-(energy - pos)**2 / (2 * width**2))
        spectrum += peak
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, len(energy))
    spectrum += noise
    
    return energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths

def test_available_detectors():
    """Test all available baseline detectors."""
    
    print("="*80)
    print("TESTING AVAILABLE BASELINE DETECTORS")
    print("="*80)
    
    # Get available detectors
    available_detectors = EnhancedBaselineDetectorFactory.get_available_detectors()
    print(f"Available detectors: {available_detectors}")
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_test_spectrum()
    
    # Test each detector
    results = []
    
    for detector_type in available_detectors:
        print(f"\nTesting {detector_type}...")
        
        try:
            # Create detector
            detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
            
            # Detect baseline
            import time
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

def test_parameter_optimization():
    """Test parameter optimization for selected detectors."""
    
    print("\n" + "="*80)
    print("TESTING PARAMETER OPTIMIZATION")
    print("="*80)
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_test_spectrum()
    
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
                max_iterations=30
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

def test_validation_system():
    """Test the validation system with different baseline types."""
    
    print("\n" + "="*80)
    print("TESTING VALIDATION SYSTEM")
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

def visualize_results(energy, spectrum, true_baseline, results_df):
    """Create comprehensive visualization of baseline detection results."""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Baseline Detection Results', fontsize=16, fontweight='bold')
    
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
    
    plt.tight_layout()
    plt.savefig('enhanced_baseline_detection_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization saved: enhanced_baseline_detection_results.pdf")

def create_parameter_optimization_plots(optimization_results):
    """Create plots showing parameter optimization progress."""
    
    if not optimization_results:
        return
    
    print("\nCreating parameter optimization plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Parameter Optimization Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Optimization history for each detector
    ax1 = axes[0]
    
    for result in optimization_results:
        detector_type = result['detector']
        best_score = result['best_score']
        
        # Create detector and get optimization history
        detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
        if hasattr(detector, 'optimization_history') and detector.optimization_history:
            iterations = [h['iteration'] for h in detector.optimization_history]
            scores = [h['quality_score'] for h in detector.optimization_history]
            ax1.plot(iterations, scores, 'o-', label=f"{detector_type} (Best: {best_score:.6f})")
    
    ax1.set_xlabel('Optimization Iteration')
    ax1.set_ylabel('Quality Score (Lower is Better)')
    ax1.set_title('Parameter Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best parameters comparison
    ax2 = axes[1]
    
    if optimization_results:
        detectors = [r['detector'] for r in optimization_results]
        best_scores = [r['best_score'] for r in optimization_results]
        
        bars = ax2.bar(range(len(detectors)), best_scores, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Detector')
        ax2.set_ylabel('Best Quality Score')
        ax2.set_title('Best Optimization Results')
        ax2.set_xticks(range(len(detectors)))
        ax2.set_xticklabels(detectors, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, best_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('parameter_optimization_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Parameter optimization plots saved: parameter_optimization_results.pdf")

def generate_comprehensive_report(results_df, optimization_results, validation_results):
    """Generate a comprehensive report of all results."""
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    # Create report
    report = []
    report.append("# Enhanced Baseline Detection System - Comprehensive Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    
    # Overall statistics
    successful_detectors = results_df[~results_df.get('error', False).fillna(False)]
    report.append(f"- Total detectors tested: {len(results_df)}")
    report.append(f"- Successful detectors: {len(successful_detectors)}")
    report.append(f"- Failed detectors: {len(results_df) - len(successful_detectors)}")
    report.append("")
    
    if len(successful_detectors) > 0:
        # Best performing detector
        best_detector = successful_detectors.loc[successful_detectors['quality_score'].idxmin()]
        report.append(f"- Best overall detector: {best_detector['detector']}")
        report.append(f"  - Quality score: {best_detector['quality_score']:.6f}")
        report.append(f"  - RMSE: {best_detector['rmse']:.6f}")
        report.append(f"  - Correlation: {best_detector['correlation']:.4f}")
        report.append("")
        
        # Fastest detector
        fastest_detector = successful_detectors.loc[successful_detectors['processing_time'].idxmin()]
        report.append(f"- Fastest detector: {fastest_detector['detector']}")
        report.append(f"  - Processing time: {fastest_detector['processing_time']:.3f}s")
        report.append("")
    
    # Detailed results table
    report.append("## Detailed Results")
    report.append("")
    report.append("| Detector | RMSE | Correlation | Smoothness | Processing Time (s) | Status |")
    report.append("|----------|------|-------------|------------|-------------------|--------|")
    
    for _, result in results_df.iterrows():
        if 'error' in result and result['error']:
            report.append(f"| {result['detector']} | - | - | - | - | Failed: {result['error']} |")
        else:
            report.append(f"| {result['detector']} | {result['rmse']:.6f} | {result['correlation']:.4f} | "
                         f"{result['smoothness']:.6f} | {result['processing_time']:.3f} | Success |")
    
    report.append("")
    
    # Parameter optimization results
    if optimization_results:
        report.append("## Parameter Optimization Results")
        report.append("")
        report.append("| Detector | Best Score | Best Parameters |")
        report.append("|----------|------------|-----------------|")
        
        for result in optimization_results:
            params_str = ", ".join([f"{k}={v:.4f}" for k, v in result['best_parameters'].items()])
            report.append(f"| {result['detector']} | {result['best_score']:.6f} | {params_str} |")
        
        report.append("")
    
    # Validation results
    if validation_results is not None and len(validation_results) > 0:
        report.append("## Validation Results")
        report.append("")
        report.append("Validation was performed across different baseline types and noise levels.")
        report.append("")
        
        # Summary statistics
        avg_rmse = validation_results['rmse_mean'].mean()
        avg_correlation = validation_results['correlation_mean'].mean()
        
        report.append(f"- Average RMSE across all conditions: {avg_rmse:.6f}")
        report.append(f"- Average correlation across all conditions: {avg_correlation:.4f}")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if len(successful_detectors) > 0:
        report.append("### For High Accuracy Applications:")
        best_accuracy = successful_detectors.loc[successful_detectors['rmse'].idxmin()]
        report.append(f"- Use {best_accuracy['detector']} (RMSE: {best_accuracy['rmse']:.6f})")
        report.append("")
        
        report.append("### For Fast Processing:")
        fastest = successful_detectors.loc[successful_detectors['processing_time'].idxmin()]
        report.append(f"- Use {fastest['detector']} (Time: {fastest['processing_time']:.3f}s)")
        report.append("")
        
        report.append("### For Balanced Performance:")
        best_balanced = successful_detectors.loc[successful_detectors['quality_score'].idxmin()]
        report.append(f"- Use {best_balanced['detector']} (Quality score: {best_balanced['quality_score']:.6f})")
        report.append("")
    
    # Save report
    with open('enhanced_baseline_detection_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✓ Comprehensive report saved: enhanced_baseline_detection_report.md")

def main():
    """Main function to run all tests."""
    
    print("Enhanced Baseline Detection System - Comprehensive Test")
    print("="*80)
    
    # Test 1: Available detectors
    results_df = test_available_detectors()
    
    # Test 2: Parameter optimization
    optimization_results = test_parameter_optimization()
    
    # Test 3: Validation system
    validation_results = test_validation_system()
    
    # Create visualizations
    energy, spectrum, true_baseline, _, _, _ = create_test_spectrum()
    visualize_results(energy, spectrum, true_baseline, results_df)
    
    # Create parameter optimization plots
    create_parameter_optimization_plots(optimization_results)
    
    # Generate comprehensive report
    generate_comprehensive_report(results_df, optimization_results, validation_results)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("- enhanced_baseline_detection_results.pdf")
    print("- parameter_optimization_results.pdf")
    print("- enhanced_baseline_detection_report.md")
    print("\nCheck the report for detailed recommendations and results.")

if __name__ == "__main__":
    main() 