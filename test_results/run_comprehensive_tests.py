"""Comprehensive testing on large datasets with detailed metrics collection."""
import sys
import warnings
import time
import json
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'reliability_automl')

from pipeline.state import PipelineState
from pipeline.validator import run as vrun
from pipeline.preprocessor import run as prun
from pipeline.structural_score import run as srun
from pipeline.statistical_score import run as trun
from pipeline.pattern_analysis import run as parun
from pipeline.deduction_score import run as drun
from pipeline.reliability_fusion import run as rfrun
from pipeline.reliability_graph import run as rgrun
from pipeline.trust_propagation import run as tprun
from pipeline.automl_trainer import run as arun
from pipeline.feedback_refinement import run as frrun
from pipeline.feature_trust import run as ftrun
from pipeline.model_selection import run as msrun
from pipeline.noise_simulation import run as nsrun
from pipeline.baseline_comparison import run as bcrun
from config import PipelineConfig

STAGES = [
    (vrun, 'Validate'),
    (prun, 'Preprocess'),
    (srun, 'Structural'),
    (trun, 'Statistical'),
    (parun, 'Pattern'),
    (drun, 'Deduction'),
    (rfrun, 'Fusion'),
    (rgrun, 'Graph'),
    (tprun, 'TrustProp'),
    (arun, 'AutoML'),
    (frrun, 'Feedback'),
    (ftrun, 'FeatureTrust'),
    (msrun, 'ModelSelect'),
    (nsrun, 'Noise'),
    # Skip baseline comparison for speed
    # (bcrun, 'Baseline'),
]

def run_pipeline(csv_path, label):
    """Run full pipeline and collect comprehensive metrics."""
    print('\n' + '='*80)
    print(f'  {label}')
    print('='*80)
    
    start_time = time.time()
    
    with open(csv_path, 'rb') as f:
        data = f.read()
    
    state = PipelineState()
    config = PipelineConfig()
    errors = []
    stage_times = {}
    
    for fn, name in STAGES:
        stage_start = time.time()
        try:
            if name == 'Validate':
                state = fn(state, config, file=data)
            else:
                state = fn(state, config)
            stage_time = time.time() - stage_start
            stage_times[name] = stage_time
            print(f'  [OK] {name:<15} ({stage_time:.2f}s)')
        except Exception as e:
            stage_time = time.time() - stage_start
            stage_times[name] = stage_time
            print(f'  [ERROR] {name:<15} ERROR: {str(e)[:60]}')
            errors.append(name)
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {
        'dataset': csv_path,
        'label': label,
        'total_time': total_time,
        'stage_times': stage_times,
        'errors': errors,
        'data_shape': {
            'rows': len(state.raw_df) if state.raw_df is not None else 0,
            'cols': len(state.raw_df.columns) if state.raw_df is not None else 0,
        },
        'scores': {
            'structural': float(state.structural_score or 0),
            'statistical': float(state.statistical_score or 0),
            'similarity': float(state.similarity_score or 0),
            'conflict': float(state.conflict_score or 0),
            'deduction': float(state.deduction_score or 0),
            'reliability': float(state.reliability_score or 0),
            'confidence': float(state.confidence_score or 0),
        },
        'task_type': state.task_type,
        'best_model': state.best_model_name,
        'model_scores': {},
        'noisy_model_scores': {},
        'baseline_comparison': {},
        'feature_trust': {},
        'risky_features': state.high_importance_low_trust_features or [],
    }
    
    # Model scores
    if state.model_scores:
        for model_name, scores in state.model_scores.items():
            results['model_scores'][model_name] = {
                'cv_score': float(scores.get('cv_score', 0)),
                'reliability_score': float(scores.get('reliability_score', 0)),
                'reliability_adjusted': float(scores.get('reliability_adjusted_score', 0)),
            }
    
    # Noisy model scores
    if state.noisy_model_scores:
        for model_name, scores in state.noisy_model_scores.items():
            results['noisy_model_scores'][model_name] = {
                'cv_score': float(scores.get('cv_score', 0)),
                'reliability_score': float(scores.get('reliability_score', 0)),
            }
    
    # Baseline comparison
    if state.baseline_scores:
        try:
            results['baseline_comparison'] = {
                'before': {k: float(v) if not isinstance(v, dict) else v.get('cv_score', 0) 
                          for k, v in state.baseline_scores.items()},
                'after': {k: float(v) if not isinstance(v, dict) else v.get('cv_score', 0) 
                         for k, v in (state.model_scores or {}).items() if k in state.baseline_scores},
            }
        except:
            results['baseline_comparison'] = {}
    
    # Feature trust
    if state.feature_trust:
        results['feature_trust'] = {
            'avg_trust': float(sum(state.feature_trust.values()) / len(state.feature_trust)),
            'min_trust': float(min(state.feature_trust.values())),
            'max_trust': float(max(state.feature_trust.values())),
            'features': {k: float(v) for k, v in state.feature_trust.items()},
        }
    
    # Print summary
    print('\n' + '-'*80)
    print('RESULTS SUMMARY')
    print('-'*80)
    print(f"Dataset: {csv_path}")
    print(f"Shape: {results['data_shape']['rows']} rows × {results['data_shape']['cols']} columns")
    print(f"Total Time: {total_time:.2f}s")
    print(f"\nQuality Scores:")
    print(f"  Structural (S):   {results['scores']['structural']:.4f}")
    print(f"  Statistical (T):  {results['scores']['statistical']:.4f}")
    print(f"  Similarity (Sim): {results['scores']['similarity']:.4f}")
    print(f"  Conflict (C):     {results['scores']['conflict']:.4f}")
    print(f"  Deduction (D):    {results['scores']['deduction']:.4f}")
    print(f"  Reliability (R):  {results['scores']['reliability']:.4f}")
    print(f"  Confidence:       {results['scores']['confidence']:.4f}")
    
    print(f"\nTask Type: {results['task_type']}")
    print(f"Best Model: {results['best_model']}")
    
    if results['model_scores']:
        print(f"\nModel Performance (Clean Data):")
        for model_name, scores in results['model_scores'].items():
            marker = " ← BEST" if model_name == results['best_model'] else ""
            print(f"  {model_name}:")
            print(f"    CV Accuracy:          {scores['cv_score']:.4f}")
            print(f"    Reliability-Adjusted: {scores['reliability_adjusted']:.4f}{marker}")
    
    if results['noisy_model_scores']:
        print(f"\nModel Performance (Under 10% Gaussian Noise):")
        for model_name, scores in results['noisy_model_scores'].items():
            print(f"  {model_name}:")
            print(f"    CV Accuracy:     {scores['cv_score']:.4f}")
            print(f"    Reliability:     {scores['reliability_score']:.4f}")
    
    if results['feature_trust']:
        print(f"\nFeature Trust:")
        print(f"  Average: {results['feature_trust']['avg_trust']:.4f}")
        print(f"  Range:   [{results['feature_trust']['min_trust']:.4f}, {results['feature_trust']['max_trust']:.4f}]")
    
    if results['risky_features']:
        print(f"\nRisky Features (High SHAP, Low Trust):")
        for feat in results['risky_features']:
            print(f"  - {feat}")
    else:
        print(f"\nRisky Features: None detected")
    
    if errors:
        print(f"\n[WARNING] Errors in stages: {', '.join(errors)}")
    else:
        print(f"\n[OK] All stages completed successfully")
    
    return results

def main():
    """Run comprehensive tests on all large datasets."""
    datasets = [
        ('large_dataset_10000.csv', 'Dataset 1: 10K rows, 20 features, 15% noise'),
        ('large_dataset_15000_noisy.csv', 'Dataset 2: 15K rows, 25 features, 25% noise (high)'),
        ('large_dataset_20000_clean.csv', 'Dataset 3: 20K rows, 15 features, 5% noise (clean)'),
        ('large_dataset_12000_extreme.csv', 'Dataset 4: 12K rows, 30 features, 35% noise (extreme)'),
    ]
    
    all_results = []
    
    for csv_path, label in datasets:
        if not Path(csv_path).exists():
            print(f"\n[WARNING] Skipping {csv_path} - file not found")
            continue
        
        try:
            results = run_pipeline(csv_path, label)
            all_results.append(results)
            
            # Save after each dataset to avoid data loss
            with open('comprehensive_test_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[OK] Results saved (total: {len(all_results)} datasets)")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process {csv_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final save
    output_file = 'comprehensive_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print('\n' + '='*80)
    print('COMPREHENSIVE TEST SUMMARY')
    print('='*80)
    print(f"Total datasets tested: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    
    # Print comparison table
    if all_results:
        print("\nComparative Results:")
        print(f"{'Dataset':<50} {'Rows':<8} {'R Score':<10} {'Best Model':<15} {'Time':<8}")
        print('-'*100)
        for r in all_results:
            print(f"{r['label']:<50} {r['data_shape']['rows']:<8} "
                  f"{r['scores']['reliability']:<10.4f} {r['best_model']:<15} "
                  f"{r['total_time']:<8.1f}s")
    
    print('\n✓ Comprehensive testing complete!')

if __name__ == "__main__":
    main()
