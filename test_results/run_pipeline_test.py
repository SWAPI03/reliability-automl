import sys, warnings
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
from config import PipelineConfig

STAGES = [
    (vrun,  'Validate'),
    (prun,  'Preprocess'),
    (srun,  'Structural'),
    (trun,  'Statistical'),
    (parun, 'Pattern'),
    (drun,  'Deduction'),
    (rfrun, 'Fusion'),
    (rgrun, 'Graph'),
    (tprun, 'TrustProp'),
    (arun,  'AutoML'),
    (frrun, 'Feedback'),
    (ftrun, 'FeatureTrust'),
    (msrun, 'ModelSelect'),
    (nsrun, 'Noise'),
]

def run_pipeline(csv_path, label):
    print('\n=== ' + label + ' ===')
    with open(csv_path, 'rb') as f:
        data = f.read()
    state = PipelineState()
    config = PipelineConfig()
    errors = []
    for fn, name in STAGES:
        try:
            if name == 'Validate':
                state = fn(state, config, file=data)
            else:
                state = fn(state, config)
            print('  ' + name + ': OK')
        except Exception as e:
            print('  ' + name + ': ERROR - ' + str(e))
            errors.append(name)
    print('  Scores: S={:.3f} T={:.3f} Sim={:.3f} C={:.3f} D={:.3f} R={:.3f}'.format(
        state.structural_score or 0, state.statistical_score or 0,
        state.similarity_score or 0, state.conflict_score or 0,
        state.deduction_score or 0, state.reliability_score or 0))
    if state.model_scores:
        for m, sc in state.model_scores.items():
            print('  ' + m + ': cv={:.4f} rel={:.4f}'.format(
                sc.get('cv_score', 0), sc.get('reliability_score', 0)))
    if state.best_model_name:
        print('  Best model: ' + state.best_model_name)
    if state.noisy_model_scores:
        print('  Noisy scores:')
        for m, sc in state.noisy_model_scores.items():
            print('    ' + m + ': cv={:.4f} rel={:.4f}'.format(
                sc.get('cv_score', 0), sc.get('reliability_score', 0)))
    if state.confidence_score is not None:
        print('  Confidence score: {:.4f}'.format(state.confidence_score))
    if state.feature_trust:
        avg_ft = sum(state.feature_trust.values()) / len(state.feature_trust)
        print('  Avg feature trust: {:.4f}'.format(avg_ft))
    print('  Errors: ' + (', '.join(errors) if errors else 'None'))
    return state

s1 = run_pipeline('large_dataset_500.csv',  'Dataset 1 - 525 rows 10 numeric features 15pct noise')
s2 = run_pipeline('large_dataset_1200.csv', 'Dataset 2 - 1200 rows 15 numeric features 20pct noise')
s3 = run_pipeline('mixed_dataset_300.csv',  'Dataset 3 - 300 rows mixed categorical+numeric 10pct noise')
print('\nDone.')