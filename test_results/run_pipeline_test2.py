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

STAGES = [(vrun,'Validate'),(prun,'Preprocess'),(srun,'Structural'),
          (trun,'Statistical'),(parun,'Pattern'),(drun,'Deduction'),
          (rfrun,'Fusion'),(rgrun,'Graph'),(tprun,'TrustProp'),
          (arun,'AutoML'),(frrun,'Feedback'),(ftrun,'FeatureTrust'),
          (msrun,'ModelSelect'),(nsrun,'Noise')]

def run_pipeline(csv_path, label):
    with open(csv_path, 'rb') as f:
        data = f.read()
    state = PipelineState()
    config = PipelineConfig()
    for fn, name in STAGES:
        try:
            if name == 'Validate':
                state = fn(state, config, file=data)
            else:
                state = fn(state, config)
        except Exception as e:
            print('ERROR ' + name + ': ' + str(e))
    R = state.reliability_score or 0
    print('\n--- ' + label + ' ---')
    print('Rows: ' + str(len(state.raw_df)) + '  Cols: ' + str(len(state.raw_df.columns)))
    print('S={:.3f}  T={:.3f}  Sim={:.3f}  C={:.3f}  D={:.3f}  R={:.3f}'.format(
        state.structural_score or 0, state.statistical_score or 0,
        state.similarity_score or 0, state.conflict_score or 0,
        state.deduction_score or 0, R))
    print('Confidence={:.4f}  AvgFeatureTrust={:.4f}'.format(
        state.confidence_score or 0,
        sum(state.feature_trust.values())/len(state.feature_trust) if state.feature_trust else 0))
    print('Task: ' + str(state.task_type))
    if state.model_scores:
        print('Model scores (cv_score x R = reliability_adjusted):')
        for m, sc in state.model_scores.items():
            cv = sc.get('cv_score', 0)
            rel_adj = cv * R
            print('  ' + m + ': cv={:.4f}  rel_adj={:.4f}'.format(cv, rel_adj))
    print('Best model: ' + str(state.best_model_name))
    if state.noisy_model_scores:
        print('Under noise (10% Gaussian):')
        for m, sc in state.noisy_model_scores.items():
            cv = sc.get('cv_score', 0)
            rel_adj = cv * R
            print('  ' + m + ': cv={:.4f}  rel_adj={:.4f}'.format(cv, rel_adj))
    if state.high_importance_low_trust_features:
        print('Risky features: ' + str(state.high_importance_low_trust_features))
    else:
        print('Risky features: None flagged')
    return state

s1 = run_pipeline('large_dataset_500.csv',  'Dataset 1: 525 rows, 10 numeric, 15pct label noise, 8pct missing')
s2 = run_pipeline('large_dataset_1200.csv', 'Dataset 2: 1200 rows, 15 numeric, 20pct label noise, 10pct missing')
s3 = run_pipeline('mixed_dataset_300.csv',  'Dataset 3: 300 rows, mixed categorical+numeric, 10pct noise')