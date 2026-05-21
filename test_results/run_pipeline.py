"""Run the full pipeline from the command line without Streamlit."""
from __future__ import annotations

import argparse
import json
import joblib
import numpy as np

from reliability_automl.config import PipelineConfig
from reliability_automl.pipeline.state import PipelineState
from reliability_automl.pipeline import (
    validator, preprocessor, structural_score, statistical_score,
    pattern_analysis, deduction_score, reliability_fusion, reliability_graph,
    trust_propagation, automl_trainer, feedback_refinement, feature_trust,
    model_selection, noise_simulation, explainability, baseline_comparison,
)


def run(csv_path: str, output_model: str = "best_model.joblib",
        output_report: str = "report.json") -> None:
    config = PipelineConfig()
    state = PipelineState()

    print(f"[1/15] Loading {csv_path}...")
    with open(csv_path, "rb") as f:
        state = validator.run(state, config, file=f)

    print("[2/15] Preprocessing...")
    state = preprocessor.run(state, config)

    print("[3/15] Structural score...")
    state = structural_score.run(state, config)

    print("[4/15] Statistical score...")
    state = statistical_score.run(state, config)

    print("[5/15] Pattern analysis...")
    state = pattern_analysis.run(state, config)

    print("[6/15] Deduction score (pseudo-source modeling)...")
    state = deduction_score.run(state, config)

    print("[7/15] Reliability fusion...")
    state = reliability_fusion.run(state, config)

    print("[8/15] Reliability graph...")
    state = reliability_graph.run(state, config)

    print("[9/15] Trust propagation...")
    state = trust_propagation.run(state, config)

    print("[10/15] AutoML training...")
    state = automl_trainer.run(state, config)

    print("[11/15] Model selection...")
    state = model_selection.run(state, config)

    print("[12/15] Feedback refinement...")
    state = feedback_refinement.run(state, config)

    print("[13/15] Feature trust & confidence score...")
    state = feature_trust.run(state, config)

    print("[14/15] Noise simulation...")
    state = noise_simulation.run(state, config)

    print("[15/16] Baseline comparison (Before vs After)...")
    state = baseline_comparison.run(state, config)

    print("[16/16] SHAP explainability...")
    state = explainability.run(state, config)

    # Print summary
    print("\n===== RESULTS =====")
    print(f"Structural Score (S):  {state.structural_score:.4f}")
    print(f"Statistical Score (T): {state.statistical_score:.4f}")
    print(f"Similarity Score:      {state.similarity_score:.4f}")
    print(f"Conflict Score:        {state.conflict_score:.4f}")
    print(f"Deduction Score (D):   {state.deduction_score:.4f}")
    print(f"Reliability Score (R): {state.reliability_score:.4f}")
    print(f"Confidence Score:      {state.confidence_score:.4f}")
    print(f"Best Model:            {state.best_model_name}")
    print(f"Task Type:             {state.task_type}")
    print("\nModel Scores:")
    for name, scores in (state.model_scores or {}).items():
        marker = " <-- BEST" if name == state.best_model_name else ""
        print(f"  {name}: cv={scores['cv_score']:.4f}, "
              f"reliability_adjusted={scores.get('reliability_adjusted_score', 0):.4f}{marker}")

    # Save model
    joblib.dump(state.trained_models[state.best_model_name], output_model)
    print(f"\nModel saved to: {output_model}")

    # Save report
    report = {
        "reliability_score": state.reliability_score,
        "confidence_score": state.confidence_score,
        "fusion_weights": {
            f"w{i+1}": float(w)
            for i, w in enumerate(
                state.fusion_weights if state.fusion_weights is not None else []
            )
        },
        "feature_trust": state.feature_trust,
        "shap_importance": state.shap_feature_importance,
        "high_importance_low_trust_features": state.high_importance_low_trust_features,
        "model_scores": state.model_scores,
        "best_model": state.best_model_name,
    }
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"Report saved to: {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability-Aware AutoML CLI")
    parser.add_argument("csv", help="Path to input CSV file")
    parser.add_argument("--model", default="best_model.joblib", help="Output model path")
    parser.add_argument("--report", default="report.json", help="Output report path")
    args = parser.parse_args()
    run(args.csv, args.model, args.report)
