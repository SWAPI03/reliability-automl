# Reliability-Aware AutoML

Most AutoML tools assume your data is clean. This one doesn't.

Upload a CSV, and before any model gets trained, the system scores your data across five quality dimensions — missing values, outliers, duplicate conflicts, feature similarity, cluster structure — and fuses them into a single **Reliability Score R**. Every row gets a trust weight. Models train on those weights. The best model is picked by `accuracy × R`, not accuracy alone.

Built as a final-year BTech project at IIIT Manipur.

---

## The core idea

Real datasets are messy. A row with 40% missing values shouldn't carry the same weight as a clean one. A row that conflicts with its neighbors (same features, different label) is probably mislabeled. Standard AutoML ignores all of this.

Here, data quality feeds directly into training. The pipeline computes a per-row trust score, propagates it through a similarity graph, and uses it as a sample weight. The result is a model that's less sensitive to noise and more honest about what it actually learned.

---

## Pipeline

16 stages run sequentially. All state flows through a single `PipelineState` dataclass.

```
Stage 1   Validation          Parse CSV, detect column types, find target column
Stage 2   Preprocessing       Mean impute + StandardScaler for numeric,
                              most_frequent + OneHotEncoder for categorical

Stage 3   Structural Score    S = 1 - (0.4·missing + 0.3·duplicates + 0.3·constants)
Stage 4   Statistical Score   T = 1 - (0.6·outlier_ratio + 0.4·low_variance_ratio)
                              Z-score for <1000 rows, IsolationForest for larger
Stage 5   Pattern Analysis    Sim = mean pairwise similarity
                              C   = fraction of conflicting row pairs
Stage 6   Deduction Score     D = KMeans cluster trust (5 pseudo-sources by default)

Stage 7   Reliability Fusion  R = w1·S + w2·T + w3·Sim + w4·(1-C) + w5·D
                              weights learned per-dataset via variance+entropy blend

Stage 8   Reliability Graph   NetworkX graph — rows as nodes, similarity as edges
                              full pairwise for <5000 rows, k-NN ball_tree for larger
Stage 9   Trust Propagation   Iterative trust flow over the graph
                              neighbor influence + conflict penalty + 70/30 blend

Stage 10  AutoML Training     RandomForest, LogisticRegression, XGBoost
                              trained with sample_weight = trust × R
Stage 11  Model Selection     best model = highest (cv_score × R)
Stage 12  Feedback Refinement trust scores updated using prediction errors
Stage 13  Feature Trust       per-feature trust = mean trust of non-null rows
Stage 14  Noise Simulation    re-train on 10%-noise data with same weights
Stage 15  Baseline Comparison unweighted baseline + noise experiment at 0/10/20/30%
Stage 16  Explainability      SHAP values, flags high-importance low-trust features
```

---

## Getting started

```bash
git clone https://github.com/<your-username>/reliability-automl.git
cd reliability-automl
pip install -r requirements.txt
streamlit run reliability_automl/app.py
```

Open `http://localhost:8501`, drop in a CSV, and the pipeline runs automatically.

The target column is detected by name (`target`, `label`, `class`) or falls back to the last column. Sample datasets are in `datasets/` if you want to try it immediately.

---

## Dashboard

Six sections in the sidebar:

- **Data Overview** — shape, task type, column types, first 10 rows
- **Reliability Scores** — S/T/Sim/C/D/R cards, component bar chart, fusion weights pie
- **Trust Graph** — network visualization (up to 200 nodes), heatmap, score histogram
- **Model Performance** — CV scores, reliability-adjusted scores, clean vs noisy bar chart
- **Explainability** — SHAP top-10, feature trust bars, trust vs SHAP scatter with flagged features
- **Before vs After** — baseline vs weighted accuracy, stability table, noise experiment line chart

Downloads at the bottom: best model as `.joblib`, full report as JSON.

---

## Project structure

```
reliability_automl/
├── app.py
├── config.py
├── pipeline/
│   ├── state.py
│   ├── validator.py
│   ├── preprocessor.py
│   ├── structural_score.py
│   ├── statistical_score.py
│   ├── pattern_analysis.py
│   ├── deduction_score.py
│   ├── reliability_fusion.py
│   ├── reliability_graph.py
│   ├── trust_propagation.py
│   ├── automl_trainer.py
│   ├── model_selection.py
│   ├── feedback_refinement.py
│   ├── feature_trust.py
│   ├── noise_simulation.py
│   ├── baseline_comparison.py
│   └── explainability.py
└── ui/
    ├── dashboard.py
    ├── components.py
    └── styles.py

datasets/       sample CSVs for testing
tests/          unit + integration tests for all pipeline modules
docs/           project documentation
diagrams/       architecture diagrams
```

---

## Configuration

`reliability_automl/config.py` — tweak any of these before running:

```python
PipelineConfig(
    similarity_threshold     = 0.5,    # minimum similarity to add a graph edge
    low_variance_threshold   = 0.01,   # below this = low-variance column
    outlier_zscore_threshold = 3.0,    # z-score cutoff (small datasets only)
    n_pseudo_sources         = 5,      # number of KMeans clusters
    trust_convergence_tol    = 0.001,  # stop propagation when delta < this
    trust_max_iterations     = 1000,   # hard cap on propagation iterations
    noise_level_fraction     = 0.10,   # noise intensity for simulation stage
    cv_folds                 = 5,
    fusion_weights           = None,   # pass a list of 5 to override dynamic weights
)
```

---

## Results

Tested on four synthetic datasets ranging from clean to extreme noise:

| Rows   | Features | Noise | R Score | Best Model   | Accuracy |
|--------|----------|-------|---------|--------------|----------|
| 10,500 | 20       | 15%   | 0.937   | RandomForest | 85.5%    |
| 16,200 | 25       | 25%   | 0.926   | RandomForest | 82.9%    |
| 20,400 | 15       | 5%    | 0.945   | XGBoost      | 93.6%    |
| 13,440 | 30       | 35%   | 0.895   | RandomForest | 79.6%    |

~300 rows/second. All 16 stages completed on every dataset. R scores tracked noise levels accurately — the 35%-noise dataset got the lowest R, the 5%-noise dataset got the highest.

---

## Tests

```bash
pytest tests/
```

Each pipeline module has its own test file. `test_integration.py` runs the full pipeline end-to-end.

---

## Dependencies

```
streamlit, pandas, numpy, scikit-learn, xgboost,
networkx, shap, plotly, scipy, joblib
```

---

## Notes on design

A few things worth knowing if you're reading the code:

- Fusion weights are learned per-dataset, not hardcoded. The variance+entropy blend means a component that varies more across rows gets more weight — it's carrying more signal for that specific dataset.
- Every scoring stage has a large-dataset fast path. Pattern analysis switches from full pairwise to k-NN at 5000 rows. The graph builder samples at 50K. Statistical scoring switches from Z-score to IsolationForest at 1000 rows.
- Trust has a floor of 0.05. No row gets completely zeroed out — that would be too aggressive given that the trust scores themselves are estimates.
- The pipeline result is cached in `st.session_state` by `(filename, filesize)`. Re-selecting a dashboard section doesn't re-run the pipeline.
- After training, prediction errors feed back into trust scores. Rows the model consistently gets wrong get lower trust on the next pass.

---

## License

MIT — see [LICENSE](LICENSE).

---

*BTech CSE, IIIT Manipur*
