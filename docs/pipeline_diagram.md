# Reliability-Aware AutoML — Report Figures

All figures are rendered in `pipeline_diagram.html` (black-and-white, print-ready).
The LaTeX source is in `Report_Format___BTech_CSE_IIIT_Manipur__1___1_/report.tex`.
The bibliography is in `Report_Format___BTech_CSE_IIIT_Manipur__1___1_/report.bib`.

## Figures in report.tex (TikZ / pgfplots — no colour)

| Label | Caption | Chapter |
|-------|---------|---------|
| `fig:highlevel`   | High-level pipeline overview: from CSV upload to trust-weighted model output | Ch. 1 |
| `fig:gantt`       | Gantt chart showing the project development phases across 13 weeks | Ch. 1 |
| `fig:usecase`     | Use case diagram: User actor interacting with the Reliability-Aware AutoML system | Ch. 3 |
| `fig:architecture`| Three-tier architecture: Streamlit UI, Pipeline Engine (16 stages), and in-memory State Store | Ch. 4 |
| `fig:ingest`      | Validation pipeline: CSV parsing, row count check, column type detection, and target identification | Ch. 4 |
| `fig:scoring`     | Five-dimensional reliability scoring and fusion into the composite score R | Ch. 4 |
| `fig:graph`       | Row-similarity graph construction based on cosine similarity threshold | Ch. 4 |
| `fig:training`    | AutoML training pipeline with trust-weighted cross-validation and reliability-adjusted model selection | Ch. 4 |
| `fig:erdiagram`   | Logical entity-relationship diagram of the PipelineState dataclass | Ch. 4 |
| `fig:uidesign`    | Conceptual UI layout: sidebar navigation, file uploader, metric cards, and chart area | Ch. 4 |
| `fig:noiseplot`   | Noise robustness experiment: trust-weighted models degrade more slowly under increasing Gaussian noise | Ch. 6 |
| `fig:trustdist`   | Trust score distribution after propagation | Ch. 6 |

## Figures in pipeline_diagram.html (SVG — for browser preview / export)

Same 12 figures rendered as black-and-white SVG/HTML for browser preview and PNG export.

## Key References Used

| Cite key | Paper |
|----------|-------|
| `Dai2008Trust` | Dai et al. (2008) — **Core base paper** — provenance-based trust |
| `Dong2015KBT` | Dong et al. (2015) — Knowledge-Based Trust |
| `Bertino2015Trust` | Bertino (2015) — Data Trustworthiness |
| `PageRank1998` | Brin & Page (1998) — PageRank (trust propagation inspiration) |
| `Zhu2002SSL` | Zhu & Ghahramani (2002) — Label Propagation |
| `kamvar2003eigentrust` | Kamvar et al. (2003) — EigenTrust |
| `zhu2003semi` | Zhu et al. (2003) — Semi-supervised learning |
| `feurer2015efficient` | Feurer et al. (2015) — Auto-sklearn |
| `olson2016tpot` | Olson et al. (2016) — TPOT |
| `IsolationForest2008` | Liu et al. (2008) — Isolation Forest |
| `DataCleaning2016` | Ilyas & Chu (2016) — Data Cleaning |
| `SHAP2017` | Lundberg & Lee (2017) — SHAP |
| `ren2018learning` | Ren et al. (2018) — Sample reweighting |
| `whang2023data` | Whang et al. (2023) — Data-centric AI |
| `schelter2018automating` | Schelter et al. (2018) — Deequ |

## Design Rules

- **No colour** in any figure — black, white, and grayscale only
- TikZ styles: `blockdark` (black fill, white text), `blockgray` (20% black), `blocklight` (8% black), `block` (white)
- All pgfplots charts use `black` line colours with different dash/marker styles to distinguish series
- ER diagram uses proper entity boxes with attribute lists, relationship lines, cardinality labels, and a legend
