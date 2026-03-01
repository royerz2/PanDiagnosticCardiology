# PanDiagnosticCardiology

**Multi-Pathology Point-of-Care Diagnostic Panel Optimisation: A Computational Framework for Acute Chest Pain Triage**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository implements a reproducible computational framework that formalises multi-pathology POCT panel selection as a weighted set-cover problem and applies it to the acute chest pain triage context (SISTER ACT trial). The framework identifies a uniquely optimal four-test panel, quantifies uncertainty honestly, and produces a prioritised measurement agenda for prospective trials.

## Key result

At a sensitivity threshold of 0.90, the ILP and exhaustive enumeration return the same unique panel:

> **hs-cTnI + D-dimer + NT-proBNP + CRP**
>
> 4 tests | 5/6 pathologies covered | EUR 36 | 15 min | 55 uL capillary blood

Panel selection is **invariant** to all plausible specificity values (confirmed across 2,000 joint Monte Carlo draws). The ED false-positive rate after full mitigation is bounded to 14.8--16.4% (95% credible interval).

## Repository layout

| File | Purpose |
|------|---------|
| `run_pipeline.py` | End-to-end pipeline runner (23 steps) |
| `biomarker_coverage_matrix.py` | Coverage matrix construction from published meta-analyses |
| `diagnostic_panel_solver.py` | ILP / exhaustive / greedy set-cover solvers |
| `pareto_ablation_analysis.py` | Pareto frontier, ablation, bootstrap, threshold sensitivity |
| `serial_testing_model.py` | Serial testing, HEAR stratification, Dutch GP patient flow |
| `correlation_dependence_model.py` | Gaussian copula false-positive correction |
| `quantitative_panel_interpretation.py` | Likelihood-ratio interpretation and pathology-directed routing |
| `health_economics.py` | Single-cycle decision tree, ICER, CEAC |
| `sister_act_score.py` | PACE composite score and extended analyses |
| `sensitivity_analysis.py` | Parametric SA, tornado, what-if envelope |
| `visualisation.py` | Publication figure generation |
| `test_pipeline.py` | Pytest suite for pipeline verification |
| `results/` | Generated CSV/JSON outputs |
| `figures/` | Generated publication figures |
| `manuscript/` | LaTeX manuscript source and bibliography |

## Quick start

### 1. Create environment

Python 3.10+ required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python run_pipeline.py
```

This executes all 23 analytical steps and populates `results/` and `figures/`.

### 3. Run tests

```bash
pytest test_pipeline.py -v
```

### 4. Compile manuscript

```bash
cd manuscript
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Pipeline steps

1. Coverage matrix construction
2. Set-cover optimisation (ILP + exhaustive verification)
3. Marginal diagnostic value analysis
4. Reference approach comparison
5. Pareto frontier
6. Ablation analysis
7. Threshold sensitivity
8. Bootstrap panel stability (*n* = 1,000)
9. Early presenter subgroup analysis
10. Weight sensitivity (penalty invariance proof)
11. Feasibility landscape (solution uniqueness)
12. Clinical utility (specificity, prevalence, NPV, net benefit)
13. Full 48-cell source provenance table
14. Monte Carlo CI propagation (*n* = 5,000)
15. Copeptin threshold sensitivity
16. Serial testing, HEAR score, and Dutch GP patient flow
17. Publication figures
18. PACE composite score analysis
19. Biomarker correlation and Gaussian copula modelling
20. Extended biomarker pool (12 markers)
21. Health-economic analysis
22. Quantitative LR interpretation and pathology-directed management
23. Parametric sensitivity analysis (tornado + what-if envelope)

## Reproducibility

- All stochastic procedures use fixed random seeds or report summary statistics.
- Input assumptions and methodological details are documented in `CHEATSHEET.md` and the manuscript.
- The full pipeline regenerates all results and figures from source data.

## Citation

If you use this repository, please cite:

> Erzurumluoglu, R. (2026). *Multi-Pathology Point-of-Care Diagnostic Panel Optimisation: A Computational Framework for Acute Chest Pain Triage.* https://github.com/royerz2/PanDiagnosticCardiology

See also `CITATION.cff` for machine-readable metadata.

## License

MIT. See [LICENSE](LICENSE).

## Disclaimer

This repository is for research and methodological evaluation. It is not a medical device and must not be used as a standalone basis for clinical decision-making.