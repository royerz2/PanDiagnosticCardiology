# PanDiagnosticCardiology

Set-cover optimisation for rapid, multi-pathology cardiology diagnostics.

This repository implements a reproducible computational pipeline to identify the minimum biomarker panel that covers major life-threatening chest-pain pathologies under sensitivity constraints, with robustness, ablation, and clinical utility analyses.

## Scope

- Builds a pathology-biomarker coverage matrix from curated literature-derived inputs.
- Solves for minimum diagnostic panels using weighted set-cover / minimal hitting-set formulations.
- Runs sensitivity, ablation, bootstrap, Monte Carlo, and serial testing analyses.
- Produces publication figures and structured result artifacts in CSV/JSON.

## Repository layout

- `run_pipeline.py`: full end-to-end analysis runner.
- `diagnostic_panel_solver.py`: core optimisation solver and scoring logic.
- `biomarker_coverage_matrix.py`: coverage/source matrix construction.
- `pareto_ablation_analysis.py`: Pareto, ablation, utility, uncertainty analyses.
- `serial_testing_model.py`: serial testing and pathway-level simulations.
- `sister_act_score.py`: SISTER ACT extension and AI e-stethoscope analysis.
- `visualisation.py`: figure generation.
- `results/`: generated tables and JSON outputs.
- `figures/`: publication figure outputs.
- `manuscript/`: manuscript source and bibliography assets.

## Quick start

### 1) Create environment

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run full pipeline

```bash
python run_pipeline.py
```

### 3) Run tests

```bash
pytest test_pipeline.py -v
```

## Outputs

After running the pipeline:

- Primary analysis tables and summaries are written to `results/`.
- Publication-grade visual assets are written to `figures/`.

## Reproducibility notes

- The project is deterministic for most analytical steps; stochastic procedures use fixed structures with reported summary statistics.
- Input assumptions and methodological details are documented in `CHEATSHEET.md` and manuscript materials.

## Citation

If you use this repository in academic work, please cite:

- Erzurumluoğlu R. PanDiagnosticCardiology (2026).

You can also use metadata in `CITATION.cff`.

## License

This project is released under the MIT License. See `LICENSE`.

## Disclaimer

This repository is for research and methodological evaluation. It is not a medical device and must not be used as a standalone basis for clinical decision-making.