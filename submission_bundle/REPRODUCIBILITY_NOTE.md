# Reproducibility Note (PanDiagnosticCardiology)

## Scope
This bundle contains the frozen manuscript PDFs and core result artifacts after integrating all 22 pipeline steps, including the quantitative likelihood-ratio interpretation and pathology-directed management analysis.

## Environment
- OS: macOS
- Python environment used for pipeline run: `.venv`
- Python executable: `/Users/royerzurumloglu/PanDiagnosticCardiology/.venv/bin/python`

## Commands used
1. Full pipeline refresh
```bash
cd /Users/royerzurumloglu/PanDiagnosticCardiology
/Users/royerzurumloglu/PanDiagnosticCardiology/.venv/bin/python run_pipeline.py
```

2. Manuscript PDF render
```bash
cd /Users/royerzurumloglu/PanDiagnosticCardiology/manuscript
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

3. Full test suite
```bash
cd /Users/royerzurumloglu/PanDiagnosticCardiology
python -m pytest test_pipeline.py -v
```

## Pipeline steps (22 total)
1. Coverage matrix construction
2. Set-cover optimisation (ILP)
3. Marginal value analysis
4. Current approach comparison
5. Pareto frontier
6. Ablation analysis
7. Threshold sensitivity
8. Bootstrap robustness (n=1000)
9. Early presenter subgroup
10. Weight sensitivity
11. Feasibility landscape
12. Clinical utility scoring
13. Full source provenance
14. Monte Carlo CI propagation (n=5000)
15. Copeptin threshold analysis
16. Serial testing & HEAR score
17. Publication figures (11 panels)
18. SISTER ACT score & AI e-stethoscope
19. Biomarker correlation & copula dependence
20. Extended biomarker pool (12 biomarkers)
21. Health-economic analysis (ICER, PSA, CEAC)
22. Quantitative LR interpretation & pathology-directed management

## Key result files
- `paper_snapshot.tex` — Frozen manuscript (31 pages)
- `refs_snapshot.bib` — Frozen bibliography
- `sister_act_score.json` — SISTER ACT CDR simulation
- `serial_testing_protocols.json` — Sequential testing results
- `monte_carlo_ci.json` — Monte Carlo CIs
- `clinical_utility.json` — Clinical utility metrics
- `approach_comparison.csv` — Strategy comparison table
- `dutch_patient_flow.json` — Dutch GP patient flow model
- `quantitative_interpretation.json` — Quantitative LR analysis

## Test suite
- 93 automated tests covering all modules
- Run with: `pytest test_pipeline.py -v`

## Notes
- The integration in `run_pipeline.py` now runs Step 18 successfully in the full pipeline.
- Manuscript and cheatsheet score-component ranges were synchronized to implementation:
  - Timeline: 0-3
  - Risk factors: 0-3
