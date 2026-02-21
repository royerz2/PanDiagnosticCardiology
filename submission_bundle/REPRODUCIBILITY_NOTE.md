# Reproducibility Note (PanDiagnosticCardiology)

## Scope
This bundle contains the frozen manuscript PDFs and core result artifacts after integrating Step 18 (SISTER ACT score + AI e-stethoscope) into the full pipeline.

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

3. Cheatsheet PDF render
```bash
cd /Users/royerzurumloglu/PanDiagnosticCardiology
pdflatex -interaction=nonstopmode CHEATSHEET.tex
```

## Key refreshed outputs in this bundle
- `paper.pdf`
- `CHEATSHEET.pdf`
- `sister_act_score.json`
- `serial_testing_protocols.json`
- `monte_carlo_ci.json`
- `clinical_utility.json`
- `approach_comparison.csv`
- `dutch_patient_flow.json`

## Key Step 18 headline metrics (from `sister_act_score.json`)
- Extended coverage: 5/6 (83%) -> 6/6 (100%) with AI e-stethoscope
- SISTER ACT sensitivity: 97.7%
- SISTER ACT specificity: 70.0%
- Low-tier NPV: 0.9955
- Referral rate: 38.2%
- Missed serious pathology: 2.8 per 1,000
- PTX sensitivity: 14.3% (biomarker-only) -> 95.9% (SISTER ACT)

## Notes
- The integration in `run_pipeline.py` now runs Step 18 successfully in the full pipeline.
- Manuscript and cheatsheet score-component ranges were synchronized to implementation:
  - Timeline: 0-3
  - Risk factors: 0-3
