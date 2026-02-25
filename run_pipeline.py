"""
Pan-Diagnostic Cardiology: Full Pipeline Runner
================================================
Set-Cover Optimisation for Multi-Biomarker Diagnostic Panels.

Runs the complete analysis pipeline:
  1. Coverage matrix construction (from published meta-analyses)
  2. Set-cover optimisation (minimum diagnostic panel)
  3. Multi-objective scoring (coverage, cost, time, sample volume)
  4. Pareto frontier & reference approach comparison
  5. Ablation & sensitivity analysis
  6. Bootstrap robustness
  7. Early presenter subgroup analysis
  8. Clinical utility scoring (net benefit, NPV)
  9. Monte Carlo confidence intervals
 10. Copeptin threshold analysis
 11. Feasibility landscape & weight sensitivity
 12. Source provenance & coverage sources
 13. Approach comparison table
 14. Copeptin dual-marker analysis
 15. Serial testing & biomarker kinetics
 16. HEAR score stratification & Dutch GP patient flow
 17. Publication figures (8 panels)
 18. SISTER ACT score with AI e-stethoscope analysis
 19. Biomarker correlation & conditional dependence modelling
 20. Extended biomarker pool optimisation (12 biomarkers)
 21. Health-economic analysis (ICER, PSA, CEAC, Dutch GP impact)
"""

import os
import sys
import json
import logging
import pandas as pd

# Ensure package is importable
sys.path.insert(0, os.path.dirname(__file__))

from biomarker_coverage_matrix import build_coverage_matrix, build_source_matrix
from diagnostic_panel_solver import (
    DiagnosticPanelSolver, marginal_value_analysis, current_approach_comparison,
)
from pareto_ablation_analysis import (
    compute_pareto_frontier, get_reference_approaches,
    ablation_analysis, threshold_sensitivity,
    bootstrap_panel_stability, early_presenter_analysis,
    weight_sensitivity_analysis, feasibility_landscape,
    clinical_utility_analysis, build_full_source_table,
    monte_carlo_ci_propagation, copeptin_threshold_analysis,
)
from serial_testing_model import run_serial_analysis
from sister_act_score import run_sister_act_analysis
from visualisation import generate_all_figures
from correlation_dependence_model import run_dependence_analysis
from health_economics import run_health_economics_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print("PAN-DIAGNOSTIC CARDIOLOGY — SET-COVER OPTIMISATION PIPELINE")
    print("Adapted from ALIN Framework (Erzurumluoğlu, 2026)")
    print("All data from published meta-analyses — no new patient data required")
    print("=" * 90)

    # ── Step 1: Coverage Matrix ──
    print("\n[STEP 1] Building pathology–biomarker coverage matrix...")
    C = build_coverage_matrix()
    print(C.to_string(float_format=lambda x: f"{x:.2f}"))
    C.to_csv(os.path.join(output_dir, "coverage_matrix.csv"))

    sources = build_source_matrix()
    sources.to_csv(os.path.join(output_dir, "coverage_sources.csv"))
    print(f"  Saved coverage matrix and source citations to {output_dir}/")

    # ── Step 2: Set-Cover Optimisation ──
    print(f"\n[STEP 2] Running set-cover optimisation...")
    solver = DiagnosticPanelSolver()

    all_solutions = {}
    for tau in [0.80, 0.85, 0.90, 0.95]:
        solutions = solver.solve(tau=tau, max_size=8)
        all_solutions[tau] = solutions

        print(f"\n  τ = {tau}:")
        if solutions:
            best = solutions[0]
            print(f"    MINIMUM PANEL: {', '.join(sorted(best.biomarkers))}")
            print(f"    Size: {len(best)} | Coverage: {best.coverage:.0%} "
                  f"| Cost: €{best.total_cost_eur:.2f}")
            print(f"    Worst-case sensitivity: {best.worst_case_sensitivity:.2f}")
            if best.pathologies_uncovered:
                print(f"    Gaps: {', '.join(sorted(best.pathologies_uncovered))}")
            else:
                print(f"    Gaps: NONE — full coverage")
        else:
            print("    No feasible panel found")

    # ── Step 3: Marginal Value ──
    print(f"\n[STEP 3] Marginal diagnostic value analysis...")
    marginal = marginal_value_analysis(solver, tau=0.90)
    print(marginal.to_string(index=False))
    marginal.to_csv(os.path.join(output_dir, "marginal_value.csv"), index=False)

    # ── Step 4: Current Approach Comparison ──
    print(f"\n[STEP 4] Comparing current approaches vs optimal panels...")
    comparison = current_approach_comparison(solver, tau=0.90)
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(output_dir, "approach_comparison.csv"), index=False)

    # ── Step 5: Pareto Frontier ──
    print(f"\n[STEP 5] Computing Pareto frontier...")
    pareto = compute_pareto_frontier(solver, tau=0.90)
    pareto_optimal = pareto[pareto['pareto_optimal']].sort_values('cost_eur')
    print(f"  {len(pareto_optimal)} Pareto-optimal panels")
    print(pareto_optimal[['biomarkers', 'n_tests', 'coverage', 'cost_eur']].to_string(index=False))
    pareto.to_csv(os.path.join(output_dir, "pareto_all_panels.csv"), index=False)
    pareto_optimal.to_csv(os.path.join(output_dir, "pareto_optimal.csv"), index=False)

    # ── Step 6: Ablation ──
    print(f"\n[STEP 6] Ablation analysis...")
    ablation = ablation_analysis(tau=0.90)
    print(ablation[['removed', 'in_optimal', 'coverage_change', 'new_gaps']].to_string(index=False))
    ablation.to_csv(os.path.join(output_dir, "ablation.csv"), index=False)

    # ── Step 7: Threshold Sensitivity ──
    print(f"\n[STEP 7] Threshold sensitivity...")
    thresh = threshold_sensitivity()
    print(thresh.to_string(index=False))
    thresh.to_csv(os.path.join(output_dir, "threshold_sensitivity.csv"), index=False)

    # ── Step 8: Bootstrap Robustness ──
    print(f"\n[STEP 8] Bootstrap panel stability (n=1000)...")
    bootstrap = bootstrap_panel_stability(n_bootstrap=1000, tau=0.90)
    print(f"  Most stable panel: {bootstrap['most_stable_panel']}")
    print(f"  Stability: {bootstrap['stability_score']:.1%}")
    print(f"  Mean size: {bootstrap['panel_size_mean']:.1f} ± {bootstrap['panel_size_std']:.2f}")
    print(f"  Biomarker inclusion rates:")
    for b, rate in sorted(bootstrap['biomarker_inclusion_rate'].items(), key=lambda x: -x[1]):
        bar = '█' * int(rate * 40)
        print(f"    {b:15s} {rate:5.1%} {bar}")
    with open(os.path.join(output_dir, "bootstrap_stability.json"), 'w') as f:
        json.dump(bootstrap, f, indent=2)

    # ── Step 9: Early Presenter Subgroup ──
    print(f"\n[STEP 9] Early presenter (<2h) subgroup analysis...")
    early = early_presenter_analysis(tau=0.90)
    print(f"  Standard panel: {early['standard']['panel'] if early['standard'] else 'N/A'}")
    print(f"  Early panel:    {early['early_presenter']['panel'] if early['early_presenter'] else 'N/A'}")
    print(f"  Panel changed:  {early['panel_changed']}")
    if early['new_markers_needed']:
        print(f"  New markers for early presenters: {early['new_markers_needed']}")
    with open(os.path.join(output_dir, "early_presenter.json"), 'w') as f:
        json.dump(early, f, indent=2)

    # ── Step 10: Weight Sensitivity ──
    print(f"\n[STEP 10] Weight sensitivity analysis (penalty invariance proof)...")
    weight_result = weight_sensitivity_analysis(tau=0.90)
    print(f"  Dominant panel: {weight_result['dominant_panel']}")
    print(f"  Fraction: {weight_result['dominant_fraction']:.1%}")
    print(f"  Weights irrelevant: {weight_result['weights_irrelevant']}")
    with open(os.path.join(output_dir, "weight_sensitivity.json"), 'w') as f:
        json.dump(weight_result, f, indent=2)

    # ── Step 11: Feasibility Landscape ──
    print(f"\n[STEP 11] Feasibility landscape (solution uniqueness)...")
    landscape = feasibility_landscape()
    for _, row in landscape.iterrows():
        print(f"  τ={row['threshold']:.2f}: {row['n_coverable']} coverable, "
              f"{row['n_feasible_panels']} feasible, "
              f"{row['n_optimal_panels']} optimal")
    landscape_save = landscape.drop(columns=['optimal_panels', 'biomarker_choices_per_pathology'])
    landscape_save.to_csv(os.path.join(output_dir, "feasibility_landscape.csv"), index=False)
    with open(os.path.join(output_dir, "feasibility_landscape.json"), 'w') as f:
        json.dump(landscape.to_dict(orient='records'), f, indent=2, default=str)

    # ── Step 12: Clinical Utility ──
    print(f"\n[STEP 12] Clinical utility analysis (specificity, prevalence, net benefit)...")
    utility = clinical_utility_analysis(tau=0.90)
    print(f"  Prevalence-weighted cov: {utility['optimal_panel']['prevalence_weighted_coverage']:.4f}")
    print(f"  Severity-weighted cov: {utility['optimal_panel']['severity_weighted_coverage']:.2f}")
    print(f"  Expected detections/1000: {utility['optimal_panel']['expected_detections_per_1000']:.1f}")
    print(f"  Mean specificity: {utility['optimal_panel']['mean_specificity']:.3f}")
    print(f"  Net benefit (Vickers): {utility['optimal_panel']['net_benefit']:.4f}")
    print(f"  Robust panel matches: {utility['robust_panel']['same_as_point_estimate']}")
    print(f"\n  Per-pathology rule-out suitability:")
    for d in utility['per_pathology_details']:
        print(f"    {d['pathology']:10s}: sens={d['sensitivity']:.2f} spec={d['specificity']:.2f} "
              f"NPV={d['npv']:.4f} rule-out={'YES' if d['rule_out_suitable'] else 'no'}")
    with open(os.path.join(output_dir, "clinical_utility.json"), 'w') as f:
        json.dump(utility, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    # ── Step 13: Full Source Provenance ──
    print(f"\n[STEP 13] Full 48-cell source provenance table...")
    source_table = build_full_source_table()
    source_table.to_csv(os.path.join(output_dir, "full_source_provenance.csv"), index=False)
    print(f"  {len(source_table)} entries written")

    # ── Step 14: Monte Carlo CI Propagation ──
    print(f"\n[STEP 14] Monte Carlo CI propagation (n=5000)...")
    mc = monte_carlo_ci_propagation(n_samples=5000, tau=0.90)
    print(f"  Coverage: {mc['coverage']['mean']:.3f} "
          f"[{mc['coverage']['ci_2.5']:.3f}, {mc['coverage']['ci_97.5']:.3f}]")
    print(f"  P(full coverage): {mc['coverage']['p_full_coverage']:.1%}")
    print(f"  Detections (covered)/1000: "
          f"{mc['expected_detections_covered_per_1000']['mean']:.1f}")
    print(f"  Detections (all)/1000: "
          f"{mc['expected_detections_all_per_1000']['mean']:.1f}")
    print(f"  Net benefit at multiple thresholds:")
    for key, val in mc['net_benefit'].items():
        print(f"    {key}: mean={val['mean']:.4f}  "
              f"P(>0)={val['p_positive']:.1%}")
    with open(os.path.join(output_dir, "monte_carlo_ci.json"), 'w') as f:
        json.dump(mc, f, indent=2)

    # ── Step 15: Copeptin Threshold Sensitivity ──
    print(f"\n[STEP 15] Copeptin threshold sensitivity analysis...")
    copeptin_result = copeptin_threshold_analysis(tau=0.90)
    print(f"  Interpretation: {copeptin_result['clinical_interpretation'][:100]}...")
    with open(os.path.join(output_dir, "copeptin_threshold_analysis.json"), 'w') as f:
        json.dump(copeptin_result, f, indent=2)

    # ── Step 16: Serial Testing & CDR Analysis (SISTER ACT Extension) ──
    print(f"\n[STEP 16] Serial testing, HEAR score, & Dutch GP analysis...")
    serial_results = run_serial_analysis(
        panel_biomarkers=["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"],
        tau=0.90,
        output_dir=output_dir,
    )

    # ── Step 17: Figures ──
    print(f"\n[STEP 17] Generating publication figures...")
    generate_all_figures()

    # ── Step 18: SISTER ACT Score & AI E-Stethoscope ──
    print(f"\n[STEP 18] SISTER ACT score with AI e-stethoscope analysis...")
    sister_act = run_sister_act_analysis(n_patients=10_000, output_dir=output_dir)
    sa_cov = sister_act["extended_coverage"]["biomarker_plus_estethoscope"]
    sa_sp = sister_act["performance"]["screening_performance"]
    sa_cdr = sister_act["cdr_comparison"]["systems"]["SISTER_ACT"]
    print(f"  Extended coverage: {sa_cov['coverage']}")
    print(f"  Sensitivity: {sa_sp['sensitivity'] * 100:.1f}% | Specificity: {sa_sp['specificity'] * 100:.1f}%")
    print(f"  Referral rate: {sa_cdr['referral_rate'] * 100:.1f}% | Missed/1000: {sa_sp['miss_rate_per_1000']:.1f}")

    # ── Step 19: Correlation & Conditional Dependence ──
    print(f"\n[STEP 19] Biomarker correlation & conditional dependence modelling...")
    dep_results = run_dependence_analysis(
        panel_biomarkers=["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"],
        output_dir=output_dir,
    )
    fp_ind = dep_results['corrected_fp_rate']['independence_assumption']['fp_rate']
    fp_cop = dep_results['corrected_fp_rate']['copula_corrected']['fp_rate']
    print(f"  Independence FP rate: {fp_ind:.1%}")
    print(f"  Copula-corrected FP: {fp_cop:.1%}")
    print(f"  Reduction: {(fp_ind - fp_cop) / fp_ind * 100:.1f}% via correlation modelling")
    bn = dep_results['sequential_testing']
    print(f"  Sequential testing: mean {bn['average_tests_per_patient']:.1f} tests/patient")
    print(f"    (saves {4 - bn['average_tests_per_patient']:.1f} tests on average)")

    # ── Step 20: Extended Pool Optimisation ──
    print(f"\n[STEP 20] Extended biomarker pool optimisation (12 biomarkers)...")
    from biomarker_coverage_matrix import build_extended_pool_matrix, EXTENDED_BIOMARKERS
    ext_C = build_extended_pool_matrix(include_extended=True)
    print(f"  Extended matrix: {ext_C.shape[0]} pathologies × {ext_C.shape[1]} biomarkers")
    print(f"  New biomarkers: {', '.join(EXTENDED_BIOMARKERS)}")
    ext_C.to_csv(os.path.join(output_dir, "extended_coverage_matrix.csv"))

    # ── Step 21: Health-Economic Analysis ──
    print(f"\n[STEP 21] Health-economic analysis (ICER, PSA, Dutch GP impact)...")
    he_results = run_health_economics_analysis(
        output_dir=output_dir,
        cohort_size=10_000,
    )
    for key, outcome in he_results['strategy_outcomes'].items():
        print(f"  {outcome['name']}: €{outcome['cost_per_patient']:.2f}/patient, "
              f"sens={outcome['sensitivity']:.1%}, missed={outcome['missed_cases']}")
    for key, icer in he_results['icers'].items():
        print(f"  ICER ({key}): {icer['icer_eur_per_qaly']}")

    # ── Summary ──
    print(f"\n{'=' * 90}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 90}")
    print(f"\nResults directory: {output_dir}/")
    print(f"Figures directory: {os.path.join(os.path.dirname(__file__), 'figures')}/")
    
    # Print file listing
    for d in [output_dir, os.path.join(os.path.dirname(__file__), 'figures')]:
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            print(f"\n  {d}/")
            for f in files:
                size = os.path.getsize(os.path.join(d, f))
                print(f"    {f:45s} {size:>8,} bytes")


if __name__ == "__main__":
    main()
