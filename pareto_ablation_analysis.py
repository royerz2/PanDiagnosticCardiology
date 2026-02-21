"""
Pareto Frontier, Ablation & Sensitivity Analysis
=================================================
Implements Steps 4-5 from the proposal:
  - Pareto frontier (coverage vs cost)
  - Bootstrap robustness across CI
  - Ablation (remove each biomarker, observe coverage collapse)  
  - Threshold sensitivity analysis
  - Early presenter subgroup analysis
  - Weight sensitivity analysis (penalty-weight invariance proof)
  - Clinical utility scoring (specificity, prevalence, net benefit)
  - Solution uniqueness / feasibility landscape
  - Full 48-cell source provenance table
"""

import json
import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

from biomarker_coverage_matrix import (
    BIOMARKERS, PATHOLOGIES, BIOMARKER_META, PATHOLOGY_SHORT,
    build_coverage_matrix, build_ci_matrices, build_early_coverage_matrix,
    build_specificity_matrix, get_prevalence_weights, get_severity_weights,
    build_beta_parameters, classify_reference_quality,
    PATHOLOGY_EPIDEMIOLOGY, COVERAGE_DATA, SPECIFICITY_DATA,
)
from diagnostic_panel_solver import DiagnosticPanelSolver, DiagnosticPanel

logger = logging.getLogger(__name__)


# ─── Pareto frontier ────────────────────────────────────────────────────────

def compute_pareto_frontier(
    solver: DiagnosticPanelSolver,
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    Enumerate all 255 possible panels. For each, compute (coverage, cost).
    Identify the Pareto-optimal set: panels where no other panel achieves
    strictly better coverage at equal or lower cost, AND vice versa.
    
    Returns DataFrame with all panels and a 'pareto_optimal' flag.
    """
    panels = solver.enumerate_all_panels(tau=tau)

    rows = []
    for panel in panels:
        rows.append({
            'biomarkers': ', '.join(sorted(panel.biomarkers)),
            'n_tests': len(panel),
            'coverage': panel.coverage,
            'worst_case_sensitivity': panel.worst_case_sensitivity,
            'cost_eur': panel.total_cost_eur,
            'time_min': panel.total_time_min,
            'sample_ul': panel.total_sample_ul,
            'pathologies_covered': len(panel.pathologies_covered),
            'pathologies_uncovered': ', '.join(sorted(panel.pathologies_uncovered)) or 'None',
        })

    df = pd.DataFrame(rows)

    # Identify Pareto front (maximize coverage, minimize cost)
    pareto_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            # j dominates i if j has >= coverage AND <= cost, with at least one strict
            if (df.loc[j, 'coverage'] >= df.loc[i, 'coverage'] and
                df.loc[j, 'cost_eur'] <= df.loc[i, 'cost_eur'] and
                (df.loc[j, 'coverage'] > df.loc[i, 'coverage'] or
                 df.loc[j, 'cost_eur'] < df.loc[i, 'cost_eur'])):
                pareto_mask[i] = False
                break

    df['pareto_optimal'] = pareto_mask

    logger.info(f"Pareto frontier: {pareto_mask.sum()} panels out of {len(df)}")
    return df


def get_reference_approaches(
    solver: DiagnosticPanelSolver,
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    Return the positions of current CDR-based approaches on the frontier.
    """
    approaches = {
        "HEART + hs-cTnI": {"hs-cTnI"},
        "Wells + D-dimer": {"D-dimer"},
        "HEART + hs-cTnI + Wells + D-dimer": {"hs-cTnI", "D-dimer"},
    }

    rows = []
    for name, biomarkers in approaches.items():
        panel = solver._make_panel(biomarkers, "reference", tau)
        rows.append({
            'approach': name,
            'biomarkers': ', '.join(sorted(biomarkers)),
            'n_tests': len(panel),
            'coverage': panel.coverage,
            'cost_eur': panel.total_cost_eur,
            'worst_case_sensitivity': panel.worst_case_sensitivity,
        })
    return pd.DataFrame(rows)


# ─── Ablation analysis (analogous to ALIN's component ablation) ─────────────

def ablation_analysis(
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    For each biomarker, remove it and observe how the optimal panel changes.
    Reports coverage collapse / panel size change when each biomarker is removed.
    
    Analogous to ALIN's ablation conditions (no_omnipath, no_perturbation, etc.)
    """
    full_solver = DiagnosticPanelSolver()
    full_solutions = full_solver.solve(tau=tau, max_size=8)

    if not full_solutions:
        logger.warning("No solutions found for full panel")
        return pd.DataFrame()

    full_best = full_solutions[0]

    results = []
    for removed_biomarker in BIOMARKERS:
        # Build reduced coverage matrix without this biomarker
        reduced_matrix = build_coverage_matrix().drop(columns=[removed_biomarker])
        reduced_solver = DiagnosticPanelSolver(coverage_matrix=reduced_matrix)

        # Need to also register the targets for the remaining biomarkers
        reduced_solver.targets = {
            b: reduced_solver.targets[b]
            for b in reduced_matrix.columns
            if b in reduced_solver.targets
        }

        reduced_solutions = reduced_solver.solve(tau=tau, max_size=8)

        if reduced_solutions:
            reduced_best = reduced_solutions[0]
            results.append({
                'removed': removed_biomarker,
                'full_panel_size': len(full_best),
                'full_coverage': full_best.coverage,
                'reduced_panel_size': len(reduced_best),
                'reduced_coverage': reduced_best.coverage,
                'coverage_change': reduced_best.coverage - full_best.coverage,
                'size_change': len(reduced_best) - len(full_best),
                'new_gaps': ', '.join(sorted(
                    reduced_best.pathologies_uncovered - full_best.pathologies_uncovered
                )) or 'None',
                'reduced_panel': ', '.join(sorted(reduced_best.biomarkers)),
                'in_optimal': removed_biomarker in full_best.biomarkers,
            })
        else:
            results.append({
                'removed': removed_biomarker,
                'full_panel_size': len(full_best),
                'full_coverage': full_best.coverage,
                'reduced_panel_size': np.nan,
                'reduced_coverage': 0.0,
                'coverage_change': -full_best.coverage,
                'size_change': np.nan,
                'new_gaps': 'ALL',
                'reduced_panel': 'NO FEASIBLE PANEL',
                'in_optimal': removed_biomarker in full_best.biomarkers,
            })

    return pd.DataFrame(results)


# ─── Threshold sensitivity analysis ─────────────────────────────────────────

def threshold_sensitivity(
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    How does the minimum panel size change as τ varies from 0.80 to 0.95?
    """
    if thresholds is None:
        thresholds = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]

    solver = DiagnosticPanelSolver()
    results = []

    for tau in thresholds:
        solutions = solver.solve(tau=tau, max_size=8)
        if solutions:
            best = solutions[0]
            results.append({
                'threshold': tau,
                'min_panel_size': len(best),
                'coverage': best.coverage,
                'optimal_panel': ', '.join(sorted(best.biomarkers)),
                'cost_eur': best.total_cost_eur,
                'worst_case_sensitivity': best.worst_case_sensitivity,
                'n_uncovered': len(best.pathologies_uncovered),
            })
        else:
            results.append({
                'threshold': tau,
                'min_panel_size': np.nan,
                'coverage': 0.0,
                'optimal_panel': 'INFEASIBLE',
                'cost_eur': np.nan,
                'worst_case_sensitivity': np.nan,
                'n_uncovered': len(PATHOLOGIES),
            })

    return pd.DataFrame(results)


# ─── Bootstrap robustness ───────────────────────────────────────────────────

def bootstrap_panel_stability(
    n_bootstrap: int = 1000,
    tau: float = 0.90,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap across sensitivity confidence intervals to assess panel stability.
    
    For each bootstrap iteration:
      1. Sample each C[p,b] from Beta(α, β) fitted to point estimate + CI
      2. Solve the MDP on the perturbed matrix
      3. Record the optimal panel
    
    Returns frequency of each panel composition and stability metrics.
    """
    rng = np.random.RandomState(seed)
    alpha_df, beta_df = build_beta_parameters()

    panel_counts = {}
    biomarker_counts = {b: 0 for b in BIOMARKERS}
    panel_sizes = []
    coverage_values = []

    for _ in range(n_bootstrap):
        # Sample sensitivities from Beta distributions
        sampled = np.zeros_like(alpha_df.values)
        for i in range(sampled.shape[0]):
            for j in range(sampled.shape[1]):
                a = alpha_df.values[i, j]
                b = beta_df.values[i, j]
                sampled[i, j] = rng.beta(a, b)
        perturbed = pd.DataFrame(
            sampled, index=PATHOLOGIES, columns=BIOMARKERS
        ).clip(0.0, 1.0)

        solver = DiagnosticPanelSolver(coverage_matrix=perturbed)
        solutions = solver.solve(tau=tau, max_size=8, min_coverage=0.5)

        if solutions:
            best = solutions[0]
            key = tuple(sorted(best.biomarkers))
            panel_counts[key] = panel_counts.get(key, 0) + 1
            panel_sizes.append(len(best))
            coverage_values.append(best.coverage)
            for b in best.biomarkers:
                biomarker_counts[b] += 1

    # Sort panels by frequency
    sorted_panels = sorted(panel_counts.items(), key=lambda x: -x[1])

    return {
        'n_bootstrap': n_bootstrap,
        'tau': tau,
        'panel_frequency': [
            {'panel': list(p), 'count': c, 'frequency': c / n_bootstrap}
            for p, c in sorted_panels[:20]
        ],
        'biomarker_inclusion_rate': {
            b: count / n_bootstrap for b, count in biomarker_counts.items()
        },
        'panel_size_mean': float(np.mean(panel_sizes)) if panel_sizes else None,
        'panel_size_std': float(np.std(panel_sizes)) if panel_sizes else None,
        'coverage_mean': float(np.mean(coverage_values)) if coverage_values else None,
        'coverage_std': float(np.std(coverage_values)) if coverage_values else None,
        'most_stable_panel': list(sorted_panels[0][0]) if sorted_panels else None,
        'stability_score': sorted_panels[0][1] / n_bootstrap if sorted_panels else 0.0,
    }


# ─── Early presenter subgroup ──────────────────────────────────────────────

def early_presenter_analysis(tau: float = 0.90) -> Dict:
    """
    Compare optimal panels for standard vs early (<2h) presenters.
    Tests the hypothesis that copeptin/H-FABP become essential for early presenters.
    """
    # Standard matrix
    standard_solver = DiagnosticPanelSolver()
    standard_solutions = standard_solver.solve(tau=tau, max_size=8)

    # Early matrix
    early_matrix = build_early_coverage_matrix()
    early_solver = DiagnosticPanelSolver(coverage_matrix=early_matrix)
    early_solutions = early_solver.solve(tau=tau, max_size=8)

    result = {
        'tau': tau,
        'standard': None,
        'early_presenter': None,
        'panel_changed': False,
        'new_markers_needed': [],
        'markers_removed': [],
    }

    if standard_solutions:
        best_std = standard_solutions[0]
        result['standard'] = {
            'panel': sorted(best_std.biomarkers),
            'size': len(best_std),
            'coverage': best_std.coverage,
            'cost_eur': best_std.total_cost_eur,
            'worst_case_sensitivity': best_std.worst_case_sensitivity,
        }

    if early_solutions:
        best_early = early_solutions[0]
        result['early_presenter'] = {
            'panel': sorted(best_early.biomarkers),
            'size': len(best_early),
            'coverage': best_early.coverage,
            'cost_eur': best_early.total_cost_eur,
            'worst_case_sensitivity': best_early.worst_case_sensitivity,
        }

    if standard_solutions and early_solutions:
        std_set = set(standard_solutions[0].biomarkers)
        early_set = set(early_solutions[0].biomarkers)
        result['panel_changed'] = std_set != early_set
        result['new_markers_needed'] = sorted(early_set - std_set)
        result['markers_removed'] = sorted(std_set - early_set)

    return result


def copeptin_threshold_analysis(
    tau: float = 0.90,
    troponin_sensitivities: Optional[List[float]] = None,
) -> Dict:
    """
    Copeptin threshold sensitivity analysis (motivated by Mu et al. 2023).

    The Mu et al. meta-analysis (13 studies, 8,966 patients) shows copeptin's
    incremental value depends on which hs-cTn threshold is used:
      - At the 99th percentile threshold: copeptin improves sensitivity 0.89→0.96
      - At the limit of detection (LoD) threshold: hs-cTn alone already 0.98;
        copeptin adds negligible benefit.

    This function sweeps hs-cTnI sensitivity for ACS across a range of values
    (simulating different troponin thresholds/analytical platforms) and re-solves
    the MHS problem at each. Two scenarios are modelled:
      (A) Standard copeptin sensitivity (0.85) — the baseline scenario
      (B) Early-presenter copeptin sensitivity (0.90) — simulating acute
          presentations where copeptin peaks early

    Returns a dictionary with sweep results for both scenarios, the crossover
    threshold, and clinical interpretation.
    """
    if troponin_sensitivities is None:
        troponin_sensitivities = [0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.82,
                                   0.80, 0.78, 0.75, 0.70, 0.65, 0.60]

    scenarios = {
        'standard_copeptin': 0.85,     # baseline copeptin sensitivity for ACS
        'early_copeptin': 0.90,        # early-presenter copeptin (peaks <1h)
    }

    all_sweeps = {}
    copeptin_entry_thresholds = {}

    for scenario_name, copeptin_acs_sensitivity in scenarios.items():
        sweeps = []
        entry_threshold = None

        for tn_sens in troponin_sensitivities:
            matrix = build_coverage_matrix()
            matrix.loc["ACS (STEMI/NSTEMI/UA)", "hs-cTnI"] = tn_sens
            matrix.loc["ACS (STEMI/NSTEMI/UA)", "Copeptin"] = copeptin_acs_sensitivity

            solver = DiagnosticPanelSolver(coverage_matrix=matrix)
            solutions = solver.solve(tau=tau, max_size=8)

            entry = {
                'troponin_sensitivity': tn_sens,
                'copeptin_acs_sensitivity': copeptin_acs_sensitivity,
                'optimal_panel': None,
                'copeptin_selected': False,
                'hs_ctni_selected': False,
                'panel_size': 0,
                'coverage': 0.0,
                'cost_eur': 0.0,
                'acs_covered': False,
                'acs_covered_by': [],
            }

            if solutions:
                best = solutions[0]
                entry['optimal_panel'] = sorted(best.biomarkers)
                entry['copeptin_selected'] = 'Copeptin' in best.biomarkers
                entry['hs_ctni_selected'] = 'hs-cTnI' in best.biomarkers
                entry['panel_size'] = len(best)
                entry['coverage'] = best.coverage
                entry['cost_eur'] = best.total_cost_eur
                # Check which markers cover ACS
                acs_coverers = [
                    b for b in best.biomarkers
                    if matrix.loc["ACS (STEMI/NSTEMI/UA)", b] >= tau
                ]
                entry['acs_covered'] = len(acs_coverers) > 0
                entry['acs_covered_by'] = acs_coverers

                if entry['copeptin_selected'] and entry_threshold is None:
                    entry_threshold = tn_sens

            sweeps.append(entry)

        all_sweeps[scenario_name] = sweeps
        copeptin_entry_thresholds[scenario_name] = entry_threshold

    # Determine clinical interpretation
    std_thresh = copeptin_entry_thresholds.get('standard_copeptin')
    early_thresh = copeptin_entry_thresholds.get('early_copeptin')

    if early_thresh is not None and std_thresh is None:
        interpretation = (
            f"Under standard conditions (copeptin ACS sensitivity 0.85), "
            f"copeptin never enters the optimal panel—it remains below τ={tau:.2f}. "
            f"However, in early-presenter settings (copeptin ACS sensitivity 0.90), "
            f"copeptin enters the panel when hs-cTnI sensitivity drops to "
            f"{early_thresh:.2f}. This confirms the Mu et al. 2023 finding: "
            f"copeptin's value is conditional on both the troponin threshold "
            f"and the clinical timing. In early presentations where copeptin "
            f"is at its peak sensitivity and troponin has not yet risen, "
            f"copeptin provides essential ACS coverage that hs-cTnI cannot."
        )
    elif early_thresh is not None and std_thresh is not None:
        interpretation = (
            f"Copeptin enters the panel at hs-cTnI={std_thresh:.2f} under "
            f"standard conditions and at hs-cTnI={early_thresh:.2f} in "
            f"early-presenter settings. The earlier entry in the early "
            f"scenario reflects copeptin's elevated sensitivity at <2h."
        )
    elif std_thresh is not None:
        interpretation = (
            f"Copeptin enters the panel at hs-cTnI={std_thresh:.2f} even "
            f"under standard conditions."
        )
    else:
        interpretation = (
            f"Copeptin does not enter the optimal panel in either scenario at "
            f"τ={tau:.2f}. When hs-cTnI drops below threshold, ACS becomes "
            f"uncoverable under standard conditions (copeptin 0.85 < τ). "
            f"In early-presenter settings (copeptin 0.90 ≥ τ), copeptin "
            f"provides the only ACS coverage pathway."
        )

    return {
        'tau': tau,
        'scenarios': all_sweeps,
        'copeptin_entry_thresholds': copeptin_entry_thresholds,
        'clinical_interpretation': interpretation,
    }


# ─── Main runner ─────────────────────────────────────────────────────────────

# ─── Weight sensitivity analysis ────────────────────────────────────────────

def weight_sensitivity_analysis(
    tau: float = 0.90,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Prove that penalty weights are irrelevant at τ=0.90 (unique feasible solution).

    Randomises (w_size, w_cost, w_time, w_sample) across orders of magnitude
    and re-solves the ILP each time.  If the panel never changes, the weights
    are formally irrelevant — the optimum is constraint-driven, not
    objective-driven.

    Also performs a structured grid sweep for a concise summary table.
    """
    rng = np.random.RandomState(seed)
    solver = DiagnosticPanelSolver()

    # ── Random sweep ──
    panel_counts: Dict[tuple, int] = {}
    for _ in range(n_samples):
        ws = 10 ** rng.uniform(-1, 3)   # w_size ∈ [0.1, 1000]
        wc = 10 ** rng.uniform(-2, 2)   # w_cost ∈ [0.01, 100]
        wt = 10 ** rng.uniform(-2, 2)
        wv = 10 ** rng.uniform(-3, 1)
        solutions = solver.solve(tau=tau, max_size=8,
                                 w_size=ws, w_cost=wc, w_time=wt, w_sample=wv)
        if solutions:
            key = tuple(sorted(solutions[0].biomarkers))
            panel_counts[key] = panel_counts.get(key, 0) + 1

    sorted_panels = sorted(panel_counts.items(), key=lambda x: -x[1])
    dominant_panel = list(sorted_panels[0][0]) if sorted_panels else []
    dominant_frac = sorted_panels[0][1] / n_samples if sorted_panels else 0

    # ── Structured grid sweep ──
    grid_weights = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    grid_results = []
    for wc in grid_weights:
        for wt in grid_weights:
            solutions = solver.solve(tau=tau, max_size=8,
                                     w_size=10.0, w_cost=wc, w_time=wt, w_sample=0.1)
            if solutions:
                best = solutions[0]
                grid_results.append({
                    'w_cost': wc, 'w_time': wt,
                    'panel': sorted(best.biomarkers),
                    'size': len(best),
                    'cost_eur': best.total_cost_eur,
                })

    n_unique_grid = len(set(
        tuple(r['panel']) for r in grid_results
    ))

    return {
        'tau': tau,
        'n_random_samples': n_samples,
        'random_panel_frequency': [
            {'panel': list(p), 'count': c, 'frequency': c / n_samples}
            for p, c in sorted_panels
        ],
        'dominant_panel': dominant_panel,
        'dominant_fraction': dominant_frac,
        'weights_irrelevant': dominant_frac == 1.0,
        'grid_sweep': grid_results[:20],
        'n_unique_grid_panels': n_unique_grid,
    }


# ─── Solution uniqueness / feasibility landscape ────────────────────────────

def feasibility_landscape(
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    For each τ, count:
      - coverable pathologies
      - number of feasible panels (covering all coverable pathologies)
      - number of distinct optimal panels (minimum-size feasible)
      - whether choice is non-trivial (>1 optimal panel)

    This quantifies the optimisation-space complexity honestly.
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]

    solver = DiagnosticPanelSolver()
    C = solver.C
    results = []

    for tau in thresholds:
        binary = (C >= tau).astype(int)
        # Coverable pathologies
        coverable = set()
        for p in C.index:
            if binary.loc[p].max() >= 1:
                coverable.add(p)
        n_coverable = len(coverable)

        # Enumerate all feasible panels
        feasible_panels = []
        min_size = None
        for k in range(1, len(BIOMARKERS) + 1):
            for combo in combinations(BIOMARKERS, k):
                combo_set = set(combo)
                covered = set()
                for p in coverable:
                    if any(binary.loc[p, b] == 1 for b in combo_set):
                        covered.add(p)
                if covered == coverable:
                    feasible_panels.append(combo_set)
                    if min_size is None:
                        min_size = k

        n_feasible = len(feasible_panels)
        # Optimal = minimum-size feasible
        optimal_panels = [p for p in feasible_panels if len(p) == min_size] if min_size else []
        n_optimal = len(optimal_panels)

        # For each coverable pathology, count how many biomarkers cover it
        choices_per_pathology = {}
        for p in coverable:
            covering_bs = [b for b in BIOMARKERS if binary.loc[p, b] == 1]
            choices_per_pathology[PATHOLOGY_SHORT.get(p, p)] = len(covering_bs)

        results.append({
            'threshold': tau,
            'n_coverable': n_coverable,
            'n_feasible_panels': n_feasible,
            'min_panel_size': min_size if min_size else np.nan,
            'n_optimal_panels': n_optimal,
            'choice_nontrivial': n_optimal > 1,
            'optimal_panels': [sorted(p) for p in optimal_panels[:5]],
            'biomarker_choices_per_pathology': choices_per_pathology,
        })

    return pd.DataFrame(results)


# ─── Clinical utility analysis ──────────────────────────────────────────────

def clinical_utility_analysis(tau: float = 0.90) -> Dict:
    """
    Score the optimal panel and all Pareto-optimal panels using the full
    clinical utility framework: specificity, prevalence weighting, severity
    weighting, expected detections, net benefit (Vickers').

    Returns dict with scored panels and a rule-out vs diagnosis distinction.
    """
    solver = DiagnosticPanelSolver()
    all_panels = solver.enumerate_all_panels(tau=tau)

    # Score each panel
    scored = []
    for panel in all_panels:
        solver.score_panel(panel, tau=tau)
        scored.append(panel)

    # Find optimal (max coverage, min cost)
    scored.sort(key=lambda p: (-p.coverage, p.total_cost_eur))
    optimal = scored[0] if scored else None

    # Build specificity table for the optimal panel
    spec_matrix = build_specificity_matrix()
    prevalence = get_prevalence_weights()
    severity = get_severity_weights()

    panel_spec_details = []
    if optimal:
        for p in PATHOLOGIES:
            best_b = max(optimal.biomarkers, key=lambda b: solver.C.loc[p, b])
            sens = solver.C.loc[p, best_b]
            spec = spec_matrix.loc[p, best_b]
            prev = prevalence[p]
            sev = severity[p]
            ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev)) if (sens * prev + (1 - spec) * (1 - prev)) > 0 else 0
            npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev) if (spec * (1 - prev) + (1 - sens) * prev) > 0 else 0
            panel_spec_details.append({
                'pathology': PATHOLOGY_SHORT.get(p, p),
                'best_biomarker': best_b,
                'sensitivity': round(sens, 3),
                'specificity': round(spec, 3),
                'prevalence': prev,
                'case_fatality': sev,
                'ppv': round(ppv, 4),
                'npv': round(npv, 4),
                'rule_out_suitable': sens >= tau and npv >= 0.99,
                'diagnosis_suitable': ppv >= 0.50,
            })

    # ── Panel-level combined false-positive rate (assuming independence) ──
    # A healthy patient gets referred if ANY of the 4 physical tests is
    # positive.  D-dimer covers both PE and AoD but is ONE test — its
    # specificity must be counted once, not twice.
    # For each unique test, use the *minimum* specificity across the
    # pathologies it covers (worst-case FP for that test).
    test_min_spec: Dict[str, float] = {}
    per_pathology_specs_covered: Dict[str, float] = {}
    for detail in panel_spec_details:
        if detail['sensitivity'] >= tau:  # only covered pathologies
            b = detail['best_biomarker']
            per_pathology_specs_covered[detail['pathology']] = detail['specificity']
            if b not in test_min_spec or detail['specificity'] < test_min_spec[b]:
                test_min_spec[b] = detail['specificity']

    # P(all 4 tests neg | healthy) = prod of per-test specificities
    panel_all_neg_prob = 1.0
    for spec_val in test_min_spec.values():
        panel_all_neg_prob *= spec_val
    panel_fp_rate = 1.0 - panel_all_neg_prob  # P(>=1 FP | no disease)

    # ── Joint NPV: P(no disease | all tests negative) ──
    # Using Bayes' with independence across pathologies:
    total_prev = sum(prevalence[p] for p in PATHOLOGIES)
    p_healthy = 1.0 - total_prev
    # P(all neg | healthy) = product of per-test specificities (computed above)
    p_all_neg_given_healthy = panel_all_neg_prob
    # For each pathology: P(all tests neg | have disease_i)
    # = (1-sens_i) for that pathology's test × prod(spec_j) for other tests
    p_all_neg_given_any_disease = 0.0
    for detail in panel_spec_details:
        p = detail['pathology']
        prev_p = detail['prevalence']
        sens_p = detail['sensitivity']
        b = detail['best_biomarker']
        # P(all neg | disease p) = (1-sens_p) × prod of specs for OTHER tests
        other_test_specs = {bt: sv for bt, sv in test_min_spec.items() if bt != b}
        p_neg_this_disease = (1 - sens_p) * (
            np.prod(list(other_test_specs.values())) if other_test_specs else 1.0
        )
        p_all_neg_given_any_disease += prev_p * p_neg_this_disease

    p_all_neg = (p_healthy * p_all_neg_given_healthy
                 + p_all_neg_given_any_disease)
    joint_npv = (p_healthy * p_all_neg_given_healthy / p_all_neg
                 if p_all_neg > 0 else 0.0)

    # ── Classify sourced vs unsourced entries (3-category) ──
    sens_quality_counts = {'bib_verified': 0, 'unverified_citation': 0, 'expert_estimate': 0}
    for pathology in PATHOLOGIES:
        for biomarker in BIOMARKERS:
            src = COVERAGE_DATA[pathology][biomarker].source
            cat = classify_reference_quality(src)
            sens_quality_counts[cat] = sens_quality_counts.get(cat, 0) + 1
    n_total_sens = sum(sens_quality_counts.values())

    # Robust optimisation
    robust_solutions = solver.solve_robust(tau=tau, max_size=8)
    robust_panel = robust_solutions[0] if robust_solutions else None
    if robust_panel:
        solver_for_robust = DiagnosticPanelSolver()
        solver_for_robust.score_panel(robust_panel, tau=tau)

    return {
        'tau': tau,
        'optimal_panel': {
            'biomarkers': sorted(optimal.biomarkers) if optimal else [],
            'coverage': optimal.coverage if optimal else 0,
            'prevalence_weighted_coverage': optimal.prevalence_weighted_coverage if optimal else 0,
            'severity_weighted_coverage': optimal.severity_weighted_coverage if optimal else 0,
            'expected_detections_per_1000': optimal.expected_detections_per_1000 if optimal else 0,
            'expected_detections_covered_per_1000': optimal.expected_detections_covered_per_1000 if optimal else 0,
            'mean_specificity': optimal.mean_panel_specificity if optimal else 0,
            'net_benefit': optimal.net_benefit if optimal else 0,
            'net_benefit_by_threshold': optimal.net_benefit_by_threshold if optimal else {},
            'clinical_utility_score': optimal.clinical_utility_score if optimal else 0,
            'net_benefit_note': (
                'Net benefit computed at clinical decision thresholds (t=0.01 to 0.10), '
                'NOT at tau=0.90 which is the sensitivity selection threshold. '
                'For rule-out of life-threatening conditions, t=0.01 is appropriate '
                '(willing to accept ~99 FPs per missed case). Only covered pathologies contribute.'
            ),
        },
        'per_pathology_details': panel_spec_details,
        'panel_level_false_positive_rate': {
            'p_at_least_one_fp': round(panel_fp_rate, 4),
            'p_all_neg_given_healthy': round(panel_all_neg_prob, 4),
            'per_test_specificities': {k: round(v, 3) for k, v in test_min_spec.items()},
            'n_unique_physical_tests': len(test_min_spec),
            'note': (
                'P(>=1 false positive | no disease) assuming independent tests. '
                'D-dimer covers PE+AoD but counted once (min spec). '
                'True panel-level FP rate requires patient-level correlation data.'
            ),
        },
        'joint_npv': {
            'value': round(joint_npv, 6),
            'p_any_disease_given_all_neg': round(1 - joint_npv, 6),
            'note': (
                'P(no disease at all | all panel tests negative). '
                'Assumes independent biomarker responses across pathologies. '
                'True joint NPV requires patient-level paired data.'
            ),
        },
        'source_audit': {
            'total_sensitivity_entries': n_total_sens,
            'bib_verified': sens_quality_counts['bib_verified'],
            'unverified_citation': sens_quality_counts['unverified_citation'],
            'expert_estimate': sens_quality_counts['expert_estimate'],
            'fraction_bib_verified': round(
                sens_quality_counts['bib_verified'] / n_total_sens, 3
            ),
            'note': (
                'bib_verified = bracketed ref cross-referenced in refs.bib; '
                'unverified_citation = author-year cited but not in refs.bib (ghost citation); '
                'expert_estimate = no literature source.'
            ),
        },
        'robust_panel': {
            'biomarkers': sorted(robust_panel.biomarkers) if robust_panel else [],
            'coverage': robust_panel.coverage if robust_panel else 0,
            'same_as_point_estimate': (
                sorted(robust_panel.biomarkers) == sorted(optimal.biomarkers)
                if robust_panel and optimal else False
            ),
        },
        'rule_out_vs_diagnosis': (
            "This framework is designed for RULE-OUT (exclusion) of "
            "life-threatening pathologies, not definitive diagnosis. High "
            "sensitivity and high NPV are prioritised; low PPV is expected "
            "given the low pre-test probabilities in primary care. Positive "
            "results trigger referral, not treatment initiation."
        ),
    }


# ─── Full 48-cell source provenance table ───────────────────────────────────

def build_full_source_table() -> pd.DataFrame:
    """
    Extract all 48 sensitivity sources from COVERAGE_DATA,
    plus all 48 specificity sources from SPECIFICITY_DATA,
    producing a comprehensive provenance table for the supplement.

    Each sensitivity source is classified into one of three evidence
    categories via :func:`classify_reference_quality`:

    * **bib_verified** – has a bracketed reference number cross-referenced
      in ``refs.bib``.
    * **expert_estimate** – explicitly labelled as an expert / clinical
      estimate with no literature citation.
    * **unverified_citation** – cites an author–year publication that has
      NOT been cross-referenced to ``refs.bib`` (ghost citation).
    """
    rows = []
    for pathology in PATHOLOGIES:
        for biomarker in BIOMARKERS:
            sens_entry = COVERAGE_DATA[pathology][biomarker]
            spec_entry = SPECIFICITY_DATA[pathology][biomarker]
            sens_quality = classify_reference_quality(sens_entry.source)
            spec_quality = classify_reference_quality(spec_entry.source)
            rows.append({
                'pathology': PATHOLOGY_SHORT.get(pathology, pathology),
                'biomarker': biomarker,
                'sensitivity': sens_entry.sensitivity,
                'sens_ci_lower': sens_entry.ci_lower,
                'sens_ci_upper': sens_entry.ci_upper,
                'sens_source': sens_entry.source,
                'sens_reference_quality': sens_quality,
                'sens_setting': sens_entry.setting,
                'specificity': spec_entry.specificity,
                'spec_ci_lower': spec_entry.ci_lower,
                'spec_ci_upper': spec_entry.ci_upper,
                'spec_source': spec_entry.source,
                'spec_reference_quality': spec_quality,
            })
    df = pd.DataFrame(rows)
    n_total = len(df)
    sens_counts = df['sens_reference_quality'].value_counts()
    spec_counts = df['spec_reference_quality'].value_counts()
    print(f"  Sensitivity source audit ({n_total} entries):")
    for cat in ['bib_verified', 'unverified_citation', 'expert_estimate']:
        print(f"    {cat}: {sens_counts.get(cat, 0)}")
    print(f"  Specificity source audit ({n_total} entries):")
    for cat in ['bib_verified', 'unverified_citation', 'expert_estimate']:
        print(f"    {cat}: {spec_counts.get(cat, 0)}")
    return df


# ─── Monte Carlo CI propagation ─────────────────────────────────────────────

def monte_carlo_ci_propagation(
    n_samples: int = 5000,
    tau: float = 0.90,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo propagation of sensitivity CI uncertainty.

    For each iteration, sample each C[p,b] from Beta(α, β) fitted to the
    published point estimate and 95% CI, then compute clinical utility
    metrics for the *fixed* optimal panel (hs-cTnI + D-dimer + NT-proBNP + CRP).

    Uses Beta rather than Uniform to respect that meta-analytic pooled
    estimates have higher density near the point estimate.

    Unlike bootstrap_panel_stability (which re-solves the optimisation),
    this fixes the panel and propagates measurement uncertainty.
    """
    rng = np.random.RandomState(seed)
    alpha_df, beta_df = build_beta_parameters()
    spec_matrix = build_specificity_matrix()
    prevalence = get_prevalence_weights()
    severity = get_severity_weights()

    # Fixed optimal panel
    panel_biomarkers = ['hs-cTnI', 'D-dimer', 'NT-proBNP', 'CRP']

    coverage_samples = []
    detections_covered_samples = []
    detections_all_samples = []
    net_benefit_samples = {t: [] for t in [0.01, 0.05, 0.10]}

    for _ in range(n_samples):
        # Sample sensitivities from Beta distributions
        sampled = np.zeros_like(alpha_df.values)
        for i in range(sampled.shape[0]):
            for j in range(sampled.shape[1]):
                a = alpha_df.values[i, j]
                b = beta_df.values[i, j]
                sampled[i, j] = rng.beta(a, b)
        C_sample = pd.DataFrame(
            sampled, index=PATHOLOGIES, columns=BIOMARKERS
        ).clip(0.0, 1.0)

        # Coverage at threshold
        covered = 0
        covered_pathologies = []
        for p in PATHOLOGIES:
            if any(C_sample.loc[p, b] >= tau for b in panel_biomarkers):
                covered += 1
                covered_pathologies.append(p)
        coverage_samples.append(covered / len(PATHOLOGIES))

        # Expected detections: separate covered vs all
        expected_covered = 0.0
        expected_all = 0.0
        for p in PATHOLOGIES:
            best_sens = max(C_sample.loc[p, b] for b in panel_biomarkers)
            det = prevalence[p] * best_sens * 1000
            expected_all += det
            if p in covered_pathologies:
                expected_covered += det
        detections_covered_samples.append(expected_covered)
        detections_all_samples.append(expected_all)

        # Net benefit at multiple clinically relevant decision thresholds
        for t in [0.01, 0.05, 0.10]:
            w = t / (1 - t)
            nb = 0.0
            for p in covered_pathologies:  # only covered pathologies
                best_b = max(panel_biomarkers, key=lambda b: C_sample.loc[p, b])
                sens_p = C_sample.loc[p, best_b]
                spec_p = spec_matrix.loc[p, best_b]
                prev_p = prevalence[p]
                nb += sens_p * prev_p - (1 - spec_p) * (1 - prev_p) * w
            net_benefit_samples[t].append(nb)

    return {
        'n_samples': n_samples,
        'tau': tau,
        'panel': panel_biomarkers,
        'sampling_distribution': 'Beta (fitted from point estimate + 95% CI)',
        'coverage': {
            'mean': float(np.mean(coverage_samples)),
            'std': float(np.std(coverage_samples)),
            'ci_2.5': float(np.percentile(coverage_samples, 2.5)),
            'ci_97.5': float(np.percentile(coverage_samples, 97.5)),
            'p_full_coverage': float(np.mean([c >= 5/6 for c in coverage_samples])),
        },
        'expected_detections_covered_per_1000': {
            'mean': float(np.mean(detections_covered_samples)),
            'std': float(np.std(detections_covered_samples)),
            'ci_2.5': float(np.percentile(detections_covered_samples, 2.5)),
            'ci_97.5': float(np.percentile(detections_covered_samples, 97.5)),
        },
        'expected_detections_all_per_1000': {
            'mean': float(np.mean(detections_all_samples)),
            'std': float(np.std(detections_all_samples)),
            'ci_2.5': float(np.percentile(detections_all_samples, 2.5)),
            'ci_97.5': float(np.percentile(detections_all_samples, 97.5)),
        },
        'net_benefit': {
            f't={t}': {
                'decision_threshold': t,
                'harm_benefit_ratio': round(t / (1 - t), 4),
                'mean': float(np.mean(net_benefit_samples[t])),
                'std': float(np.std(net_benefit_samples[t])),
                'ci_2.5': float(np.percentile(net_benefit_samples[t], 2.5)),
                'ci_97.5': float(np.percentile(net_benefit_samples[t], 97.5)),
                'p_positive': float(np.mean([nb > 0 for nb in net_benefit_samples[t]])),
            }
            for t in [0.01, 0.05, 0.10]
        },
        'note': (
            'Net benefit computed at clinically relevant decision thresholds '
            '(not at tau=0.90 which is the sensitivity selection threshold, '
            'not the clinical decision threshold). '
            't=0.01 means willing to accept 99 FPs per missed case. '
            'Only covered pathologies (sens >= tau) contribute.'
        ),
    }


# ─── Main runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print("DIAGNOSTIC PANEL ANALYSIS — FULL PIPELINE")
    print("=" * 90)

    solver = DiagnosticPanelSolver()

    # ── 1. Pareto frontier ──
    print("\n[1/7] Computing Pareto frontier...")
    pareto = compute_pareto_frontier(solver, tau=0.90)
    pareto_optimal = pareto[pareto['pareto_optimal']].sort_values('cost_eur')
    print(f"  {len(pareto_optimal)} Pareto-optimal panels identified")
    print(pareto_optimal[['biomarkers', 'n_tests', 'coverage', 'cost_eur',
                           'worst_case_sensitivity']].to_string(index=False))
    pareto.to_csv(os.path.join(output_dir, "pareto_frontier.csv"), index=False)

    # ── 2. Reference approaches ──
    print("\n[2/7] Mapping current approaches on frontier...")
    refs = get_reference_approaches(solver, tau=0.90)
    print(refs.to_string(index=False))
    refs.to_csv(os.path.join(output_dir, "reference_approaches.csv"), index=False)

    # ── 3. Ablation ──
    print("\n[3/7] Running ablation analysis...")
    ablation = ablation_analysis(tau=0.90)
    print(ablation[['removed', 'in_optimal', 'coverage_change', 'new_gaps']].to_string(index=False))
    ablation.to_csv(os.path.join(output_dir, "ablation_analysis.csv"), index=False)

    # ── 4. Threshold sensitivity ──
    print("\n[4/12] Threshold sensitivity analysis...")
    thresh = threshold_sensitivity()
    print(thresh.to_string(index=False))
    thresh.to_csv(os.path.join(output_dir, "threshold_sensitivity.csv"), index=False)

    # ── 5. Bootstrap stability ──
    print("\n[5/12] Bootstrap panel stability (n=1000)...")
    bootstrap = bootstrap_panel_stability(n_bootstrap=1000, tau=0.90)
    print(f"  Most stable panel: {bootstrap['most_stable_panel']}")
    print(f"  Stability score: {bootstrap['stability_score']:.1%}")
    print(f"  Mean panel size: {bootstrap['panel_size_mean']:.1f} ± {bootstrap['panel_size_std']:.2f}")
    print(f"  Biomarker inclusion rates:")
    for b, rate in sorted(bootstrap['biomarker_inclusion_rate'].items(), key=lambda x: -x[1]):
        print(f"    {b:15s}: {rate:.1%}")
    with open(os.path.join(output_dir, "bootstrap_stability.json"), 'w') as f:
        json.dump(bootstrap, f, indent=2)

    # ── 6. Early presenter ──
    print("\n[6/12] Early presenter subgroup analysis...")
    early = early_presenter_analysis(tau=0.90)
    print(f"  Standard panel:  {early['standard']}")
    print(f"  Early panel:     {early['early_presenter']}")
    print(f"  Panel changed:   {early['panel_changed']}")
    print(f"  New markers for early: {early['new_markers_needed']}")
    with open(os.path.join(output_dir, "early_presenter_analysis.json"), 'w') as f:
        json.dump(early, f, indent=2)

    # ── 7. Copeptin threshold sensitivity ──
    print("\n[7/12] Copeptin threshold sensitivity analysis (Mu et al. 2023)...")
    copeptin_result = copeptin_threshold_analysis(tau=0.90)
    print(f"  Interpretation: {copeptin_result['clinical_interpretation']}")
    for scenario_name, sweeps in copeptin_result['scenarios'].items():
        thresh_val = copeptin_result['copeptin_entry_thresholds'].get(scenario_name)
        print(f"\n  Scenario: {scenario_name} (copeptin entry: {thresh_val})")
        for s in sweeps:
            cop_flag = " ← COPEPTIN" if s['copeptin_selected'] else ""
            acs_flag = " [ACS gap]" if not s['acs_covered'] else ""
            acs_by = f" (by {', '.join(s['acs_covered_by'])})" if s['acs_covered_by'] else ""
            print(f"    hs-cTnI={s['troponin_sensitivity']:.2f}: "
                  f"panel={s['optimal_panel']}, "
                  f"size={s['panel_size']}, "
                  f"cov={s['coverage']:.1%}, "
                  f"cost=€{s['cost_eur']:.2f}"
                  f"{acs_by}{cop_flag}{acs_flag}")
    with open(os.path.join(output_dir, "copeptin_threshold_analysis.json"), 'w') as f:
        json.dump(copeptin_result, f, indent=2)

    # ── 8. Weight sensitivity ──
    print("\n[8/12] Weight sensitivity analysis (penalty invariance)...")
    weight_result = weight_sensitivity_analysis(tau=0.90)
    print(f"  Dominant panel: {weight_result['dominant_panel']}")
    print(f"  Fraction: {weight_result['dominant_fraction']:.1%}")
    print(f"  Weights irrelevant: {weight_result['weights_irrelevant']}")
    print(f"  Unique panels in grid: {weight_result['n_unique_grid_panels']}")
    with open(os.path.join(output_dir, "weight_sensitivity.json"), 'w') as f:
        json.dump(weight_result, f, indent=2)

    # ── 9. Feasibility landscape ──
    print("\n[9/12] Feasibility landscape (solution uniqueness)...")
    landscape = feasibility_landscape()
    for _, row in landscape.iterrows():
        print(f"  τ={row['threshold']:.2f}: {row['n_coverable']} coverable, "
              f"{row['n_feasible_panels']} feasible, "
              f"{row['n_optimal_panels']} optimal (trivial={not row['choice_nontrivial']})")
    landscape_save = landscape.drop(columns=['optimal_panels', 'biomarker_choices_per_pathology'])
    landscape_save.to_csv(os.path.join(output_dir, "feasibility_landscape.csv"), index=False)
    # Save full detail as JSON
    with open(os.path.join(output_dir, "feasibility_landscape.json"), 'w') as f:
        json.dump(landscape.to_dict(orient='records'), f, indent=2, default=str)

    # ── 10. Clinical utility analysis ──
    print("\n[10/12] Clinical utility analysis (specificity, net benefit)...")
    utility = clinical_utility_analysis(tau=0.90)
    print(f"  Optimal: {utility['optimal_panel']['biomarkers']}")
    print(f"  Prevalence-weighted cov: {utility['optimal_panel']['prevalence_weighted_coverage']:.4f}")
    print(f"  Severity-weighted cov: {utility['optimal_panel']['severity_weighted_coverage']:.2f}")
    print(f"  Expected detections/1000: {utility['optimal_panel']['expected_detections_per_1000']:.1f}")
    print(f"  Mean specificity: {utility['optimal_panel']['mean_specificity']:.3f}")
    print(f"  Net benefit: {utility['optimal_panel']['net_benefit']:.4f}")
    print(f"  Robust panel same: {utility['robust_panel']['same_as_point_estimate']}")
    print(f"\n  Per-pathology:")
    for d in utility['per_pathology_details']:
        print(f"    {d['pathology']:10s}: sens={d['sensitivity']:.2f} spec={d['specificity']:.2f} "
              f"PPV={d['ppv']:.3f} NPV={d['npv']:.4f} "
              f"rule-out={'✓' if d['rule_out_suitable'] else '✗'} "
              f"dx={'✓' if d['diagnosis_suitable'] else '✗'}")
    with open(os.path.join(output_dir, "clinical_utility.json"), 'w') as f:
        json.dump(utility, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    # ── 11. Full source provenance table ──
    print("\n[11/12] Building full 48-cell source provenance table...")
    source_table = build_full_source_table()
    source_table.to_csv(os.path.join(output_dir, "full_source_provenance.csv"), index=False)
    print(f"  {len(source_table)} entries written")

    # ── 12. Monte Carlo CI propagation ──
    print("\n[12/12] Monte Carlo CI propagation (n=5000)...")
    mc = monte_carlo_ci_propagation(n_samples=5000, tau=0.90)
    print(f"  Coverage: {mc['coverage']['mean']:.3f} "
          f"[{mc['coverage']['ci_2.5']:.3f}, {mc['coverage']['ci_97.5']:.3f}]")
    print(f"  P(full coverage): {mc['coverage']['p_full_coverage']:.1%}")
    print(f"  Detections/1000: {mc['expected_detections_per_1000']['mean']:.1f} "
          f"± {mc['expected_detections_per_1000']['std']:.1f}")
    print(f"  Net benefit: {mc['net_benefit']['mean']:.4f} "
          f"[{mc['net_benefit']['ci_2.5']:.4f}, {mc['net_benefit']['ci_97.5']:.4f}]")
    with open(os.path.join(output_dir, "monte_carlo_ci.json"), 'w') as f:
        json.dump(mc, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"All results saved to {output_dir}/")
    print(f"{'=' * 90}")
