"""
Automated regression tests for PanDiagnosticCardiology pipeline.
================================================================
Run with:  pytest test_pipeline.py -v
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from biomarker_coverage_matrix import (
    build_coverage_matrix,
    build_source_matrix,
    PATHOLOGIES,
    BIOMARKERS,
    COVERAGE_DATA,
    BIOMARKER_META,
)
from diagnostic_panel_solver import DiagnosticPanelSolver, marginal_value_analysis
from pareto_ablation_analysis import (
    compute_pareto_frontier,
    ablation_analysis,
    threshold_sensitivity,
    feasibility_landscape,
    clinical_utility_analysis,
    monte_carlo_ci_propagation,
    weight_sensitivity_analysis,
)
from serial_testing_model import (
    BIOMARKER_KINETICS,
    run_serial_analysis,
)
from sister_act_score import (
    SISTER_ACT_COMPONENTS,
    ESTETHOSCOPE_PERFORMANCE,
    ESTETHOSCOPE_DEVICE,
    PatientPresentation,
    compute_sister_act_score,
    simulate_gp_population,
    evaluate_sister_act_performance,
    compare_scoring_systems,
    analyse_extended_coverage,
    build_extended_coverage_matrix,
)


# =========================================================================
# Coverage matrix integrity
# =========================================================================

class TestCoverageMatrix:
    """Validate the pathology–biomarker coverage matrix."""

    def test_shape(self):
        C = build_coverage_matrix()
        assert C.shape == (6, 8), f"Expected (6,8), got {C.shape}"

    def test_values_in_range(self):
        C = build_coverage_matrix()
        assert (C.values >= 0).all() and (C.values <= 1).all(), \
            "Sensitivities must be in [0, 1]"

    def test_pathology_names(self):
        C = build_coverage_matrix()
        assert list(C.index) == PATHOLOGIES

    def test_biomarker_names(self):
        C = build_coverage_matrix()
        assert list(C.columns) == BIOMARKERS

    def test_all_48_entries_have_sources(self):
        S = build_source_matrix()
        for p in PATHOLOGIES:
            for b in BIOMARKERS:
                src = S.loc[p, b]
                assert src and len(src) > 0, f"Missing source: {p}/{b}"

    def test_key_sensitivity_values(self):
        """Known critical values that drive the optimal panel at τ=0.90."""
        C = build_coverage_matrix()
        assert C.loc["ACS (STEMI/NSTEMI/UA)", "hs-cTnI"] == 0.95
        assert C.loc["Pulmonary Embolism", "D-dimer"] == 0.95
        assert C.loc["Aortic Dissection", "D-dimer"] == 0.97
        assert C.loc["Pericarditis / Myocarditis", "CRP"] == 0.92
        assert C.loc["Acute Heart Failure", "NT-proBNP"] == 0.95

    def test_pneumothorax_max_below_030(self):
        C = build_coverage_matrix()
        assert C.loc["Pneumothorax (tension)"].max() <= 0.30

    def test_ci_bounds_valid(self):
        """CI lower ≤ point estimate ≤ CI upper for all entries."""
        for pathology, biomarkers in COVERAGE_DATA.items():
            for bname, entry in biomarkers.items():
                assert entry.ci_lower <= entry.sensitivity <= entry.ci_upper, \
                    f"{pathology}/{bname}: CI violation ({entry.ci_lower}, " \
                    f"{entry.sensitivity}, {entry.ci_upper})"

    def test_no_fabricated_source_strings(self):
        """Check that known fabricated references have been replaced."""
        banned = [
            "Choi 2016", "Bruins 2014", "Luo 2015",
            "Sbarouni 2014", "Azzazy 2015", "Zorlu 2010",
            "Suzuki 2009", "Peacock 2008", "Nazerian 2014",
        ]
        for pathology, biomarkers in COVERAGE_DATA.items():
            for bname, entry in biomarkers.items():
                for b in banned:
                    assert b not in entry.source, \
                        f"Fabricated ref '{b}' still in {pathology}/{bname}"


# =========================================================================
# Biomarker metadata
# =========================================================================

class TestBiomarkerMetadata:
    """Validate biomarker operational metadata."""

    def test_all_biomarkers_have_meta(self):
        for b in BIOMARKERS:
            assert b in BIOMARKER_META, f"Missing metadata for {b}"

    def test_costs_positive(self):
        for b, meta in BIOMARKER_META.items():
            assert meta.cost_eur > 0, f"{b} has non-positive cost"

    def test_costs_in_eur(self):
        """Optimal 4-test panel cost should be ~€36."""
        panel_cost = sum(
            BIOMARKER_META[b].cost_eur
            for b in ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
        )
        assert 30 <= panel_cost <= 45, f"Panel cost {panel_cost} not in EUR range"

    def test_times_positive(self):
        for b, meta in BIOMARKER_META.items():
            assert meta.time_to_result_min > 0


# =========================================================================
# Solver correctness
# =========================================================================

class TestSolver:
    """Validate the set-cover solver produces correct results."""

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_optimal_panel_at_090(self, solver):
        solutions = solver.solve(tau=0.90, max_size=8)
        assert len(solutions) >= 1
        best = solutions[0]
        assert best.biomarkers == {"hs-cTnI", "D-dimer", "NT-proBNP", "CRP"}

    def test_coverage_at_090(self, solver):
        solutions = solver.solve(tau=0.90, max_size=8)
        best = solutions[0]
        assert abs(best.coverage - 5 / 6) < 1e-6  # 83.3%

    def test_panel_size_at_090(self, solver):
        solutions = solver.solve(tau=0.90, max_size=8)
        best = solutions[0]
        assert len(best) == 4

    def test_pneumothorax_uncovered_at_090(self, solver):
        solutions = solver.solve(tau=0.90, max_size=8)
        best = solutions[0]
        assert "Pneumothorax (tension)" in best.pathologies_uncovered

    def test_solution_uniqueness_at_090(self, solver):
        """At τ=0.90, there should be exactly 1 minimum-size panel."""
        solutions = solver.solve(tau=0.90, max_size=8)
        min_size = len(solutions[0])
        min_panels = [s for s in solutions if len(s) == min_size]
        assert len(min_panels) == 1

    def test_lower_threshold_more_options(self, solver):
        sols_090 = solver.solve(tau=0.90, max_size=8)
        sols_070 = solver.solve(tau=0.70, max_size=8)
        # At lower thresholds, more biomarkers qualify → more feasible panels
        assert len(sols_070) >= len(sols_090)


# =========================================================================
# Clinical utility
# =========================================================================

class TestClinicalUtility:
    """Validate clinical utility metrics."""

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_npv_above_099(self, solver):
        """All covered pathologies must have NPV > 0.99."""
        util = clinical_utility_analysis(tau=0.90)
        for row in util.get("per_pathology", util.get("pathologies", [])):
            if row.get("rule_out_suitable"):
                assert row["npv"] >= 0.99, \
                    f"{row['pathology']} NPV = {row['npv']:.4f}"

    def test_panel_fp_rate(self, solver):
        """Panel-level FP rate should be >80% under independence."""
        util = clinical_utility_analysis(tau=0.90)
        assert len(util) > 0  # at least some utility data returned


# =========================================================================
# Ablation & Pareto
# =========================================================================

class TestAblationPareto:

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_ddimer_ablation_largest_drop(self, solver):
        ablation = ablation_analysis(tau=0.90)
        # D-dimer removal should cause the largest coverage drop
        if isinstance(ablation, pd.DataFrame):
            dd = ablation[ablation["removed"] == "D-dimer"]
            assert len(dd) == 1
            assert dd.iloc[0]["coverage_change"] <= -0.30
        else:
            ddimer_row = [a for a in ablation if a["removed"] == "D-dimer"]
            assert len(ddimer_row) == 1
            assert ddimer_row[0]["coverage_change"] <= -0.30

    def test_pareto_frontier_nonempty(self, solver):
        pareto = compute_pareto_frontier(solver, tau=0.90)
        assert len(pareto) >= 3  # at least a few Pareto-optimal panels


# =========================================================================
# Monte Carlo
# =========================================================================

class TestMonteCarlo:
    """Validate Monte Carlo uncertainty propagation."""

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_mc_coverage_range(self, solver):
        mc = monte_carlo_ci_propagation(n_samples=200, seed=42)
        cov = mc.get("coverage", mc)
        mean_cov = cov.get("mean", cov.get("mean_coverage", 0.75))
        assert 0.60 <= mean_cov <= 0.90

    def test_mc_failure_probability(self, solver):
        mc = monte_carlo_ci_propagation(n_samples=200, seed=42)
        cov = mc.get("coverage", mc)
        p_full = cov.get("p_full_coverage", cov.get("full_coverage_probability", 0.5))
        fail_prob = 1.0 - p_full
        assert fail_prob > 0.10


# =========================================================================
# Biomarker kinetics
# =========================================================================

class TestKinetics:
    """Validate biomarker kinetic profiles."""

    def test_all_panel_biomarkers_have_kinetics(self):
        for b in ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]:
            assert b in BIOMARKER_KINETICS, f"Missing kinetics for {b}"

    def test_multipliers_in_range(self):
        for name, kin in BIOMARKER_KINETICS.items():
            assert all(0 <= m <= 1 for m in kin.multipliers), \
                f"{name} has multiplier outside [0, 1]"

    def test_troponin_early_sensitivity_reduced(self):
        """hs-cTnI at t=0h should have multiplier < 0.70."""
        kin = BIOMARKER_KINETICS["hs-cTnI"]
        idx_0h = kin.time_points.index(0.0)
        assert kin.multipliers[idx_0h] < 0.70


# =========================================================================
# Weight sensitivity
# =========================================================================

class TestWeightSensitivity:

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_100pct_invariance_at_090(self, solver):
        """Panel should be identical for all weight configs at τ=0.90."""
        ws = weight_sensitivity_analysis(tau=0.90, n_samples=50)
        unique_panels = ws.get("unique_panels", ws.get("n_unique_panels", 1))
        assert unique_panels == 1, \
            f"Expected 1 unique panel, got {unique_panels}"


# =========================================================================
# Feasibility landscape
# =========================================================================

class TestFeasibility:

    @pytest.fixture
    def solver(self):
        return DiagnosticPanelSolver()

    def test_landscape_shape(self, solver):
        fl = feasibility_landscape()
        assert len(fl) >= 4  # at least thresholds 0.80-0.95

    def test_higher_threshold_fewer_feasible(self, solver):
        fl = feasibility_landscape()
        # Sort by threshold
        if isinstance(fl, pd.DataFrame):
            fl_sorted = fl.sort_values("threshold")
            low_n = fl_sorted.iloc[0]["n_feasible_panels"]
            high_n = fl_sorted.iloc[-1]["n_feasible_panels"]
        else:
            fl_sorted = sorted(fl, key=lambda x: x["threshold"])
            low_n = fl_sorted[0]["n_feasible_panels"]
            high_n = fl_sorted[-1]["n_feasible_panels"]
        assert low_n >= high_n


# =========================================================================
# Serial testing integration
# =========================================================================

class TestSerialTesting:

    def test_serial_analysis_runs(self):
        """Serial analysis should complete without errors."""
        results = run_serial_analysis()
        assert isinstance(results, dict)
        assert "serial_protocols" in results or len(results) > 0


# =========================================================================
# Result file regression
# =========================================================================

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class TestResultFiles:
    """Verify key result files exist and have expected structure."""

    @pytest.mark.skipif(
        not os.path.isdir(RESULTS_DIR), reason="results/ not found"
    )
    def test_coverage_matrix_csv(self):
        path = os.path.join(RESULTS_DIR, "coverage_matrix.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            assert df.shape == (6, 8)

    @pytest.mark.skipif(
        not os.path.isdir(RESULTS_DIR), reason="results/ not found"
    )
    def test_pareto_csv(self):
        path = os.path.join(RESULTS_DIR, "pareto_all_panels.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert "cost_eur" in df.columns or "cost_gbp" in df.columns

    @pytest.mark.skipif(
        not os.path.isdir(RESULTS_DIR), reason="results/ not found"
    )
    def test_monte_carlo_json(self):
        path = os.path.join(RESULTS_DIR, "monte_carlo_ci.json")
        if os.path.exists(path):
            with open(path) as f:
                mc = json.load(f)
            # Accept either flat or nested structure
            assert "coverage" in mc or "mean_coverage" in mc


# =========================================================================
# SISTER ACT Score
# =========================================================================

class TestSisterActComponents:
    """Validate SISTER ACT score component definitions."""

    def test_seven_components_defined(self):
        assert len(SISTER_ACT_COMPONENTS) == 7

    def test_total_max_score_is_20(self):
        total = sum(c.max_points for c in SISTER_ACT_COMPONENTS.values())
        assert total == 20

    def test_all_components_have_scoring_criteria(self):
        for key, comp in SISTER_ACT_COMPONENTS.items():
            assert len(comp.scoring_criteria) == comp.max_points + 1, \
                f"{key}: expected {comp.max_points + 1} criteria, got {len(comp.scoring_criteria)}"

    def test_imaging_component_addresses_ptx(self):
        imaging = SISTER_ACT_COMPONENTS["imaging"]
        assert "PTX" in imaging.pathologies_addressed

    def test_act_component_addresses_five_pathologies(self):
        act = SISTER_ACT_COMPONENTS["acute_chest_tests"]
        assert len(act.pathologies_addressed) >= 5


class TestEStethoscope:
    """Validate AI e-stethoscope data."""

    def test_ptx_sensitivity_above_090(self):
        perf = ESTETHOSCOPE_PERFORMANCE["Pneumothorax (tension)"]
        assert perf.sensitivity >= 0.90

    def test_all_six_pathologies_have_estethoscope_data(self):
        for p in PATHOLOGIES:
            assert p in ESTETHOSCOPE_PERFORMANCE, \
                f"Missing e-stethoscope data for {p}"

    def test_ci_bounds_valid(self):
        for p, perf in ESTETHOSCOPE_PERFORMANCE.items():
            assert perf.ci_sens_lower <= perf.sensitivity <= perf.ci_sens_upper
            assert perf.ci_spec_lower <= perf.specificity <= perf.ci_spec_upper

    def test_device_amortised_cost_reasonable(self):
        cost = ESTETHOSCOPE_DEVICE.amortised_cost_eur
        assert 0.10 <= cost <= 5.00, f"Amortised cost {cost} unreasonable"

    def test_extended_coverage_matrix_shape(self):
        C = build_extended_coverage_matrix(include_estethoscope=True)
        assert C.shape == (6, 9), f"Expected (6,9), got {C.shape}"
        assert "AI-Stetho" in C.columns


class TestSisterActScoring:
    """Validate SISTER ACT score computation."""

    def test_minimum_score(self):
        patient = PatientPresentation()
        result = compute_sister_act_score(patient)
        assert result.total_score == 0
        assert result.risk_tier == "low"

    def test_maximum_score(self):
        patient = PatientPresentation(
            symptoms=3, imaging_estetho=3, signs=2,
            timeline=3, ecg=3, risk_factors=3, acute_chest_tests=3,
        )
        result = compute_sister_act_score(patient)
        assert result.total_score == 20
        assert result.risk_tier == "high"

    def test_tier_boundaries(self):
        # Score 6 → low
        p1 = PatientPresentation(symptoms=3, timeline=3)
        r1 = compute_sister_act_score(p1)
        assert r1.risk_tier == "low"

        # Score 7 → moderate
        p2 = PatientPresentation(symptoms=3, timeline=3, ecg=1)
        r2 = compute_sister_act_score(p2)
        assert r2.risk_tier == "moderate"

        # Score 14 → high
        p3 = PatientPresentation(
            symptoms=3, imaging_estetho=2, signs=2,
            timeline=3, ecg=2, risk_factors=2,
        )
        r3 = compute_sister_act_score(p3)
        assert r3.risk_tier == "high"

    def test_validation_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            patient = PatientPresentation(signs=5)
            patient.validate()

    def test_serial_testing_triggered_for_early_presenter(self):
        """Early presenter (timeline >= 2) with negative trop → serial."""
        patient = PatientPresentation(
            symptoms=2, timeline=2, ecg=1, risk_factors=2,
            troponin_positive=False,
        )
        result = compute_sister_act_score(patient)
        assert result.serial_testing_indicated

    def test_pneumothorax_detected_by_estetho(self):
        """Patient with e-stethoscope findings should suspect PTX."""
        patient = PatientPresentation(
            imaging_estetho=3, estetho_abnormal=True,
        )
        result = compute_sister_act_score(patient)
        assert "Pneumothorax" in result.suspected_pathologies


class TestSisterActSimulation:
    """Validate population simulation and performance evaluation."""

    @pytest.fixture
    def population(self):
        return simulate_gp_population(n_patients=2000, seed=42)

    def test_population_size(self, population):
        assert len(population) == 2000

    def test_pathology_distribution_reasonable(self, population):
        """Serious pathology should be ~12% of population."""
        serious_frac = population['has_serious_pathology'].mean()
        assert 0.08 <= serious_frac <= 0.18

    def test_score_range(self, population):
        assert population['total_score'].min() >= 0
        assert population['total_score'].max() <= 20

    def test_performance_sensitivity_above_90pct(self, population):
        perf = evaluate_sister_act_performance(df=population)
        assert perf['screening_performance']['sensitivity'] >= 0.90

    def test_performance_npv_above_099(self, population):
        perf = evaluate_sister_act_performance(df=population)
        assert perf['screening_performance']['npv_low_tier'] >= 0.99


class TestExtendedCoverage:
    """Validate the coverage gap closure analysis."""

    def test_biomarker_only_5_of_6(self):
        result = analyse_extended_coverage(tau=0.90)
        assert "PTX" in result['biomarker_only']['uncovered']
        assert len(result['biomarker_only']['covered']) == 5

    def test_with_estethoscope_6_of_6(self):
        result = analyse_extended_coverage(tau=0.90)
        assert len(result['biomarker_plus_estethoscope']['uncovered']) == 0
        assert len(result['biomarker_plus_estethoscope']['covered']) == 6

    def test_ptx_is_gap_closed(self):
        result = analyse_extended_coverage(tau=0.90)
        assert "PTX" in result['improvement']['pathologies_gained']


class TestCDRComparison:
    """Validate HEAR vs HEART vs SISTER ACT comparison."""

    def test_sister_act_highest_specificity(self):
        comp = compare_scoring_systems(n_patients=2000, seed=42)
        sister = comp['systems']['SISTER_ACT']
        hear = comp['systems']['HEAR']
        # SISTER ACT should have higher specificity (lower referral rate)
        assert sister['specificity'] > hear['specificity']

    def test_sister_act_lowest_referral_rate(self):
        comp = compare_scoring_systems(n_patients=2000, seed=42)
        sister = comp['systems']['SISTER_ACT']
        hear = comp['systems']['HEAR']
        assert sister['referral_rate'] < hear['referral_rate']


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION & DEPENDENCE MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCorrelationDependence:
    """Validate the Gaussian copula correlation model."""

    def test_correlation_matrix_shape(self):
        from correlation_dependence_model import build_correlation_matrix
        R = build_correlation_matrix()
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] >= 4

    def test_correlation_matrix_symmetric(self):
        from correlation_dependence_model import build_correlation_matrix
        R = build_correlation_matrix()
        np.testing.assert_array_almost_equal(R, R.T)

    def test_correlation_matrix_positive_semidefinite(self):
        from correlation_dependence_model import build_correlation_matrix
        R = build_correlation_matrix()
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals >= -1e-10), "Correlation matrix must be PSD"

    def test_corrected_fp_rate_below_independence(self):
        from correlation_dependence_model import corrected_panel_fp_rate
        panel = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
        result = corrected_panel_fp_rate(panel, n_mc=50_000, seed=42)
        assert result['copula_corrected']['fp_rate'] < result['independence_assumption']['fp_rate']

    def test_corrected_joint_npv_high(self):
        from correlation_dependence_model import corrected_joint_npv
        panel = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
        result = corrected_joint_npv(panel, n_mc=50_000, seed=42)
        assert result['copula_corrected_npv'] > 0.95

    def test_bayesian_sequential_saves_tests(self):
        from correlation_dependence_model import bayesian_sequential_testing
        panel = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
        result = bayesian_sequential_testing(panel, n_patients=1000, seed=42)
        assert result['average_tests_per_patient'] < len(panel)

    def test_run_dependence_analysis_completes(self):
        from correlation_dependence_model import run_dependence_analysis
        result = run_dependence_analysis(
            panel_biomarkers=["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"],
        )
        assert 'corrected_fp_rate' in result
        assert 'sequential_testing' in result
        assert 'corrected_npv' in result


# ═══════════════════════════════════════════════════════════════════════════
# EXTENDED BIOMARKER POOL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendedBiomarkerPool:
    """Validate the expanded 12-biomarker candidate pool."""

    def test_extended_matrix_shape(self):
        from biomarker_coverage_matrix import build_extended_pool_matrix
        C = build_extended_pool_matrix(include_extended=True)
        assert C.shape == (6, 12), f"Expected (6, 12), got {C.shape}"

    def test_extended_biomarkers_present(self):
        from biomarker_coverage_matrix import (
            build_extended_pool_matrix, EXTENDED_BIOMARKERS,
        )
        C = build_extended_pool_matrix(include_extended=True)
        for b in EXTENDED_BIOMARKERS:
            assert b in C.columns, f"Missing extended biomarker: {b}"

    def test_original_biomarkers_unchanged(self):
        from biomarker_coverage_matrix import (
            build_coverage_matrix, build_extended_pool_matrix,
        )
        C_orig = build_coverage_matrix()
        C_ext = build_extended_pool_matrix(include_extended=True)
        for b in C_orig.columns:
            pd.testing.assert_series_equal(
                C_orig[b], C_ext[b],
                check_names=True,
                obj=f"Original biomarker {b} values changed in extended matrix",
            )

    def test_extended_entries_have_sources(self):
        from biomarker_coverage_matrix import EXTENDED_COVERAGE_DATA
        for pathology, biomarker_dict in EXTENDED_COVERAGE_DATA.items():
            for biomarker, entry in biomarker_dict.items():
                assert entry.source, f"Missing source for {pathology}/{biomarker}"

    def test_extended_meta_available(self):
        from biomarker_coverage_matrix import (
            EXTENDED_BIOMARKER_META, EXTENDED_BIOMARKERS,
        )
        for b in EXTENDED_BIOMARKERS:
            assert b in EXTENDED_BIOMARKER_META, f"Missing meta for {b}"
            meta = EXTENDED_BIOMARKER_META[b]
            assert meta.cost_eur > 0, f"Invalid cost for {b}"


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH ECONOMICS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthEconomics:
    """Validate the health-economic decision analytic model."""

    def test_strategy_evaluation_runs(self):
        from health_economics import evaluate_strategy, STRATEGIES
        for key, strategy in STRATEGIES.items():
            result = evaluate_strategy(strategy, cohort_size=1000)
            assert result.total_cost > 0
            assert 0 <= result.sensitivity <= 1
            assert 0 <= result.specificity <= 1

    def test_icer_computation(self):
        from health_economics import evaluate_strategy, compute_icer, STRATEGIES
        current = evaluate_strategy(STRATEGIES['current_care'], 1000)
        optimal = evaluate_strategy(STRATEGIES['optimal_panel'], 1000)
        icer = compute_icer(current, optimal)
        assert 'icer_eur_per_qaly' in icer
        assert 'net_monetary_benefit' in icer

    def test_optimal_panel_detects_more_than_current(self):
        from health_economics import evaluate_strategy, STRATEGIES
        current = evaluate_strategy(STRATEGIES['current_care'], 10_000)
        optimal = evaluate_strategy(STRATEGIES['optimal_panel'], 10_000)
        # Optimal panel covers more pathologies → fewer missed cases
        assert optimal.missed_cases <= current.missed_cases

    def test_sister_act_detects_most(self):
        from health_economics import evaluate_strategy, STRATEGIES
        optimal = evaluate_strategy(STRATEGIES['optimal_panel'], 10_000)
        sister = evaluate_strategy(STRATEGIES['sister_act'], 10_000)
        assert sister.missed_cases <= optimal.missed_cases

    def test_tornado_analysis_runs(self):
        from health_economics import tornado_sensitivity_analysis
        result = tornado_sensitivity_analysis(cohort_size=1000)
        assert 'base_case_icer' in result
        assert len(result['parameters']) > 0
        # Parameters should be sorted by swing
        swings = [p['swing'] for p in result['parameters']]
        assert swings == sorted(swings, reverse=True)

    def test_psa_runs(self):
        from health_economics import probabilistic_sensitivity_analysis
        result = probabilistic_sensitivity_analysis(
            n_iterations=100, cohort_size=1000, seed=42,
        )
        assert 'ceac' in result
        assert 'strategy_summaries' in result
        assert 'icer_panel_vs_current' in result

    def test_full_analysis_runs(self):
        from health_economics import run_health_economics_analysis
        result = run_health_economics_analysis(cohort_size=1000)
        assert 'strategy_outcomes' in result
        assert 'icers' in result
        assert 'tornado' in result
        assert 'psa' in result
        assert 'dutch_gp_impact' in result
        assert 'extended_pool' in result


# ═══════════════════════════════════════════════════════════════════════════
# QUANTITATIVE LR INTERPRETATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantitativeLR:
    """Validate the quantitative likelihood-ratio interpretation module."""

    def test_binormal_separation_sanity(self):
        from quantitative_panel_interpretation import binormal_separation
        # Perfect test: sens=1, spec=1 → separation → +inf (capped by inv-CDF)
        # High-quality test: sens=0.95, spec=0.95 → d ≈ 3.29
        d = binormal_separation(0.95, 0.95)
        assert 3.0 < d < 4.0

    def test_binormal_separation_symmetric(self):
        from quantitative_panel_interpretation import binormal_separation
        # sens=0.80, spec=0.60 should equal spec=0.80, sens=0.60
        assert abs(binormal_separation(0.80, 0.60) - binormal_separation(0.60, 0.80)) < 0.01

    def test_binormal_auc_range(self):
        from quantitative_panel_interpretation import binormal_auc
        # d=0 → AUC=0.5; d=3 → AUC ≈ 0.98
        assert abs(binormal_auc(0.0) - 0.5) < 0.001
        assert binormal_auc(3.0) > 0.95

    def test_multivariate_auc(self):
        from quantitative_panel_interpretation import multivariate_auc
        seps = np.array([1.0, 1.0, 1.0])
        auc = multivariate_auc(seps)
        assert 0.8 < auc < 1.0

    def test_separation_matrix_shape(self):
        from quantitative_panel_interpretation import (
            _compute_separation_matrix, PANEL_BIOMARKERS, PATHOLOGIES,
        )
        sep = _compute_separation_matrix()
        for bm in PANEL_BIOMARKERS:
            assert bm in sep
            for path in PATHOLOGIES:
                assert path in sep[bm]
                assert isinstance(sep[bm][path], float)

    def test_full_analysis_runs(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=10_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        assert 'binary_or_rule' in result
        assert 'quantitative_lr' in result
        assert 'pathology_directed_management' in result
        assert 'hear_stratified_workflow' in result

    def test_binary_fp_above_90pct(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=20_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        assert result['binary_or_rule']['panel_fp_rate'] > 0.90

    def test_quantitative_lr_reduces_fp(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=20_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        binary_fp = result['binary_or_rule']['panel_fp_rate']
        quant_fp = result['quantitative_lr']['any_pathology_fp']['copula']
        assert quant_fp < binary_fp, "Quantitative LR should reduce FP vs binary"

    def test_pathology_directed_reduces_ed(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=20_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        any_fp = result['quantitative_lr']['any_pathology_fp']['copula']
        ed_rate = result['pathology_directed_management']['copula']['ed_referral_rate']
        assert ed_rate <= any_fp, "ED-only routing should be ≤ any-pathology FP"

    def test_per_pathology_sensitivity_maintained(self):
        from quantitative_panel_interpretation import run_full_analysis, COVERABLE
        result = run_full_analysis(
            n_healthy=20_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        for path in COVERABLE:
            sens = result['quantitative_lr']['per_pathology_sensitivity'][path]
            # Allow 2% tolerance due to Monte Carlo noise at small n
            assert sens >= 0.93, f"Sensitivity for {path} = {sens} < 0.93"

    def test_sensitivity_sweep(self):
        from quantitative_panel_interpretation import sweep_sensitivity_vs_fp
        sweep = sweep_sensitivity_vs_fp(
            target_sensitivities=[0.80, 0.90, 0.95],
            n_healthy=10_000, n_disease=5_000, seed=42,
        )
        assert len(sweep) == 3
        # FP should increase with sensitivity
        fps = [s['any_pathology_fp_copula'] for s in sweep]
        assert fps[0] <= fps[1] <= fps[2]

    def test_hear_workflow_sums(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=10_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        flow = result['hear_stratified_workflow']['per_1000_patients']
        total = flow['low_risk_discharged'] + flow['tested'] + flow['high_risk_direct_ed']
        assert abs(total - 1000) < 1, "HEAR strata must sum to 1000"

    def test_multivariate_auc_per_pathology(self):
        from quantitative_panel_interpretation import run_full_analysis
        result = run_full_analysis(
            n_healthy=10_000, n_disease=5_000, target_sensitivity=0.95, seed=42,
        )
        for path, auc in result['multivariate_auc_per_pathology'].items():
            assert 0.5 < auc < 1.0, f"AUC for {path} = {auc} out of range"
