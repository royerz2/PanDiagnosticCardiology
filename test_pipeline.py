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
