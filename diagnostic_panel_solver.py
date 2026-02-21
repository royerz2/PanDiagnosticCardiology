"""
Diagnostic Panel Optimiser — Set-Cover / Minimum Hitting Set Solver
====================================================================
Adapted from the ALIN framework's MinimalHittingSetSolver.

The structural isomorphism:
  ALIN: genes hit viability paths   →   Diagnostics: biomarkers cover pathologies
  
Solvers: greedy weighted set cover, ILP (scipy.optimize.milp), exhaustive.
The ALIN codebase uses paths with multiple genes per path (set intersection).
Here, each pathology is a "path" and each biomarker either covers it (sensitivity ≥ τ)
or does not — a simpler binary incidence that maps directly to unweighted set cover.
"""

import math
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from biomarker_coverage_matrix import (
    BIOMARKERS, PATHOLOGIES, BIOMARKER_META,
    build_coverage_matrix, build_ci_matrices, build_early_coverage_matrix,
    build_specificity_matrix, get_prevalence_weights, get_severity_weights,
    PATHOLOGY_EPIDEMIOLOGY,
    BiomarkerMeta,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─── Data structures (isomorphic to ALIN's ViabilityPath / HittingSet) ──────

@dataclass(frozen=True)
class DiagnosticTarget:
    """A biomarker seen as a 'target' in the set-cover sense."""
    name: str
    cost_eur: float
    time_to_result_min: float
    sample_volume_ul: float
    poc_available: bool

    @staticmethod
    def from_meta(meta: BiomarkerMeta) -> "DiagnosticTarget":
        return DiagnosticTarget(
            name=meta.name,
            cost_eur=meta.cost_eur,
            time_to_result_min=meta.time_to_result_min,
            sample_volume_ul=meta.sample_volume_ul,
            poc_available=meta.poc_available,
        )


@dataclass
class DiagnosticPanel:
    """A candidate biomarker panel (isomorphic to ALIN's HittingSet)."""
    biomarkers: FrozenSet[str]
    total_cost_eur: float
    coverage: float                         # fraction of pathologies covered at threshold
    pathologies_covered: Set[str]           # names of covered pathologies
    pathologies_uncovered: Set[str]         # names of NOT covered pathologies
    worst_case_sensitivity: float           # min over pathologies of max sensitivity among panel
    total_time_min: float                   # max turnaround of any test in panel
    total_sample_ul: float                  # sum of sample volumes
    total_penalty: float = 0.0              # ILP objective value (weighted sum of all penalties)
    solver_method: str = "unknown"
    # Extended scoring (populated by score_panel)
    prevalence_weighted_coverage: float = 0.0   # Σ prevalence_p · covered_p
    severity_weighted_coverage: float = 0.0     # Σ fatality_p · covered_p
    expected_detections_per_1000: float = 0.0   # Σ prevalence_p · sensitivity_p · 1000 (all pathologies)
    expected_detections_covered_per_1000: float = 0.0  # only covered pathologies
    mean_panel_specificity: float = 0.0         # mean specificity across coverable pathologies
    net_benefit: float = 0.0                    # Vickers' NB at t=0.01 (rule-out appropriate)
    net_benefit_by_threshold: Dict = None       # NB at multiple decision thresholds
    clinical_utility_score: float = 0.0         # composite: sens*prev - (1-spec)*(1-prev)*w

    def __len__(self):
        return len(self.biomarkers)


# ─── Solver ─────────────────────────────────────────────────────────────────

class DiagnosticPanelSolver:
    """
    Minimum diagnostic panel solver using set-cover optimisation.
    
    Adapted from ALIN's MinimalHittingSetSolver with the mapping:
      viability paths  →  pathologies
      genes            →  biomarkers
      path.nodes       →  {biomarkers covering pathology at threshold τ}
    """

    EXHAUSTIVE_THRESHOLD = 20  # biomarker pool will always be ≤ 8, so always exact
    ILP_THRESHOLD = 500

    def __init__(self, coverage_matrix: Optional[pd.DataFrame] = None):
        """
        Args:
            coverage_matrix: P×B DataFrame of sensitivities. 
                             Defaults to build_coverage_matrix().
        """
        self.C = coverage_matrix if coverage_matrix is not None else build_coverage_matrix()
        self.targets = {
            b: DiagnosticTarget.from_meta(BIOMARKER_META[b])
            for b in self.C.columns
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def solve(
        self,
        tau: float = 0.90,
        max_size: int = 8,
        min_coverage: float = 1.0,
        w_size: float = 10.0,
        w_cost: float = 1.0,
        w_time: float = 0.5,
        w_sample: float = 0.1,
    ) -> List[DiagnosticPanel]:
        """
        Find minimum diagnostic panels covering all pathologies at threshold τ.

        The ILP objective is a weighted penalty function:
            min  Σ_b [ w_size + w_cost·cost(b) + w_time·time(b) + w_sample·sample(b) ] · x_b

        This is the MHS formulation with penalty weights — each biomarker incurs
        a base cardinality penalty (w_size), plus practical penalties for cost,
        turnaround time, and sample volume. Setting w_cost=w_time=w_sample=0
        recovers the pure minimum-cardinality hitting set.

        Args:
            tau: sensitivity threshold — biomarker b covers pathology p iff C[p,b] ≥ τ
            max_size: maximum panel size
            min_coverage: minimum fraction of pathologies to cover (default: all)
            w_size: penalty weight for panel cardinality (per biomarker)
            w_cost: penalty weight for cost (€ EUR)
            w_time: penalty weight for time-to-result (minutes)
            w_sample: penalty weight for sample volume (µL)

        Returns:
            List of DiagnosticPanel solutions, sorted by (size, total_penalty).
        """
        # Build binary incidence matrix at threshold τ
        binary = (self.C >= tau).astype(int)

        logger.info(
            f"Solving MDP: τ={tau}, {len(PATHOLOGIES)} pathologies, "
            f"{len(BIOMARKERS)} biomarkers, min_coverage={min_coverage}"
        )
        logger.info(f"Penalty weights: w_size={w_size}, w_cost={w_cost}, "
                     f"w_time={w_time}, w_sample={w_sample}")
        logger.info(f"Binary coverage matrix at τ={tau}:\n{binary.to_string()}")

        # Build biomarker → covered pathologies mapping
        biomarker_covers: Dict[str, Set[str]] = {}
        for b in self.C.columns:
            covered = set(binary.index[binary[b] == 1])
            biomarker_covers[b] = covered

        # Penalty function per biomarker (analogous to ALIN's NodeCost.total_cost)
        # Each biomarker incurs: base cardinality + cost + time + sample penalties
        biomarker_penalties = {
            b: (w_size
                + w_cost * self.targets[b].cost_eur
                + w_time * self.targets[b].time_to_result_min
                + w_sample * self.targets[b].sample_volume_ul)
            for b in self.C.columns
        }

        logger.info("Per-biomarker penalties:")
        for b, p in sorted(biomarker_penalties.items(), key=lambda x: x[1]):
            t = self.targets[b]
            logger.info(f"  {b:15s}: penalty={p:.2f}  "
                        f"(€{t.cost_eur:.1f}, {t.time_to_result_min:.0f}min, "
                        f"{t.sample_volume_ul:.0f}µL)")

        solutions = []

        # 1. Greedy
        greedy = self._solve_greedy(biomarker_covers, biomarker_penalties, max_size)
        if greedy:
            panel = self._make_panel(greedy, "greedy", tau)
            solutions.append(panel)

        # 2. Exhaustive (always feasible with 8 biomarkers)
        if len(self.C.columns) <= self.EXHAUSTIVE_THRESHOLD:
            exhaustive = self._solve_exhaustive(
                biomarker_covers, biomarker_penalties, max_size, min_coverage
            )
            for targets in exhaustive:
                panel = self._make_panel(targets, "exhaustive", tau)
                solutions.append(panel)

        # 3. ILP
        ilp = self._solve_ilp(biomarker_covers, biomarker_penalties, max_size, min_coverage)
        if ilp:
            panel = self._make_panel(ilp, "ilp", tau)
            solutions.append(panel)

        # Deduplicate
        seen: Set[FrozenSet[str]] = set()
        unique = []
        for panel in solutions:
            if panel.biomarkers not in seen:
                seen.add(panel.biomarkers)
                unique.append(panel)

        unique.sort(key=lambda p: (len(p), p.total_cost_eur))

        logger.info(f"Found {len(unique)} unique panel solutions")
        return unique

    def enumerate_all_panels(self, tau: float = 0.90) -> List[DiagnosticPanel]:
        """
        Enumerate ALL possible panels of size 1...|B| and score each.
        Used for Pareto frontier analysis.
        """
        panels = []
        biomarkers = list(self.C.columns)
        for k in range(1, len(biomarkers) + 1):
            for combo in combinations(biomarkers, k):
                panel = self._make_panel(set(combo), f"enum_k{k}", tau)
                panels.append(panel)
        logger.info(f"Enumerated {len(panels)} total panels")
        return panels

    # ── Greedy solver (adapted from ALIN) ────────────────────────────────────

    def _solve_greedy(
        self,
        biomarker_covers: Dict[str, Set[str]],
        biomarker_penalties: Dict[str, float],
        max_size: int,
    ) -> Optional[Set[str]]:
        """
        Greedy weighted set cover: pick biomarker with best coverage/penalty ratio.
        Identical logic to ALIN's MinimalHittingSetSolver._solve_greedy.
        """
        all_pathologies = set(self.C.index)
        uncovered = set(all_pathologies)
        selected: Set[str] = set()

        while uncovered and len(selected) < max_size:
            best_b = None
            best_ratio = -np.inf

            for b in set(biomarker_covers.keys()) - selected:
                hits = len(biomarker_covers[b] & uncovered)
                penalty = biomarker_penalties[b]
                if hits > 0:
                    ratio = hits / (penalty + 0.01)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_b = b

            if best_b is None:
                break

            selected.add(best_b)
            uncovered -= biomarker_covers[best_b]

        return selected if selected else None

    # ── ILP solver (adapted from ALIN) ───────────────────────────────────────

    def _solve_ilp(
        self,
        biomarker_covers: Dict[str, Set[str]],
        biomarker_penalties: Dict[str, float],
        max_size: int,
        min_coverage: float,
    ) -> Optional[Set[str]]:
        """
        Exact MHS solver via ILP (scipy.optimize.milp).
        
        Formulation (the proposal's Eq. 1 with penalties):
            x_b ∈ {0,1} for each biomarker b
            y_p ∈ {0,1} for each pathology p
            
            min  Σ_b penalty(b) · x_b
            
            where penalty(b) = w_size + w_cost·cost(b) + w_time·time(b) + w_sample·sample(b)
            
            s.t. Σ_{b covers p} x_b ≥ y_p           ∀p   (coverage linking)
                 Σ_p y_p ≥ ⌈min_coverage · |P|⌉          (minimum pathologies)
                 Σ_b x_b ≤ max_size                       (panel size limit)
                 x_b, y_p ∈ {0,1}
        
        This is isomorphic to ALIN's _solve_ilp with the variable mapping:
            ALIN gene targets  →  biomarkers (x_b)
            viability paths    →  pathologies (y_p)
            NodeCost           →  penalty(b)
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csc_matrix, hstack, eye
        except ImportError:
            logger.warning("scipy not available; skipping ILP")
            return None

        biomarkers = sorted(biomarker_covers.keys())
        pathologies = list(self.C.index)
        b_idx = {b: i for i, b in enumerate(biomarkers)}
        n_b = len(biomarkers)
        n_p = len(pathologies)
        n_vars = n_b + n_p

        # Objective: minimize weighted penalty of selected biomarkers
        c = np.zeros(n_vars)
        c[:n_b] = [biomarker_penalties[b] for b in biomarkers]

        # Build incidence matrix A (n_pathologies × n_biomarkers)
        rows, cols = [], []
        for p_i, pathology in enumerate(pathologies):
            for b in biomarkers:
                if pathology in biomarker_covers[b]:
                    rows.append(p_i)
                    cols.append(b_idx[b])

        if not rows:
            return None

        A = csc_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_p, n_b)
        )

        # Constraint 1: A @ x - I @ y ≥ 0  →  [A | -I] @ [x;y] ≥ 0
        neg_I = -eye(n_p, format='csc')
        cov_matrix = hstack([A, neg_I], format='csc')
        cov_constraint = LinearConstraint(cov_matrix, lb=np.zeros(n_p))

        # Constraint 2: Σ y_p ≥ ⌈min_coverage · n_p⌉
        min_covered = math.ceil(min_coverage * n_p)
        sum_y = csc_matrix(
            np.concatenate([np.zeros(n_b), np.ones(n_p)]).reshape(1, -1)
        )
        coverage_constraint = LinearConstraint(sum_y, lb=np.array([min_covered]))

        # Constraint 3: Σ x_b ≤ max_size
        sum_x = csc_matrix(
            np.concatenate([np.ones(n_b), np.zeros(n_p)]).reshape(1, -1)
        )
        card_constraint = LinearConstraint(sum_x, ub=np.array([max_size]))

        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)

        try:
            result = milp(
                c=c,
                constraints=[cov_constraint, coverage_constraint, card_constraint],
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': 30}
            )
        except Exception as e:
            logger.warning(f"ILP solver failed: {e}")
            return None

        if not result.success:
            logger.info(f"ILP infeasible: {result.message}")
            return None

        x_vals = result.x[:n_b]
        selected = {biomarkers[i] for i in range(n_b) if x_vals[i] > 0.5}
        return selected if selected else None

    # ── Exhaustive solver (adapted from ALIN) ────────────────────────────────

    def _solve_exhaustive(
        self,
        biomarker_covers: Dict[str, Set[str]],
        biomarker_penalties: Dict[str, float],
        max_size: int,
        min_coverage: float,
    ) -> List[Set[str]]:
        """
        Enumerate all subsets up to max_size, keep those meeting min_coverage.
        Provably optimal. Feasible because |B| = 8 → max 255 subsets.
        Sorted by (size, total_penalty) to prefer smaller, cheaper panels.
        """
        biomarkers = list(biomarker_covers.keys())
        all_pathologies = set(self.C.index)
        n_p = len(all_pathologies)
        min_covered = math.ceil(min_coverage * n_p)

        solutions: List[Tuple[Set[str], float]] = []

        for k in range(1, max_size + 1):
            for combo in combinations(biomarkers, k):
                combo_set = set(combo)
                covered = set()
                for b in combo_set:
                    covered |= biomarker_covers[b]
                if len(covered) >= min_covered:
                    penalty = sum(biomarker_penalties[b] for b in combo_set)
                    solutions.append((combo_set, penalty))

        # Sort by (size, total_penalty) and return
        solutions.sort(key=lambda x: (len(x[0]), x[1]))
        return [s[0] for s in solutions[:20]]

    # ── Panel construction helper ────────────────────────────────────────────

    def _make_panel(
        self, biomarkers_selected: Set[str], method: str, tau: float
    ) -> DiagnosticPanel:
        """Build a DiagnosticPanel from a set of selected biomarker names."""
        all_pathologies = set(self.C.index)
        covered = set()
        for pathology in all_pathologies:
            if any(self.C.loc[pathology, b] >= tau for b in biomarkers_selected):
                covered.add(pathology)
        uncovered = all_pathologies - covered

        # Worst-case sensitivity: for each pathology, take the best biomarker
        # in the panel, then take the minimum across pathologies
        worst_case = 1.0
        for pathology in all_pathologies:
            best_sens = max(
                self.C.loc[pathology, b] for b in biomarkers_selected
            )
            worst_case = min(worst_case, best_sens)

        total_cost = sum(self.targets[b].cost_eur for b in biomarkers_selected)
        max_time = max(self.targets[b].time_to_result_min for b in biomarkers_selected)
        total_volume = sum(self.targets[b].sample_volume_ul for b in biomarkers_selected)

        # Compute total penalty (default weights — matches solve() defaults)
        total_penalty = sum(
            10.0 + 1.0 * self.targets[b].cost_eur
            + 0.5 * self.targets[b].time_to_result_min
            + 0.1 * self.targets[b].sample_volume_ul
            for b in biomarkers_selected
        )

        return DiagnosticPanel(
            biomarkers=frozenset(biomarkers_selected),
            total_cost_eur=total_cost,
            coverage=len(covered) / len(all_pathologies),
            pathologies_covered=covered,
            pathologies_uncovered=uncovered,
            worst_case_sensitivity=worst_case,
            total_time_min=max_time,
            total_sample_ul=total_volume,
            total_penalty=total_penalty,
            solver_method=method,
        )

    def score_panel(self, panel: DiagnosticPanel, tau: float = 0.90) -> DiagnosticPanel:
        """
        Compute extended clinical utility scores for a panel.

        Adds prevalence-weighted coverage, severity-weighted coverage,
        expected detections per 1000 presentations, mean panel specificity,
        and Vickers' net benefit.

        Mutates and returns the same panel object.
        """
        prevalence = get_prevalence_weights()
        severity = get_severity_weights()
        spec_matrix = build_specificity_matrix()

        # Prevalence-weighted coverage: Σ prevalence_p · I(covered_p)
        prev_cov = sum(
            prevalence[p] for p in panel.pathologies_covered
        )
        panel.prevalence_weighted_coverage = prev_cov

        # Severity-weighted coverage: Σ fatality_p · I(covered_p)
        sev_cov = sum(
            severity[p] for p in panel.pathologies_covered
        )
        panel.severity_weighted_coverage = sev_cov

        # Expected detections per 1000: separate covered vs all pathologies
        expected_all = 0.0
        expected_covered = 0.0
        for p in self.C.index:
            best_sens = max(self.C.loc[p, b] for b in panel.biomarkers)
            det = prevalence[p] * best_sens * 1000
            expected_all += det
            if p in panel.pathologies_covered:
                expected_covered += det
        panel.expected_detections_per_1000 = expected_all
        panel.expected_detections_covered_per_1000 = expected_covered

        # Mean panel specificity across coverable pathologies
        spec_values = []
        for p in panel.pathologies_covered:
            # For each covered pathology, use the specificity of the best
            # sensitivity biomarker in the panel for that pathology
            best_b = max(panel.biomarkers, key=lambda b: self.C.loc[p, b])
            spec_values.append(spec_matrix.loc[p, best_b])
        panel.mean_panel_specificity = np.mean(spec_values) if spec_values else 0.0

        # Net benefit (Vickers' decision curve framework):
        # NB = Σ_p [ sens_p * prev_p - (1-spec_p) * (1-prev_p) * w ]
        # IMPORTANT: w = t/(1-t) where t is the CLINICAL DECISION threshold,
        # NOT the sensitivity selection threshold τ. For rule-out of
        # life-threatening conditions, t should be low (0.01-0.10).
        # Only covered pathologies contribute (uncovered pathologies are
        # not claimed to be screened).
        panel.net_benefit_by_threshold = {}
        for t in [0.01, 0.02, 0.05, 0.10]:
            w = t / (1 - t)
            nb = 0.0
            for p in panel.pathologies_covered:
                best_b = max(panel.biomarkers, key=lambda b: self.C.loc[p, b])
                sens_p = self.C.loc[p, best_b]
                spec_p = spec_matrix.loc[p, best_b]
                prev_p = prevalence[p]
                nb += sens_p * prev_p - (1 - spec_p) * (1 - prev_p) * w
            panel.net_benefit_by_threshold[t] = round(nb, 6)
        # Primary NB: t=0.01 (appropriate for life-threatening rule-out)
        panel.net_benefit = panel.net_benefit_by_threshold.get(0.01, 0.0)

        # Clinical utility score: prevalence*severity weighted expected benefit
        # CU = Σ_p prevalence_p · fatality_p · max_sens_p(panel) · spec_p
        cu = 0.0
        for p in panel.pathologies_covered:
            best_b = max(panel.biomarkers, key=lambda b: self.C.loc[p, b])
            sens_p = self.C.loc[p, best_b]
            spec_p = spec_matrix.loc[p, best_b]
            cu += prevalence[p] * severity[p] * sens_p * spec_p
        panel.clinical_utility_score = cu

        return panel

    def solve_robust(
        self,
        tau: float = 0.90,
        max_size: int = 8,
        w_size: float = 10.0,
        w_cost: float = 1.0,
        w_time: float = 0.5,
        w_sample: float = 0.1,
    ) -> List[DiagnosticPanel]:
        """
        Robust optimization using CI lower bounds instead of point estimates.

        Uses the lower 95% CI boundary of published meta-analytic sensitivities,
        guaranteeing that the panel is optimal even under worst-case uncertainty.
        If the panel survives under lower CI bounds, it is robust to sampling
        variation in the original meta-analyses.
        """
        lower_ci, _ = build_ci_matrices()
        robust_solver = DiagnosticPanelSolver(coverage_matrix=lower_ci)
        return robust_solver.solve(
            tau=tau, max_size=max_size,
            w_size=w_size, w_cost=w_cost, w_time=w_time, w_sample=w_sample,
        )


# ─── Analysis functions ─────────────────────────────────────────────────────

def marginal_value_analysis(
    solver: DiagnosticPanelSolver,
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    Compute marginal diagnostic value as panel size grows.
    For each k=1..8, find the best k-panel and report coverage gain.
    """
    biomarkers = list(solver.C.columns)
    all_pathologies = set(solver.C.index)
    results = []

    best_coverage_prev = 0.0
    for k in range(1, len(biomarkers) + 1):
        best_panel = None
        best_coverage = 0.0
        best_cost = np.inf

        for combo in combinations(biomarkers, k):
            combo_set = set(combo)
            covered = set()
            for p in all_pathologies:
                if any(solver.C.loc[p, b] >= tau for b in combo_set):
                    covered.add(p)
            cov_frac = len(covered) / len(all_pathologies)
            cost = sum(solver.targets[b].cost_eur for b in combo_set)

            if cov_frac > best_coverage or (cov_frac == best_coverage and cost < best_cost):
                best_coverage = cov_frac
                best_cost = cost
                best_panel = combo_set

        marginal = best_coverage - best_coverage_prev
        results.append({
            'panel_size': k,
            'best_panel': ', '.join(sorted(best_panel)) if best_panel else '',
            'coverage': best_coverage,
            'marginal_coverage_gain': marginal,
            'cost_eur': best_cost,
            'pathologies_covered': int(best_coverage * len(all_pathologies)),
        })
        best_coverage_prev = best_coverage

    return pd.DataFrame(results)


def current_approach_comparison(
    solver: DiagnosticPanelSolver,
    tau: float = 0.90,
) -> pd.DataFrame:
    """
    Compare current CDR+biomarker approaches against optimal panels.
    
    Current approaches (from proposal):
      - HEART + hs-cTnI: single-biomarker ACS-focused
      - Wells + D-dimer: PE-focused
    """
    approaches = [
        {"name": "HEART + hs-cTnI (ACS-focused)", "biomarkers": {"hs-cTnI"}},
        {"name": "Wells + D-dimer (PE-focused)", "biomarkers": {"D-dimer"}},
        {"name": "Combined: hs-cTnI + D-dimer", "biomarkers": {"hs-cTnI", "D-dimer"}},
    ]

    # Add optimal panels from solver
    solutions = solver.solve(tau=tau, max_size=8, min_coverage=0.5)
    for panel in solutions[:5]:
        approaches.append({
            "name": f"Optimal {len(panel)}-test ({panel.solver_method})",
            "biomarkers": set(panel.biomarkers),
        })

    rows = []
    for approach in approaches:
        panel = solver._make_panel(approach["biomarkers"], "reference", tau)
        rows.append({
            'approach': approach['name'],
            'biomarkers': ', '.join(sorted(approach['biomarkers'])),
            'n_tests': len(approach['biomarkers']),
            'coverage': panel.coverage,
            'pathologies_covered': len(panel.pathologies_covered),
            'pathologies_uncovered': ', '.join(sorted(panel.pathologies_uncovered)) or 'None',
            'worst_case_sensitivity': panel.worst_case_sensitivity,
            'cost_eur': panel.total_cost_eur,
            'max_time_min': panel.total_time_min,
            'sample_volume_ul': panel.total_sample_ul,
        })

    return pd.DataFrame(rows)


# ─── Main entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 90)
    print("DIAGNOSTIC PANEL SET-COVER OPTIMISATION")
    print("Adapted from ALIN Minimum Hitting Set Solver")
    print("=" * 90)

    solver = DiagnosticPanelSolver()

    # ── Run at multiple thresholds ──
    for tau in [0.80, 0.85, 0.90, 0.95]:
        print(f"\n{'─' * 90}")
        print(f"THRESHOLD τ = {tau}")
        print(f"{'─' * 90}")

        solutions = solver.solve(tau=tau, max_size=8)

        if solutions:
            print(f"\nMinimum panel: {', '.join(sorted(solutions[0].biomarkers))}")
            print(f"  Size: {len(solutions[0])}")
            print(f"  Coverage: {solutions[0].coverage:.0%}")
            print(f"  Cost: €{solutions[0].total_cost_eur:.2f}")
            print(f"  Worst-case sensitivity: {solutions[0].worst_case_sensitivity:.2f}")
            if solutions[0].pathologies_uncovered:
                print(f"  Gaps: {', '.join(sorted(solutions[0].pathologies_uncovered))}")
            else:
                print(f"  Gaps: NONE — full coverage achieved")
        else:
            print("  No feasible panel found at this threshold.")

    # ── Marginal value analysis ──
    print(f"\n{'=' * 90}")
    print("MARGINAL DIAGNOSTIC VALUE ANALYSIS (τ = 0.90)")
    print(f"{'=' * 90}")
    marginal = marginal_value_analysis(solver, tau=0.90)
    print(marginal.to_string(index=False))

    # ── Current approach comparison ──
    print(f"\n{'=' * 90}")
    print("COMPARISON: CURRENT APPROACHES vs OPTIMAL PANELS (τ = 0.90)")
    print(f"{'=' * 90}")
    comparison = current_approach_comparison(solver, tau=0.90)
    print(comparison.to_string(index=False))
