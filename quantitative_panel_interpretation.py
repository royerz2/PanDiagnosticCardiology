"""
quantitative_panel_interpretation.py
====================================

Silver-bullet analysis: switching from binary any-positive interpretation
to quantitative likelihood-ratio (LR) interpretation for the 4-biomarker
panel, **combined with pathology-directed management**.

Three-layer FP reduction
-------------------------
1. Quantitative LR (replace binary cutoffs with continuous LRs)
2. Pathology-directed management (ED-refer only for ACS/PE/AoD;
   outpatient workup for pericarditis/AHF)
3. HEAR pre-stratification (test only moderate-risk 35%)

Together these transform the clinical picture from "93% of healthy
patients get unnecessary ED referrals" to a manageable workflow.

Mathematical framework
----------------------
For each biomarker–pathology pair (b, j), the **binormal model** defines:

    Healthy:  X_b ~ N(0, 1)
    Disease:  X_b | pathology_j ~ N(d_bj, 1)

where the *separation* d_bj = Φ⁻¹(GlobalSpec_b) + Φ⁻¹(Sens_bj) and the
cutoff is c_b = Φ⁻¹(GlobalSpec_b).

The continuous likelihood ratio for value x_b under pathology j is:

    LR_bj(x) = φ(x - d_bj) / φ(x) = exp(d_bj · x − d_bj² / 2)

For a panel of K biomarkers (conditional independence given health
status), the combined LR for pathology j is:

    LR_j(x₁…xₖ) = ∏_b  LR_bj(xb) = exp(Σ_b [d_bj · x_b − d_bj²/2])

Author: R. Erzurumluoglu, 2025–2026
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ── Parameters ──────────────────────────────────────────────────────────────

PANEL_BIOMARKERS = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]

PATHOLOGIES = [
    "ACS", "PE", "AoD", "Pericarditis", "Pneumothorax", "AHF",
]

# Coverable pathologies (excludes Pneumothorax — no biomarker coverage)
COVERABLE = ["ACS", "PE", "AoD", "Pericarditis", "AHF"]

# High-acuity pathologies requiring ED referral
ED_REQUIRED = ["ACS", "PE", "AoD"]

# Lower-acuity pathologies manageable in outpatient / primary care
OUTPATIENT_MANAGEABLE = ["Pericarditis", "AHF"]

# Sensitivity of each panel biomarker for each pathology
SENSITIVITY_MATRIX: Dict[str, Dict[str, float]] = {
    "hs-cTnI":   {"ACS": 0.95, "PE": 0.45, "AoD": 0.28, "Pericarditis": 0.82, "Pneumothorax": 0.10, "AHF": 0.68},
    "D-dimer":   {"ACS": 0.38, "PE": 0.95, "AoD": 0.97, "Pericarditis": 0.35, "Pneumothorax": 0.15, "AHF": 0.60},
    "NT-proBNP": {"ACS": 0.60, "PE": 0.60, "AoD": 0.45, "Pericarditis": 0.70, "Pneumothorax": 0.20, "AHF": 0.95},
    "CRP":       {"ACS": 0.50, "PE": 0.65, "AoD": 0.72, "Pericarditis": 0.92, "Pneumothorax": 0.15, "AHF": 0.55},
}

# Global specificity per biomarker (property of the healthy population,
# NOT pathology-specific  →  consistent cutoff per biomarker)
GLOBAL_SPECIFICITY = {
    "hs-cTnI": 0.54,
    "D-dimer": 0.42,
    "NT-proBNP": 0.76,
    "CRP": 0.38,
}

# Primary-care prevalence per pathology
PREVALENCES = {
    "ACS": 0.035,
    "PE": 0.015,
    "AoD": 0.003,
    "Pericarditis": 0.040,
    "Pneumothorax": 0.005,
    "AHF": 0.025,
}

# Correlation matrix for the 4 panel biomarkers in healthy individuals
# Order: hs-cTnI, D-dimer, NT-proBNP, CRP
PANEL_CORRELATION = np.array([
    [1.00, 0.20, 0.45, 0.25],
    [0.20, 1.00, 0.30, 0.35],
    [0.45, 0.30, 1.00, 0.25],
    [0.25, 0.35, 0.25, 1.00],
])


# ── Binormal model utilities ───────────────────────────────────────────────

def binormal_separation(sens: float, spec: float) -> float:
    """d = Φ⁻¹(Spec) + Φ⁻¹(Sens)."""
    s = np.clip(sens, 1e-6, 1 - 1e-6)
    p = np.clip(spec, 1e-6, 1 - 1e-6)
    return float(stats.norm.ppf(p) + stats.norm.ppf(s))


def binormal_auc(d: float) -> float:
    return float(stats.norm.cdf(d / np.sqrt(2)))


def multivariate_auc(separations: np.ndarray) -> float:
    """AUC_MV = Φ(√(Σ d²) / √2)."""
    d_mv = np.sqrt(np.sum(separations ** 2))
    return float(stats.norm.cdf(d_mv / np.sqrt(2)))


# ── Separation matrix ──────────────────────────────────────────────────────

def _compute_separation_matrix() -> Dict[str, Dict[str, float]]:
    """d[bm][path] using GLOBAL specificity for consistent cutoffs."""
    d = {}
    for bm in PANEL_BIOMARKERS:
        d[bm] = {}
        spec = GLOBAL_SPECIFICITY[bm]          # <── global, not pathology-specific
        for path in PATHOLOGIES:
            sens = SENSITIVITY_MATRIX[bm][path]
            d[bm][path] = binormal_separation(sens, spec)
    return d


# ── Simulation helpers ─────────────────────────────────────────────────────

def _simulate_healthy(n: int, rng: np.random.Generator,
                      correlated: bool = False) -> np.ndarray:
    if correlated:
        L = np.linalg.cholesky(PANEL_CORRELATION)
        z = rng.standard_normal((n, len(PANEL_BIOMARKERS)))
        return z @ L.T
    return rng.standard_normal((n, len(PANEL_BIOMARKERS)))


def _simulate_disease(pathology: str, n: int, rng: np.random.Generator,
                      sep_matrix: Dict) -> np.ndarray:
    means = np.array([sep_matrix[bm][pathology] for bm in PANEL_BIOMARKERS])
    return rng.standard_normal((n, len(PANEL_BIOMARKERS))) + means


# ── Binary OR rule ─────────────────────────────────────────────────────────

def _binary_panel_results(healthy: np.ndarray) -> np.ndarray:
    """Returns boolean array: True if any biomarker exceeds its cutoff."""
    cutoffs = np.array([stats.norm.ppf(GLOBAL_SPECIFICITY[bm])
                        for bm in PANEL_BIOMARKERS])
    return (healthy > cutoffs).any(axis=1)


def _binary_panel_fp(healthy: np.ndarray) -> float:
    return float(_binary_panel_results(healthy).mean())


def _binary_sensitivity(disease_vals: np.ndarray) -> float:
    """ANY panel biomarker positive → detected."""
    cutoffs = np.array([stats.norm.ppf(GLOBAL_SPECIFICITY[bm])
                        for bm in PANEL_BIOMARKERS])
    return float((disease_vals > cutoffs).any(axis=1).mean())


# ── Quantitative LR interpretation ────────────────────────────────────────

def _quantitative_posteriors(values: np.ndarray,
                             sep_matrix: Dict) -> np.ndarray:
    """n × len(PATHOLOGIES) array of posterior probabilities."""
    n = len(values)
    posteriors = np.zeros((n, len(PATHOLOGIES)))
    for j, path in enumerate(PATHOLOGIES):
        prior = PREVALENCES[path]
        log_lr = np.zeros(n)
        for i, bm in enumerate(PANEL_BIOMARKERS):
            d = sep_matrix[bm][path]
            log_lr += d * values[:, i] - 0.5 * d ** 2
        lr = np.exp(np.clip(log_lr, -50, 50))
        posteriors[:, j] = (prior * lr) / (prior * lr + (1 - prior))
    return posteriors


def _find_per_pathology_thresholds(
    sep_matrix: Dict,
    target_sens: float = 0.95,
    pathologies: Optional[List[str]] = None,
    n_disease: int = 50_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Per-pathology posterior threshold achieving ≥target_sens each."""
    if pathologies is None:
        pathologies = COVERABLE
    thresholds = {}
    for path in pathologies:
        rng = np.random.default_rng(abs(hash(path)) % (2**32) + seed)
        disease_vals = _simulate_disease(path, n_disease, rng, sep_matrix)
        posteriors = _quantitative_posteriors(disease_vals, sep_matrix)
        j = PATHOLOGIES.index(path)
        # 5th percentile → 95% are above → sensitivity ≥ 95%
        thresh = float(np.percentile(posteriors[:, j], (1 - target_sens) * 100))
        thresholds[path] = thresh
    return thresholds


def _quantitative_panel_fp(
    healthy: np.ndarray,
    sep_matrix: Dict,
    thresholds: Dict[str, float],
    pathologies: Optional[List[str]] = None,
) -> float:
    """FP rate: any pathology in *pathologies* exceeds its threshold."""
    if pathologies is None:
        pathologies = COVERABLE
    posteriors = _quantitative_posteriors(healthy, sep_matrix)
    any_above = np.zeros(len(healthy), dtype=bool)
    for path in pathologies:
        j = PATHOLOGIES.index(path)
        any_above |= (posteriors[:, j] > thresholds[path])
    return float(any_above.mean())


def _verify_sensitivity(
    sep_matrix: Dict,
    thresholds: Dict[str, float],
    n_disease: int = 50_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Verify per-pathology sensitivity at calibrated thresholds."""
    result = {}
    for path in COVERABLE:
        rng = np.random.default_rng(abs(hash(path)) % (2**32) + seed)
        dv = _simulate_disease(path, n_disease, rng, sep_matrix)
        posteriors = _quantitative_posteriors(dv, sep_matrix)
        j = PATHOLOGIES.index(path)
        result[path] = float((posteriors[:, j] > thresholds[path]).mean())
    return result


# ── Pathology-directed management ─────────────────────────────────────────

def _pathology_directed_analysis(
    healthy: np.ndarray,
    sep_matrix: Dict,
    thresholds: Dict[str, float],
) -> Dict:
    """Three-tier triage: ED referral / Outpatient workup / Discharge.

    - ED referral: posterior for ACS, PE, or AoD exceeds threshold
    - Outpatient:  posterior for Pericarditis or AHF exceeds threshold
                   (but no ED-requiring pathology triggered)
    - Discharge:   all posteriors below thresholds
    """
    posteriors = _quantitative_posteriors(healthy, sep_matrix)
    n = len(healthy)

    ed_triggered = np.zeros(n, dtype=bool)
    for path in ED_REQUIRED:
        j = PATHOLOGIES.index(path)
        ed_triggered |= (posteriors[:, j] > thresholds[path])

    outpt_triggered = np.zeros(n, dtype=bool)
    for path in OUTPATIENT_MANAGEABLE:
        j = PATHOLOGIES.index(path)
        outpt_triggered |= (posteriors[:, j] > thresholds[path])
    outpt_only = outpt_triggered & ~ed_triggered

    discharged = ~(ed_triggered | outpt_triggered)

    return {
        "ed_referral_rate": float(ed_triggered.mean()),
        "outpatient_workup_rate": float(outpt_only.mean()),
        "discharge_rate": float(discharged.mean()),
        "any_action_rate": float((ed_triggered | outpt_triggered).mean()),
    }


# ── HEAR pre-stratification layer ────────────────────────────────────────

def _hear_stratified_analysis(
    pathology_directed: Dict,
    binary_fp: float,
) -> Dict:
    """Combine pathology-directed triage with HEAR pre-stratification.

    HEAR distribution: Low 50%, Moderate 35%, High 15%
    Only moderate-risk patients (35%) receive POC testing.
    """
    pct_tested = 0.35
    pct_low = 0.50
    pct_high = 0.15

    # Per 1000 patients
    n = 1000
    n_low = int(n * pct_low)       # 500 discharged without testing
    n_tested = int(n * pct_tested)  # 350 tested
    n_high = int(n * pct_high)      # 150 direct ED referral

    # Among tested patients (all healthy in this FP scenario):
    ed_from_testing = n_tested * pathology_directed["ed_referral_rate"]
    outpt_from_testing = n_tested * pathology_directed["outpatient_workup_rate"]
    discharged_from_testing = n_tested * pathology_directed["discharge_rate"]

    total_ed = n_high + ed_from_testing  # high-risk + test-triggered
    total_outpt = outpt_from_testing
    total_discharged = n_low + discharged_from_testing

    # Binary OR comparison: all 350 tested patients with FP → ED
    binary_ed_from_testing = n_tested * binary_fp
    binary_total_ed = n_high + binary_ed_from_testing

    return {
        "per_1000_patients": {
            "low_risk_discharged": n_low,
            "tested": n_tested,
            "high_risk_direct_ed": n_high,
            "test_triggered_ed_referrals": round(ed_from_testing, 1),
            "test_triggered_outpatient_workups": round(outpt_from_testing, 1),
            "test_discharged": round(discharged_from_testing, 1),
            "total_ed_referrals": round(total_ed, 1),
            "total_unnecessary_ed_pct": f"{total_ed/n:.1%}",
        },
        "vs_binary_or": {
            "binary_ed_from_testing": round(binary_ed_from_testing, 1),
            "binary_total_ed": round(binary_total_ed, 1),
            "ed_referrals_avoided": round(binary_total_ed - total_ed, 1),
            "reduction_pct": f"{(binary_total_ed - total_ed) / binary_total_ed:.1%}" if binary_total_ed > 0 else "N/A",
        },
    }


# ── Sensitivity sweep ────────────────────────────────────────────────────

def sweep_sensitivity_vs_fp(
    target_sensitivities: Optional[List[float]] = None,
    n_healthy: int = 200_000,
    n_disease: int = 50_000,
    seed: int = 42,
) -> List[Dict]:
    if target_sensitivities is None:
        target_sensitivities = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

    sep_matrix = _compute_separation_matrix()
    rng = np.random.default_rng(seed)
    healthy_indep = _simulate_healthy(n_healthy, rng, correlated=False)
    rng2 = np.random.default_rng(seed + 1)
    healthy_copula = _simulate_healthy(n_healthy, rng2, correlated=True)

    results = []
    for ts in target_sensitivities:
        thresholds = _find_per_pathology_thresholds(sep_matrix, ts, COVERABLE, n_disease, seed + 3)

        fp_all_indep = _quantitative_panel_fp(healthy_indep, sep_matrix, thresholds, COVERABLE)
        fp_all_copula = _quantitative_panel_fp(healthy_copula, sep_matrix, thresholds, COVERABLE)
        fp_ed_indep = _quantitative_panel_fp(healthy_indep, sep_matrix, thresholds, ED_REQUIRED)
        fp_ed_copula = _quantitative_panel_fp(healthy_copula, sep_matrix, thresholds, ED_REQUIRED)

        results.append({
            "target_sensitivity": ts,
            "any_pathology_fp_indep": round(fp_all_indep, 4),
            "any_pathology_fp_copula": round(fp_all_copula, 4),
            "ed_only_fp_indep": round(fp_ed_indep, 4),
            "ed_only_fp_copula": round(fp_ed_copula, 4),
        })

    return results


# ── Main analysis ──────────────────────────────────────────────────────────

def run_full_analysis(
    n_healthy: int = 200_000,
    n_disease: int = 50_000,
    target_sensitivity: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Full comparison: binary OR vs quantitative LR vs pathology-directed."""
    print("  [Quantitative Panel] Computing separation matrix...")
    sep_matrix = _compute_separation_matrix()

    # ── Multivariate AUC per pathology ──────────────────────────────────
    mv_auc = {}
    theoretical = {}
    for path in PATHOLOGIES:
        seps = np.array([sep_matrix[bm][path] for bm in PANEL_BIOMARKERS])
        d_mv = float(np.sqrt(np.sum(seps ** 2)))
        auc = multivariate_auc(seps)
        c_mv = d_mv - stats.norm.ppf(0.95)
        spec95 = float(stats.norm.cdf(c_mv))
        mv_auc[path] = round(auc, 4)
        theoretical[path] = {
            "separations": {bm: round(sep_matrix[bm][path], 3) for bm in PANEL_BIOMARKERS},
            "d_multivariate": round(d_mv, 3),
            "auc_multivariate": round(auc, 4),
            "specificity_at_95pct_sensitivity": round(spec95, 4),
        }

    # ── Simulate healthy ────────────────────────────────────────────────
    print("  [Quantitative Panel] Simulating healthy populations...")
    rng_i = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed + 1)
    healthy_indep = _simulate_healthy(n_healthy, rng_i, correlated=False)
    healthy_copula = _simulate_healthy(n_healthy, rng_c, correlated=True)

    # ── Binary OR rule ──────────────────────────────────────────────────
    binary_fp = _binary_panel_fp(healthy_indep)
    binary_sens = {}
    rng_d = np.random.default_rng(seed + 2)
    for path in COVERABLE:
        dv = _simulate_disease(path, n_disease, rng_d, sep_matrix)
        binary_sens[path] = round(_binary_sensitivity(dv), 4)

    # ── Calibrate per-pathology thresholds ──────────────────────────────
    print("  [Quantitative Panel] Calibrating per-pathology thresholds...")
    thresholds = _find_per_pathology_thresholds(
        sep_matrix, target_sensitivity, COVERABLE, n_disease, seed + 3
    )

    # ── Verify sensitivity ──────────────────────────────────────────────
    quant_sens = _verify_sensitivity(sep_matrix, thresholds, n_disease, seed + 3)

    # ── FP rates under quantitative LR ──────────────────────────────────
    print("  [Quantitative Panel] Computing FP rates...")
    fp_all_indep = _quantitative_panel_fp(healthy_indep, sep_matrix, thresholds, COVERABLE)
    fp_all_copula = _quantitative_panel_fp(healthy_copula, sep_matrix, thresholds, COVERABLE)
    fp_ed_indep = _quantitative_panel_fp(healthy_indep, sep_matrix, thresholds, ED_REQUIRED)
    fp_ed_copula = _quantitative_panel_fp(healthy_copula, sep_matrix, thresholds, ED_REQUIRED)

    # ── Pathology-directed management ───────────────────────────────────
    print("  [Quantitative Panel] Pathology-directed management analysis...")
    pd_indep = _pathology_directed_analysis(healthy_indep, sep_matrix, thresholds)
    pd_copula = _pathology_directed_analysis(healthy_copula, sep_matrix, thresholds)

    # ── HEAR-stratified patient flow ────────────────────────────────────
    hear_flow = _hear_stratified_analysis(pd_copula, binary_fp)

    # ── Sensitivity sweep ───────────────────────────────────────────────
    print("  [Quantitative Panel] Sensitivity sweep...")
    sweep = sweep_sensitivity_vs_fp(seed=seed)

    # ── Assemble output ─────────────────────────────────────────────────
    output = {
        "method": "Quantitative LR interpretation + pathology-directed management",
        "description": (
            "Three-layer FP reduction: (1) quantitative LR replaces binary cutoffs, "
            "(2) pathology-directed management routes pericarditis/AHF to outpatient "
            "instead of ED, (3) HEAR pre-stratification limits testing to 35% moderate-risk."
        ),
        "binary_or_rule": {
            "panel_fp_rate": round(binary_fp, 4),
            "panel_specificity": round(1 - binary_fp, 4),
            "per_pathology_sensitivity": binary_sens,
            "note": "Any positive → ED referral. No pathology discrimination.",
        },
        "quantitative_lr": {
            "per_pathology_thresholds": {k: round(v, 6) for k, v in thresholds.items()},
            "per_pathology_sensitivity": {k: round(v, 4) for k, v in quant_sens.items()},
            "any_pathology_fp": {
                "independent": round(fp_all_indep, 4),
                "copula": round(fp_all_copula, 4),
            },
            "ed_only_fp": {
                "independent": round(fp_ed_indep, 4),
                "copula": round(fp_ed_copula, 4),
            },
        },
        "pathology_directed_management": {
            "independent": pd_indep,
            "copula": pd_copula,
            "explanation": (
                "ED referral triggered ONLY for ACS/PE/AoD. "
                "Pericarditis/AHF → outpatient echo + medical Rx. "
                "All below threshold → safe discharge."
            ),
        },
        "hear_stratified_workflow": hear_flow,
        "multivariate_auc_per_pathology": mv_auc,
        "theoretical_mv_roc": theoretical,
        "sensitivity_fp_tradeoff": sweep,
        "fp_reduction_summary": {
            "binary_or_total_fp": f"{binary_fp:.1%}",
            "quant_lr_total_fp_copula": f"{fp_all_copula:.1%}",
            "quant_lr_ed_only_fp_copula": f"{fp_ed_copula:.1%}",
            "pathology_directed_ed_rate_copula": f"{pd_copula['ed_referral_rate']:.1%}",
            "pathology_directed_discharge_rate_copula": f"{pd_copula['discharge_rate']:.1%}",
            "hear_plus_quant_unnecessary_ed_per_1000": hear_flow["per_1000_patients"]["total_ed_referrals"],
            "hear_plus_binary_unnecessary_ed_per_1000": hear_flow["vs_binary_or"]["binary_total_ed"],
            "ed_referrals_avoided_per_1000": hear_flow["vs_binary_or"]["ed_referrals_avoided"],
        },
    }

    # Save
    out_path = Path(__file__).parent / "results" / "quantitative_interpretation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  [Quantitative Panel] Results saved to {out_path}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_full_analysis()

    print("\n" + "=" * 78)
    print("  SILVER BULLET ANALYSIS: Three-Layer FP Reduction")
    print("=" * 78)

    b = results["binary_or_rule"]
    q = results["quantitative_lr"]
    pd_c = results["pathology_directed_management"]["copula"]

    # ── Layer 1: Binary vs Quantitative ─────────────────────────────────
    print("\n--- LAYER 1: Binary OR  →  Quantitative LR ---")
    print(f"  {'Metric':<42} {'Binary OR':>12} {'Quant LR':>12}")
    print(f"  {'─'*68}")
    print(f"  {'Panel FP (any pathology, copula)':<42} {b['panel_fp_rate']:>11.1%} {q['any_pathology_fp']['copula']:>11.1%}")
    print(f"  {'Panel FP (ED-required only, copula)':<42} {b['panel_fp_rate']:>11.1%} {q['ed_only_fp']['copula']:>11.1%}")

    print(f"\n  Per-pathology sensitivity at calibrated thresholds:")
    print(f"  {'Pathology':<16} {'Binary':>8} {'Quant':>8} {'Threshold':>12}")
    print(f"  {'─'*46}")
    for path in COVERABLE:
        bs = b["per_pathology_sensitivity"].get(path, 0)
        qs = q["per_pathology_sensitivity"].get(path, 0)
        th = q["per_pathology_thresholds"].get(path, 0)
        print(f"  {path:<16} {bs:>7.1%} {qs:>7.1%} {th:>12.6f}")

    # ── Layer 2: Pathology-directed management ──────────────────────────
    print(f"\n--- LAYER 2: Pathology-Directed Management ---")
    print(f"  Instead of 'any positive → ED', classify by suspected pathology:")
    print(f"  • ACS / PE / AoD  → ED referral          {pd_c['ed_referral_rate']:>6.1%} of healthy")
    print(f"  • Pericarditis / AHF → Outpatient workup  {pd_c['outpatient_workup_rate']:>6.1%} of healthy")
    print(f"  • All below threshold → Safe discharge    {pd_c['discharge_rate']:>6.1%} of healthy")

    # ── Layer 3: HEAR pre-stratification ────────────────────────────────
    hf = results["hear_stratified_workflow"]
    pf = hf["per_1000_patients"]
    vb = hf["vs_binary_or"]
    print(f"\n--- LAYER 3: HEAR Pre-Stratification (per 1000 healthy patients) ---")
    print(f"  Low risk (HEAR 0–2):  {pf['low_risk_discharged']:>4} discharged without testing")
    print(f"  Moderate (HEAR 3–5):  {pf['tested']:>4} receive POC panel")
    print(f"    → ED referral:      {pf['test_triggered_ed_referrals']:>6.1f}")
    print(f"    → Outpatient:       {pf['test_triggered_outpatient_workups']:>6.1f}")
    print(f"    → Discharged:       {pf['test_discharged']:>6.1f}")
    print(f"  High risk (HEAR 6–8): {pf['high_risk_direct_ed']:>4} direct ED referral")
    print(f"  ──────────────────────")
    print(f"  TOTAL unnecessary ED: {pf['total_ed_referrals']:>6.1f}  ({pf['total_unnecessary_ed_pct']})")
    print(f"  vs Binary OR:         {vb['binary_total_ed']:>6.1f}")
    print(f"  ED referrals AVOIDED: {vb['ed_referrals_avoided']:>6.1f}  ({vb['reduction_pct']})")

    # ── Multivariate AUC ────────────────────────────────────────────────
    print(f"\n  Multivariate AUC (optimal 4-test combination):")
    for path in COVERABLE:
        auc = results["multivariate_auc_per_pathology"][path]
        t = results["theoretical_mv_roc"][path]
        print(f"  {path:<16} AUC = {auc:.3f}   Spec@95%Sens = {t['specificity_at_95pct_sensitivity']:.1%}")

    # ── Sensitivity sweep ───────────────────────────────────────────────
    print(f"\n  Sensitivity vs FP trade-off (copula):")
    print(f"  {'Sens':>6} {'Any FP':>8} {'ED FP':>8} {'ED Spec':>9}")
    print(f"  {'─'*33}")
    for row in results["sensitivity_fp_tradeoff"]:
        print(f"  {row['target_sensitivity']:>5.0%} {row['any_pathology_fp_copula']:>7.1%} "
              f"{row['ed_only_fp_copula']:>7.1%} {1-row['ed_only_fp_copula']:>8.1%}")

    # ── Summary ─────────────────────────────────────────────────────────
    s = results["fp_reduction_summary"]
    print(f"\n{'='*78}")
    print(f"  SUMMARY: FP Reduction Pipeline")
    print(f"{'='*78}")
    print(f"  Binary OR (current):           {s['binary_or_total_fp']} panel FP → ALL to ED")
    print(f"  + Quantitative LR:             {s['quant_lr_total_fp_copula']} any-action FP")
    print(f"    (ED-only FP:                 {s['quant_lr_ed_only_fp_copula']})")
    print(f"  + Pathology-directed mgmt:     {s['pathology_directed_ed_rate_copula']} → ED   "
          f"(rest to outpatient or discharged)")
    print(f"  + HEAR stratification:")
    print(f"    Unnecessary ED:              {s['hear_plus_quant_unnecessary_ed_per_1000']}/1000 "
          f"(vs {s['hear_plus_binary_unnecessary_ed_per_1000']}/1000 binary)")
    print(f"    ED referrals avoided:        {s['ed_referrals_avoided_per_1000']}/1000 patients")
    print(f"{'='*78}")
