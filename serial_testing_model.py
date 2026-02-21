"""
Serial Testing Model for Multi-Biomarker Diagnostic Panels
============================================================
Models time-dependent biomarker kinetics and the ESC 0h/1h/3h serial
testing protocol, extended from single-biomarker (troponin) to the
full multi-biomarker panel selected by the set-cover optimiser.

The SISTER ACT project (Safe and Integrated Serial Testing for Early
Rule-out of Acute Chest Threats) centres on serial POC testing in Dutch
primary care.  This module provides the computational backbone for
evaluating how serial measurements improve or degrade coverage, NPV,
and net benefit over time.

Key concepts
------------
1. **Biomarker kinetics** — Each biomarker's sensitivity is a function
   of time since symptom onset.  Troponin rises over 2–6 h; copeptin
   peaks immediately and falls; D-dimer and CRP are comparatively
   stable over the first few hours.

2. **Serial protocol** — At each time-point (0 h, 1 h, 3 h), all
   panel biomarkers are measured simultaneously.  The *cumulative*
   sensitivity at time T is the union of rule-out achieved by any
   measurement up to T (i.e. once ruled out, stays ruled out).

3. **Decision nodes** — After each measurement, a patient is triaged
   into one of three categories: rule-out, rule-in, or observe
   (requiring the next serial measurement).

Literature grounding
--------------------
* ESC 0/1h algorithm: Collet 2021 (ESC NSTE-ACS Guidelines)
* ESC 0/3h algorithm: Roffi 2016
* hs-cTnI kinetics:   Lipinski 2015, Mueller 2019, Chapman 2020
* Copeptin kinetics:   Keller 2011, Möckel 2015
* D-dimer kinetics:    Geersing 2012, Konstantinides 2020
* CRP kinetics:        Imazio 2011
* Primary care serial: Van Den Bulk 2023 (BMJ Open), HEART-GP 2025

Sources
-------
[ESC-2021]  Collet et al., Eur Heart J 2021;42:1289-1367
[ESC-2016]  Roffi et al., Eur Heart J 2016;37:267-315
[Chapman20] Chapman & Mills, Heart 2020;106:955-957
[Mueller19] Mueller et al., BMJ 2019;l6055
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from biomarker_coverage_matrix import (
    BIOMARKERS, PATHOLOGIES, PATHOLOGY_SHORT, PATHOLOGY_EPIDEMIOLOGY,
    BIOMARKER_META,
    build_coverage_matrix, build_specificity_matrix,
    build_beta_parameters,
)

logger = logging.getLogger(__name__)

# ─── Biomarker kinetics ─────────────────────────────────────────────────────
# For each biomarker we define a time-dependent sensitivity *multiplier*
# relative to the standard (peak / optimal-timing) sensitivity in
# COVERAGE_DATA.  The multiplier is 1.0 at the time-point where the
# published meta-analytic sensitivity was measured (typically ≥3 h for
# troponin, 0 h for copeptin).
#
# Structure: KINETICS[biomarker] = list of (time_hours, multiplier) tuples
#   - multiplier is applied to the base sensitivity
#   - intermediate times are linearly interpolated

@dataclass
class BiomarkerKinetics:
    """Time-dependent sensitivity profile for a single biomarker.

    Parameters
    ----------
    name : str
        Biomarker name (must match BIOMARKERS list).
    time_points : list of float
        Hours since symptom onset.
    multipliers : list of float
        Sensitivity multiplier at each time point (1.0 = published value).
    source : str
        Literature source for kinetic curve.
    note : str
        Clinical interpretation note.
    """
    name: str
    time_points: List[float]
    multipliers: List[float]
    source: str
    note: str = ""

    def multiplier_at(self, t_hours: float) -> float:
        """Linearly interpolate the sensitivity multiplier at time t."""
        if t_hours <= self.time_points[0]:
            return self.multipliers[0]
        if t_hours >= self.time_points[-1]:
            return self.multipliers[-1]
        return float(np.interp(t_hours, self.time_points, self.multipliers))


# ── Kinetic profiles ────────────────────────────────────────────────────────
# These encode the *ratio* of sensitivity at time T relative to the
# published pooled sensitivity (which typically reflects optimal timing).
#
# Sources:
#   hs-cTnI: Lipinski 2015, Chapman 2020 — sensitivity at 0h ~55-65%
#            of peak; at 1h ~75-85%; at 3h ~95-100%
#   Copeptin: Keller 2011, Möckel 2015 — peaks at 0h (immediate stress
#             release), falls by 50% at 6h
#   D-dimer: Geersing 2012 — relatively stable over first hours
#   NT-proBNP: Mueller 2019 — kinetics slow; stable over 0-3h
#   CRP: Imazio 2011 — very slow rise; 6-12h delay
#   H-FABP: Body 2015 — early release, peak at 2-4h
#   Myoglobin: Lipinski 2015 — early release, peak at 2-4h
#   Procalcitonin: Crawford 2019 — slow rise, 6-12h

BIOMARKER_KINETICS: Dict[str, BiomarkerKinetics] = {
    "hs-cTnI": BiomarkerKinetics(
        name="hs-cTnI",
        time_points=[0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0],
        multipliers= [0.60, 0.68, 0.78, 0.88, 0.95, 1.00, 1.00, 0.95],
        source="Lipinski 2015 [4]; Chapman 2020; ESC 0/1h data",
        note="Troponin rises 2-6h post onset. At 0h only ~60% of peak "
             "sensitivity. ESC 0/1h algorithm exploits the delta between "
             "0h and 1h measurements.",
    ),
    "D-dimer": BiomarkerKinetics(
        name="D-dimer",
        time_points=[0.0, 1.0, 3.0, 6.0, 12.0, 24.0],
        multipliers= [0.92, 0.95, 0.98, 1.00, 1.00, 0.98],
        source="Geersing 2012 [5]; Konstantinides 2020",
        note="D-dimer is relatively stable in first hours. Fibrin "
             "degradation products reach plateau within 1-2h.",
    ),
    "NT-proBNP": BiomarkerKinetics(
        name="NT-proBNP",
        time_points=[0.0, 1.0, 3.0, 6.0, 12.0, 24.0],
        multipliers= [0.90, 0.93, 0.97, 1.00, 1.00, 1.00],
        source="Mueller 2019 [7]; McDonagh 2021 ESC HF Guidelines",
        note="BNP/NT-proBNP have slow kinetics. Mostly stable at "
             "presentation for acute HF.",
    ),
    "CRP": BiomarkerKinetics(
        name="CRP",
        time_points=[0.0, 1.0, 3.0, 6.0, 12.0, 24.0],
        multipliers= [0.55, 0.60, 0.72, 0.85, 0.95, 1.00],
        source="Imazio 2011 [Circulation]; Koenig 1999",
        note="CRP is an acute-phase reactant with 6-12h delay in "
             "hepatic synthesis. Very low sensitivity at 0h for acute "
             "presentations. Reaches diagnostic levels by 12-24h.",
    ),
    "Copeptin": BiomarkerKinetics(
        name="Copeptin",
        time_points=[0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0],
        multipliers= [1.00, 0.95, 0.85, 0.70, 0.55, 0.40, 0.30],
        source="Keller 2011 [6]; Möckel 2015; Raskovalova 2013 [8]",
        note="Copeptin (C-terminal pro-vasopressin) is an immediate "
             "stress marker. Peak at symptom onset, falls rapidly. This "
             "is why it complements troponin early (0h) but not late.",
    ),
    "H-FABP": BiomarkerKinetics(
        name="H-FABP",
        time_points=[0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0],
        multipliers= [0.70, 0.82, 0.90, 1.00, 0.95, 0.80, 0.50],
        source="Body 2015 [9]",
        note="H-FABP is released early from damaged cardiomyocytes. "
             "Peaks at 2-4h, cleared renally by 12h.",
    ),
    "Myoglobin": BiomarkerKinetics(
        name="Myoglobin",
        time_points=[0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0],
        multipliers= [0.65, 0.78, 0.90, 1.00, 0.95, 0.75, 0.40],
        source="Lipinski 2015 [4]",
        note="Myoglobin is the fastest cardiac biomarker. Peaks 2-4h, "
             "cleared renally. Very low specificity.",
    ),
    "Procalcitonin": BiomarkerKinetics(
        name="Procalcitonin",
        time_points=[0.0, 1.0, 3.0, 6.0, 12.0, 24.0],
        multipliers= [0.40, 0.50, 0.65, 0.80, 0.95, 1.00],
        source="Crawford 2019 [10]; Schuetz 2017",
        note="PCT rises slowly with bacterial infection or severe systemic "
             "inflammation. Low early sensitivity (<6h). Mainly useful for "
             "infectious aetiology.",
    ),
}


# ─── Pathology-specific presentation context ────────────────────────────────
# Each pathology has a different typical delay from symptom onset to GP
# presentation.  This is critical for kinetics: a pericarditis patient
# presenting to the GP has had symptoms for 48+ hours (CRP already at
# peak), whereas an MI patient may arrive within 1-3 hours (troponin
# still rising).
#
# The published meta-analytic sensitivities were measured in hospital
# studies at a characteristic time after onset (typical_study_delay_hours).
# By calibrating against the study measurement time, we avoid the
# double-penalty bug of applying a raw kinetic multiplier on top of an
# already-temporally-mixed pooled sensitivity.

PATHOLOGY_PRESENTATION_CONTEXT: Dict[str, Dict] = {
    "ACS (STEMI/NSTEMI/UA)": {
        "typical_gp_delay_hours": 2.5,
        "typical_study_delay_hours": 5.0,
        "source": "Schols 2019; Jennings 2009",
        "note": "Troponin still rising at GP presentation (2-3h post "
                "onset). Published sensitivities from ED studies at "
                "mean ~5h post onset.",
    },
    "Pulmonary Embolism": {
        "typical_gp_delay_hours": 8.0,
        "typical_study_delay_hours": 10.0,
        "source": "Geersing 2012; Konstantinides 2020",
        "note": "D-dimer reaches plateau within 1-2h. Relatively "
                "stable at GP presentation.",
    },
    "Aortic Dissection": {
        "typical_gp_delay_hours": 1.5,
        "typical_study_delay_hours": 3.0,
        "source": "Erbel 2014 ESC AoD guidelines",
        "note": "Catastrophic onset, rapid presentation. D-dimer "
                "rises promptly with intimal flap.",
    },
    "Pericarditis / Myocarditis": {
        "typical_gp_delay_hours": 48.0,
        "typical_study_delay_hours": 72.0,
        "source": "Imazio 2011; Adler 2015 ESC guidelines",
        "note": "Sub-acute presentation: patients have days of chest "
                "pain before consulting GP. CRP already maximally "
                "elevated.",
    },
    "Pneumothorax (tension)": {
        "typical_gp_delay_hours": 0.5,
        "typical_study_delay_hours": 1.0,
        "source": "Clinical consensus",
        "note": "Acute onset, diagnosed clinically or by imaging, "
                "not by biomarkers.",
    },
    "Acute Heart Failure": {
        "typical_gp_delay_hours": 72.0,
        "typical_study_delay_hours": 72.0,
        "source": "McDonagh 2021 ESC HF guidelines",
        "note": "Gradual decompensation over days/weeks. NT-proBNP "
                "already chronically elevated at GP presentation.",
    },
}


# ─── Published serial algorithm performance ─────────────────────────────────
# For specific pathology-biomarker-protocol combinations, the serial
# algorithm performance has been directly validated in prospective
# studies.  These OVERRIDE the snapshot-based estimates because the
# ESC algorithms use delta criteria that capture information
# unavailable in individual timepoint measurements (S1 fix).

PUBLISHED_SERIAL_ALGORITHM_PERFORMANCE: Dict[tuple, Dict] = {
    ("ACS (STEMI/NSTEMI/UA)", "hs-cTnI", "0/1h"): {
        "sensitivity": 0.985,
        "source": "Boeddinghaus 2021; Mueller 2019; Collet 2021 ESC",
        "note": "ESC 0/1h hs-cTnI algorithm uses absolute thresholds "
                "AND 1h delta. Rule-out if 0h<5 ng/L OR (0h<12 AND "
                "delta_1h<3). Sensitivity 98.5%% for NSTEMI.",
    },
    ("ACS (STEMI/NSTEMI/UA)", "hs-cTnI", "0/3h"): {
        "sensitivity": 0.993,
        "source": "Roffi 2016; Mueller 2019",
        "note": "ESC 0/3h algorithm. Longer observation gives higher "
                "sensitivity. Standard in most European EDs.",
    },
}


# ─── Time-dependent coverage matrix builder ─────────────────────────────────

def build_time_coverage_matrix(gp_measurement_time: float) -> pd.DataFrame:
    """
    Build a coverage matrix at a specific GP measurement time.

    Unlike a naive multiplier approach, this accounts for pathology-
    specific delays between true symptom onset and GP presentation.

    For each (pathology, biomarker) cell::

        hours_since_onset = gp_delay + gp_measurement_time
        ratio = kinetics(hours_since_onset) / kinetics(study_delay)
        sensitivity = base_sensitivity × ratio

    This ensures:
      - Chronic presentations (pericarditis, AHF) retain full published
        sensitivity at GP time 0 (biomarkers already elevated for days).
      - Only genuinely time-sensitive biomarkers (troponin for MI) show
        reduced sensitivity at early GP measurement times.
      - The matrix converges to the published base values when
        gp_delay + gp_measurement_time ≈ study_delay.

    Parameters
    ----------
    gp_measurement_time : float
        Hours since GP presentation (0 = first measurement at GP).
    """
    base = build_coverage_matrix()
    result = base.copy()

    for pathology in PATHOLOGIES:
        ctx = PATHOLOGY_PRESENTATION_CONTEXT.get(pathology, {})
        gp_delay = ctx.get("typical_gp_delay_hours", 4.0)
        study_delay = ctx.get("typical_study_delay_hours", 6.0)

        for biomarker in BIOMARKERS:
            if biomarker not in BIOMARKER_KINETICS:
                continue

            kinetics = BIOMARKER_KINETICS[biomarker]

            # Time since true symptom onset at this GP measurement
            onset_time = gp_delay + gp_measurement_time

            # Published study was measured at study_delay after onset
            mult_at_study = kinetics.multiplier_at(study_delay)
            mult_at_now = kinetics.multiplier_at(onset_time)

            # Adjustment ratio (capped at 1.05 to avoid inflating
            # beyond published precision)
            if mult_at_study > 0.01:
                adjustment = min(mult_at_now / mult_at_study, 1.05)
            else:
                adjustment = 1.0

            result.loc[pathology, biomarker] = float(np.clip(
                base.loc[pathology, biomarker] * adjustment, 0.0, 1.0
            ))

    return result


# ─── Serial testing protocol ────────────────────────────────────────────────

@dataclass
class SerialTestResult:
    """Result of a serial testing protocol simulation."""
    protocol_name: str
    time_points: List[float]
    panel_biomarkers: List[str]
    tau: float

    # Per time-point results
    per_timepoint: List[Dict]

    # Cumulative (union over all time-points)
    cumulative_sensitivity: Dict[str, float]    # pathology → max sens seen
    cumulative_coverage: float                  # fraction with max_sens >= tau
    cumulative_pathologies_covered: List[str]
    cumulative_pathologies_uncovered: List[str]

    # Clinical flow
    n_patients_ruled_out_per_timepoint: List[float]   # fraction ruled out at each T
    n_patients_needing_serial: List[float]             # fraction needing next measurement
    total_time_to_decision_minutes: float              # weighted mean time to decision

    # Comparison
    gain_over_single: float     # coverage improvement vs single (0h) measurement


def simulate_serial_protocol(
    panel_biomarkers: List[str],
    time_points: List[float] = [0.0, 1.0, 3.0],
    tau: float = 0.90,
    protocol_name: str = "ESC 0/1/3h",
) -> SerialTestResult:
    """
    Simulate a serial testing protocol for the given panel.

    At each time-point, all panel biomarkers are measured simultaneously.
    For each pathology, the *best* sensitivity seen across all time-points
    (from any single biomarker in the panel) determines final coverage.

    Model
    -----
    Once a pathology is "ruled out" (max panel sensitivity ≥ τ at any
    time-point), it stays ruled out.  The serial protocol accumulates
    evidence but never retracts a rule-out decision.

    Parameters
    ----------
    panel_biomarkers : list of str
        Biomarkers in the panel (subset of BIOMARKERS).
    time_points : list of float
        Hours since symptom onset to measure.
    tau : float
        Sensitivity threshold for rule-out.
    protocol_name : str
        Human-readable name for the protocol.

    Returns
    -------
    SerialTestResult
    """
    n_pathologies = len(PATHOLOGIES)
    cumulative_best_sens = {p: 0.0 for p in PATHOLOGIES}
    per_timepoint = []
    cumulative_covered_at_t = []  # track how coverage grows

    for t in time_points:
        C_t = build_time_coverage_matrix(t)

        # At this time-point, compute per-pathology best-in-panel sensitivity
        tp_sens = {}
        tp_covered = []
        for pathology in PATHOLOGIES:
            best_sens_this_t = 0.0
            best_biomarker_this_t = None
            for bm in panel_biomarkers:
                if bm in C_t.columns:
                    s = C_t.loc[pathology, bm]
                    if s > best_sens_this_t:
                        best_sens_this_t = s
                        best_biomarker_this_t = bm
            tp_sens[pathology] = {
                'sensitivity': round(best_sens_this_t, 4),
                'best_biomarker': best_biomarker_this_t,
                'covered': best_sens_this_t >= tau,
            }
            # Update cumulative
            if best_sens_this_t > cumulative_best_sens[pathology]:
                cumulative_best_sens[pathology] = best_sens_this_t
            if cumulative_best_sens[pathology] >= tau:
                tp_covered.append(pathology)

        cum_coverage = len(tp_covered) / n_pathologies
        cumulative_covered_at_t.append(cum_coverage)

        per_timepoint.append({
            'time_hours': t,
            'per_pathology': tp_sens,
            'instantaneous_coverage': sum(
                1 for p in PATHOLOGIES
                if tp_sens[p]['sensitivity'] >= tau
            ) / n_pathologies,
            'cumulative_coverage': cum_coverage,
        })

    # ── Apply published serial algorithm overrides (S1 fix) ──
    # For specific pathology-biomarker-protocol combinations where the
    # serial algorithm has been directly validated (ESC 0/1h, 0/3h),
    # use the published sensitivity.  This captures delta-criterion
    # information that max(snapshot_0h, snapshot_1h) cannot.
    protocol_key = None
    if len(time_points) >= 2 and 0.0 in time_points:
        if 1.0 in time_points and 3.0 not in time_points:
            protocol_key = "0/1h"
        elif 3.0 in time_points:
            protocol_key = "0/3h"  # use conservative longer estimate

    if protocol_key:
        for pathology in PATHOLOGIES:
            for bm in panel_biomarkers:
                key = (pathology, bm, protocol_key)
                if key in PUBLISHED_SERIAL_ALGORITHM_PERFORMANCE:
                    override_sens = PUBLISHED_SERIAL_ALGORITHM_PERFORMANCE[key]["sensitivity"]
                    if override_sens > cumulative_best_sens[pathology]:
                        cumulative_best_sens[pathology] = override_sens

    # Final cumulative results
    final_covered = [
        p for p in PATHOLOGIES if cumulative_best_sens[p] >= tau
    ]
    final_uncovered = [
        p for p in PATHOLOGIES if cumulative_best_sens[p] < tau
    ]

    # Patient flow model: what fraction can be ruled out at each time-point?
    # Assume: a patient is "done" once ALL coverable pathologies have been
    # ruled out for that patient.  In a population sense, we model the
    # fraction of the (disease-positive) population that can be correctly
    # identified at each step.
    fraction_ruled_out = []
    prev_cumulative = 0.0
    for i, tp in enumerate(per_timepoint):
        new_ruled_out = tp['cumulative_coverage'] - prev_cumulative
        fraction_ruled_out.append(round(max(new_ruled_out, 0), 4))
        prev_cumulative = tp['cumulative_coverage']

    fraction_needing_serial = []
    for i in range(len(time_points)):
        remaining = 1.0 - cumulative_covered_at_t[i]
        fraction_needing_serial.append(round(remaining, 4))

    # Weighted mean time to decision (for covered pathologies)
    # Weight by incremental coverage gain at each time point
    total_gain = sum(fraction_ruled_out)
    if total_gain > 0:
        weighted_time = sum(
            t * g for t, g in zip(time_points, fraction_ruled_out)
        ) / total_gain
    else:
        weighted_time = time_points[-1]

    # Convert to minutes for clinical relevance (add test turnaround)
    max_tat = max(
        BIOMARKER_META[bm].time_to_result_min
        for bm in panel_biomarkers if bm in BIOMARKER_META
    )
    total_time_minutes = weighted_time * 60 + max_tat

    # Gain over single measurement (0h only)
    single_coverage = per_timepoint[0]['instantaneous_coverage']
    gain = cumulative_covered_at_t[-1] - single_coverage

    return SerialTestResult(
        protocol_name=protocol_name,
        time_points=time_points,
        panel_biomarkers=panel_biomarkers,
        tau=tau,
        per_timepoint=per_timepoint,
        cumulative_sensitivity={
            p: round(cumulative_best_sens[p], 4) for p in PATHOLOGIES
        },
        cumulative_coverage=cumulative_covered_at_t[-1],
        cumulative_pathologies_covered=final_covered,
        cumulative_pathologies_uncovered=final_uncovered,
        n_patients_ruled_out_per_timepoint=fraction_ruled_out,
        n_patients_needing_serial=fraction_needing_serial,
        total_time_to_decision_minutes=round(total_time_minutes, 1),
        gain_over_single=round(gain, 4),
    )


# ─── Compare protocols ──────────────────────────────────────────────────────

def compare_serial_protocols(
    panel_biomarkers: List[str],
    tau: float = 0.90,
) -> Dict:
    """
    Compare multiple serial testing strategies for the given panel.

    Protocols compared:
      1. Single measurement (0h only)
      2. ESC 0/1h rapid algorithm
      3. ESC 0/3h standard algorithm
      4. ESC 0/1/3h extended algorithm
      5. Single measurement at 3h (delayed presentation)

    Returns
    -------
    Dict with protocol comparison and clinical interpretation.
    """
    protocols = [
        ("Single 0h",        [0.0]),
        ("ESC 0/1h",         [0.0, 1.0]),
        ("ESC 0/3h",         [0.0, 3.0]),
        ("ESC 0/1/3h",       [0.0, 1.0, 3.0]),
        ("Delayed (3h only)", [3.0]),
    ]

    results = {}
    for name, tps in protocols:
        result = simulate_serial_protocol(
            panel_biomarkers=panel_biomarkers,
            time_points=tps,
            tau=tau,
            protocol_name=name,
        )
        results[name] = {
            'time_points': tps,
            'cumulative_coverage': round(result.cumulative_coverage, 4),
            'pathologies_covered': [
                PATHOLOGY_SHORT.get(p, p)
                for p in result.cumulative_pathologies_covered
            ],
            'pathologies_uncovered': [
                PATHOLOGY_SHORT.get(p, p)
                for p in result.cumulative_pathologies_uncovered
            ],
            'total_time_to_decision_minutes': result.total_time_to_decision_minutes,
            'gain_over_single': result.gain_over_single,
            'fraction_needing_serial': result.n_patients_needing_serial,
            'per_timepoint_detail': result.per_timepoint,
            'cumulative_sensitivity': result.cumulative_sensitivity,
        }

    # Find optimal protocol (best coverage, then fewest time-points,
    # but exclude single delayed measurement — it's not a real protocol)
    serial_protocols = {
        k: v for k, v in results.items() if len(v['time_points']) > 1
    }
    if serial_protocols:
        best_name = max(serial_protocols, key=lambda k: (
            serial_protocols[k]['cumulative_coverage'],
            -len(serial_protocols[k]['time_points']),
        ))
    else:
        best_name = max(results, key=lambda k: results[k]['cumulative_coverage'])

    # Interpret the ACS-specific serial benefit
    # (this is the key SISTER ACT question)
    # ACS-specific serial benefit (the key SISTER ACT metric)
    acs_key = "ACS (STEMI/NSTEMI/UA)"
    acs_sens_0h = results["Single 0h"]['cumulative_sensitivity'].get(acs_key, 0)
    # Use override-adjusted cumulative sensitivity (not snapshot)
    acs_sens_serial = results["ESC 0/1h"]['cumulative_sensitivity'].get(acs_key, 0)

    troponin_delta = None
    if acs_sens_0h is not None and acs_sens_serial is not None:
        troponin_delta = round(acs_sens_serial - acs_sens_0h, 4)

    interpretation = (
        f"The optimal protocol is '{best_name}' with "
        f"{results[best_name]['cumulative_coverage']:.0%} coverage. "
    )
    if troponin_delta and troponin_delta > 0:
        interpretation += (
            f"Serial testing adds {troponin_delta:.0%} absolute ACS sensitivity "
            f"(0h→1h delta), consistent with ESC 0/1h algorithm evidence. "
        )
    interpretation += (
        "CRP sensitivity improves substantially with serial testing (slow "
        "acute-phase rise), benefiting pericarditis/myocarditis rule-out. "
        "D-dimer and NT-proBNP are relatively stable, gaining little from "
        "serial measurement."
    )

    return {
        'panel': sorted(panel_biomarkers),
        'tau': tau,
        'protocols': results,
        'optimal_protocol': best_name,
        'acs_troponin_0h_1h_delta': troponin_delta,
        'clinical_interpretation': interpretation,
    }


# ─── Monte Carlo serial testing ─────────────────────────────────────────────

def monte_carlo_serial(
    panel_biomarkers: List[str],
    time_points: List[float] = [0.0, 1.0, 3.0],
    tau: float = 0.90,
    n_samples: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo simulation of serial testing with sensitivity uncertainty.

    For each iteration, sensitivity values are sampled from Beta
    distributions (fitted from published CIs), then multiplied by
    kinetic multipliers at each time-point.

    Returns coverage CIs, fraction needing serial CIs, and time-to-
    decision CIs.
    """
    rng = np.random.default_rng(seed)
    alpha_df, beta_df = build_beta_parameters()

    # Pre-compute kinetic adjustment ratios per (pathology, biomarker, timepoint)
    # Uses pathology-specific presentation offsets (C1/C2/S2/S3 fix)
    adjustment_ratios = {}
    for i, pathology in enumerate(PATHOLOGIES):
        pctx = PATHOLOGY_PRESENTATION_CONTEXT.get(pathology, {})
        gp_delay = pctx.get("typical_gp_delay_hours", 4.0)
        study_delay = pctx.get("typical_study_delay_hours", 6.0)
        for bm in panel_biomarkers:
            if bm in BIOMARKER_KINETICS:
                k = BIOMARKER_KINETICS[bm]
                mult_study = k.multiplier_at(study_delay)
                for t_idx, t in enumerate(time_points):
                    onset_time = gp_delay + t
                    mult_now = k.multiplier_at(onset_time)
                    ratio = min(mult_now / mult_study, 1.05) if mult_study > 0.01 else 1.0
                    adjustment_ratios[(i, bm, t_idx)] = ratio
            else:
                for t_idx in range(len(time_points)):
                    adjustment_ratios[(i, bm, t_idx)] = 1.0

    # Determine protocol key for published overrides
    mc_protocol_key = None
    if len(time_points) >= 2 and 0.0 in time_points:
        if 1.0 in time_points and 3.0 not in time_points:
            mc_protocol_key = "0/1h"
        elif 3.0 in time_points:
            mc_protocol_key = "0/3h"

    # Storage
    coverage_at_T = {t: [] for t in time_points}  # cumulative across timepoints
    cumulative_coverage_samples = []
    fraction_needing_serial_at_T = {t: [] for t in time_points}

    for _ in range(n_samples):
        # Sample base sensitivities from Beta distributions
        sampled_base = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
        for i, pathology in enumerate(PATHOLOGIES):
            for j, biomarker in enumerate(BIOMARKERS):
                a = alpha_df.iloc[i, j]
                b = beta_df.iloc[i, j]
                sampled_base[i, j] = rng.beta(a, b)

        # Simulate serial protocol with sampled sensitivities
        cumulative_best = np.zeros(len(PATHOLOGIES))
        for t_idx, t in enumerate(time_points):
            # Apply kinetics with pathology-specific offsets
            for i, pathology in enumerate(PATHOLOGIES):
                best_sens = 0.0
                for bm in panel_biomarkers:
                    j = BIOMARKERS.index(bm)
                    s_base = sampled_base[i, j]
                    ratio = adjustment_ratios[(i, bm, t_idx)]
                    s_t = min(s_base * ratio, 1.0)
                    best_sens = max(best_sens, s_t)
                cumulative_best[i] = max(cumulative_best[i], best_sens)

            # Apply published serial algorithm overrides
            if mc_protocol_key and t_idx == len(time_points) - 1:
                for i, pathology in enumerate(PATHOLOGIES):
                    for bm in panel_biomarkers:
                        key = (pathology, bm, mc_protocol_key)
                        if key in PUBLISHED_SERIAL_ALGORITHM_PERFORMANCE:
                            override = PUBLISHED_SERIAL_ALGORITHM_PERFORMANCE[key]["sensitivity"]
                            cumulative_best[i] = max(cumulative_best[i], override)

            # Cumulative coverage at this time-point
            inst_cov = np.mean(cumulative_best >= tau)
            coverage_at_T[t].append(float(inst_cov))

            remaining = 1.0 - inst_cov
            fraction_needing_serial_at_T[t].append(float(remaining))

        cumulative_coverage_samples.append(float(np.mean(cumulative_best >= tau)))

    # Summarise
    result = {
        'n_samples': n_samples,
        'panel': sorted(panel_biomarkers),
        'tau': tau,
        'protocol_time_points': time_points,
        'cumulative_coverage': {
            'mean': float(np.mean(cumulative_coverage_samples)),
            'std': float(np.std(cumulative_coverage_samples)),
            'ci_2.5': float(np.percentile(cumulative_coverage_samples, 2.5)),
            'ci_97.5': float(np.percentile(cumulative_coverage_samples, 97.5)),
        },
        'per_timepoint': {},
    }
    for t in time_points:
        arr = np.array(coverage_at_T[t])
        serial_arr = np.array(fraction_needing_serial_at_T[t])
        result['per_timepoint'][f'{t}h'] = {
            'cumulative_coverage_mean': float(np.mean(arr)),
            'cumulative_coverage_ci': [
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            ],
            'fraction_needing_serial_mean': float(np.mean(serial_arr)),
            'fraction_needing_serial_ci': [
                float(np.percentile(serial_arr, 2.5)),
                float(np.percentile(serial_arr, 97.5)),
            ],
        }

    return result


# ─── HEART Score Integration ────────────────────────────────────────────────
# The HEART score (History, ECG, Age, Risk factors, Troponin) is the
# primary CDR used in Dutch primary care for ACS risk stratification.
#
# Sources:
#   Six et al. 2008 (original HEART)
#   Van Den Berg & Body 2018, EHJACC — meta-analysis:
#     HEART ≤ 3 + negative troponin: sens 98.1%, spec 55.6% for MACE
#   Fernando et al. 2019, Acad Emerg Med — meta-analysis:
#     HEART ≤ 3: sens 96.7% (95.5-97.6), spec 47.0% (36.6-57.6)
#   Harskamp et al. 2021, Neth Heart J — HEART-GP:
#     Simplified HEART for GP: comparable performance
#   Van den Bulk et al. 2024, Ann Fam Med — systematic review:
#     Only Marburg Heart Score validated in primary care;
#     HEART score needs prospective GP validation

@dataclass
class HEARScoreDistribution:
    """Distribution of HEAR scores (without Troponin) in GP chest pain.

    HEAR = History + ECG + Age + Risk factors (0-8 scale).
    The Troponin component is deliberately EXCLUDED because it IS the
    POC test being evaluated.  Including it would create a conditional-
    independence violation (double-counting the same measurement).

    Sources:
      Fernando 2019 meta-analysis (adapted for GP without troponin)
      Harskamp 2021 HEART-GP simplified score
      Van den Bulk 2024: only Marburg Heart Score validated in GP

    Note: These distributions are ESTIMATED from ED-derived data adapted
    for primary care.  Prospective GP validation is a core SISTER ACT
    objective.
    """
    # Fraction of primary care chest pain patients in each HEAR category
    low_risk_fraction: float = 0.50     # HEAR 0-2
    moderate_risk_fraction: float = 0.35  # HEAR 3-5
    high_risk_fraction: float = 0.15    # HEAR 6-8

    # ACS prevalence conditional on HEAR (NOT HEART) category.
    # Higher than HEART-conditioned prevalences because the troponin
    # filter has not yet been applied.
    acs_prevalence_low: float = 0.015   # ~1.5% in HEAR-low
    acs_prevalence_moderate: float = 0.085  # ~8.5% in HEAR-moderate
    acs_prevalence_high: float = 0.280   # ~28% in HEAR-high

    # Any serious thoracic pathology (ACS + PE + AoD + Peri/Myo + AHF)
    any_serious_prevalence_low: float = 0.030     # ~3% (very low risk)
    any_serious_prevalence_moderate: float = 0.175  # ~17.5%
    any_serious_prevalence_high: float = 0.350     # ~35%

    source: str = (
        "Estimated from Fernando 2019; Harskamp 2021; adapted for "
        "Dutch GP WITHOUT troponin.  Requires prospective validation."
    )


def hear_score_stratified_analysis(
    panel_biomarkers: List[str],
    tau: float = 0.90,
    heart_dist: Optional[HEARScoreDistribution] = None,
) -> Dict:
    """
    Analyse the biomarker panel stratified by HEAR score risk categories.

    Key design (C3 fix): the T (Troponin) component is excluded from
    the CDR so that POC troponin is used ONLY as a diagnostic test.
    The GP calculates HEAR (0–8), then APPLIES the POC panel.

    For each HEAR category (low / moderate / high):
      - Different pre-test ACS probability (conditioned on HEAR)
      - Panel sensitivity/specificity applied via Bayes' theorem
      - Net benefit at category-appropriate decision thresholds

    Returns
    -------
    Dict with per-category analysis and overall value proposition.
    """
    if heart_dist is None:
        heart_dist = HEARScoreDistribution()

    # Get panel sensitivities and specificities for ACS
    C = build_coverage_matrix()
    S = build_specificity_matrix()
    acs_path = "ACS (STEMI/NSTEMI/UA)"

    # Best-in-panel sensitivity and specificity for ACS
    best_sens = 0.0
    best_bm = None
    for bm in panel_biomarkers:
        if bm in C.columns:
            s = C.loc[acs_path, bm]
            if s > best_sens:
                best_sens = s
                best_bm = bm
    best_spec = S.loc[acs_path, best_bm] if best_bm else 0.5

    categories = {
        'low_risk': {
            'hear_range': '0-2',
            'fraction_of_population': heart_dist.low_risk_fraction,
            'acs_prevalence': heart_dist.acs_prevalence_low,
            'clinical_action_without_test': 'Consider discharge (no POC)',
            'decision_threshold': 0.005,
            'poc_tested': False,  # Low-risk NOT tested
        },
        'moderate_risk': {
            'hear_range': '3-5',
            'fraction_of_population': heart_dist.moderate_risk_fraction,
            'acs_prevalence': heart_dist.acs_prevalence_moderate,
            'clinical_action_without_test': 'Observe / serial troponin',
            'decision_threshold': 0.02,
            'poc_tested': True,  # Key group — POC changes management
        },
        'high_risk': {
            'hear_range': '6-8',
            'fraction_of_population': heart_dist.high_risk_fraction,
            'acs_prevalence': heart_dist.acs_prevalence_high,
            'clinical_action_without_test': 'Immediate referral',
            'decision_threshold': 0.05,
            'poc_tested': False,  # High-risk always referred
        },
    }

    for cat_name, cat in categories.items():
        prev = cat['acs_prevalence']
        sens = best_sens
        spec = best_spec

        if cat.get('poc_tested', True):
            # Bayes' theorem — only meaningful for tested categories
            ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev)) if (sens * prev + (1 - spec) * (1 - prev)) > 0 else 0
            npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev) if (spec * (1 - prev) + (1 - sens) * prev) > 0 else 0
            post_test_neg = 1 - npv
            t = cat['decision_threshold']
            w = t / (1 - t) if t < 1 else float('inf')
            nb = sens * prev - (1 - spec) * (1 - prev) * w
        else:
            # Not tested — PPV/NPV/NB not applicable
            ppv = float('nan')
            npv = float('nan')
            post_test_neg = float('nan')
            nb = float('nan')

        # Clinical interpretation
        poc_adds_value = False
        if cat_name == 'low_risk':
            poc_adds_value = False  # discharged without POC
        elif cat_name == 'moderate_risk':
            poc_adds_value = True  # always the key group
        elif cat_name == 'high_risk':
            poc_adds_value = False  # always referred regardless

        cat.update({
            'sensitivity': round(sens, 4) if cat.get('poc_tested', True) else None,
            'specificity': round(spec, 4) if cat.get('poc_tested', True) else None,
            'ppv': round(ppv, 4) if cat.get('poc_tested', True) else None,
            'npv': round(npv, 6) if cat.get('poc_tested', True) else None,
            'post_test_prob_if_negative': round(post_test_neg, 6) if cat.get('poc_tested', True) else None,
            'net_benefit': round(nb, 6) if cat.get('poc_tested', True) else None,
            'poc_adds_clinical_value': poc_adds_value,
        })

    # Referral reduction estimate:
    # Without POC: moderate + high-risk patients referred (50%)
    # With POC: only positive-test moderate + high-risk patients referred
    without_poc_referral = (
        heart_dist.moderate_risk_fraction + heart_dist.high_risk_fraction
    )
    # With POC: moderate-risk referred only if test positive
    # P(test positive | moderate-risk) = sens*prev + (1-spec)*(1-prev)
    p_pos_moderate = (
        best_sens * categories['moderate_risk']['acs_prevalence'] +
        (1 - best_spec) * (1 - categories['moderate_risk']['acs_prevalence'])
    )
    with_poc_referral = (
        heart_dist.moderate_risk_fraction * p_pos_moderate +
        heart_dist.high_risk_fraction * 1.0  # always referred
    )
    referral_reduction = without_poc_referral - with_poc_referral

    return {
        'panel': sorted(panel_biomarkers),
        'acs_biomarker': best_bm,
        'acs_sensitivity': round(best_sens, 4),
        'acs_specificity': round(best_spec, 4),
        'cdr_model': 'HEAR (without Troponin) — T excluded to avoid double-counting',
        'hear_score_distribution': heart_dist.source,
        'per_category': categories,
        'overall': {
            'referral_rate_without_poc': round(without_poc_referral, 4),
            'referral_rate_with_poc': round(with_poc_referral, 4),
            'referral_reduction_absolute': round(referral_reduction, 4),
            'referral_reduction_relative': round(
                referral_reduction / without_poc_referral, 4
            ) if without_poc_referral > 0 else 0,
        },
        'clinical_interpretation': (
            f"HEAR-stratified (without T) multi-biomarker POC testing could "
            f"reduce referrals by {referral_reduction:.0%} (absolute) in "
            f"primary care. The POC troponin contributes BOTH to completing "
            f"the HEART score (adding the T component) AND to the multi-test "
            f"panel coverage. For HEAR 3-5 (moderate risk): a negative panel "
            f"yields NPV {categories['moderate_risk']['npv']:.4f} for ACS. "
            f"HEAR 0-2 patients are discharged without testing (low pre-test "
            f"probability). HEAR 6-8 patients are referred immediately."
        ),
        'limitation': (
            "This model assumes biomarker sensitivity is independent of HEAR "
            "score category. In reality, higher HEAR scores correlate with "
            "larger infarcts and higher troponin, so sensitivity may be "
            "slightly higher in high-risk patients. The HEAR score "
            "distribution is estimated from ED data — prospective GP "
            "validation is essential (a core SISTER ACT objective). "
            "Van den Bulk 2024: only Marburg Heart Score has been "
            "prospectively validated in Dutch primary care."
        ),
    }


# ─── Dutch GP Context ───────────────────────────────────────────────────────

@dataclass
class DutchGPContext:
    """Epidemiological and operational parameters for Dutch primary care.

    Sources:
      - Schols 2019 (Ann Fam Med) flash-mob study
      - Harskamp 2018 (BJGP Open) nationwide survey
      - Van Den Bulk 2023 (BMJ Open) trial protocol
      - NZa 2024 (tariffs)
      - CBS 2024 (population statistics)
    """
    # Population
    avg_patients_per_gp_practice: int = 2350
    chest_pain_consultations_per_1000_per_year: float = 15.0
    # → ~35 chest pain patients per GP per year

    # Prevalence of serious pathology in GP chest pain
    acs_prevalence_gp: float = 0.035           # 3.5% — Hoorweg 2017
    any_serious_pathology_prevalence: float = 0.12  # ~12% combined

    # Current referral practice
    referral_rate_current: float = 0.50  # 50% of chest pain referred (Schols 2019)
    unnecessary_referral_fraction: float = 0.70  # ~70% of referrals are negative

    # Costs (EUR, 2024)
    gp_consultation_eur: float = 10.39   # NZa 2024 kort consult
    ed_visit_eur: float = 265.0          # average ED visit cost
    ambulance_eur: float = 600.0         # average transport
    overnight_observation_eur: float = 750.0

    # POC testing estimates (EUR, approximate list prices)
    poc_panel_4test_cost_eur: float = 36.00   # 4-test panel (hs-cTnI + D-dimer + NT-proBNP + CRP)

    source: str = "Schols 2019; Harskamp 2018; Van Den Bulk 2023; NZa 2024"


def dutch_patient_flow_analysis(
    panel_biomarkers: List[str],
    tau: float = 0.90,
    ctx: Optional[DutchGPContext] = None,
) -> Dict:
    """
    Model patient flow and cost impact of POC panel in Dutch primary care.

    Two scenarios are compared:
      1. Current practice: GP assessment + referral to ED
      2. With POC panel: GP assessment + HEART score + POC panel,
         referral only if test positive or high-risk

    Returns
    -------
    Dict with patient flow, referrals avoided, cost analysis.
    """
    if ctx is None:
        ctx = DutchGPContext()

    # Panel performance
    C = build_coverage_matrix()
    S = build_specificity_matrix()

    # Per 1000 chest pain patients
    n = 1000

    # Current practice (no POC)
    n_referred_current = n * ctx.referral_rate_current
    n_unnecessary_current = n_referred_current * ctx.unnecessary_referral_fraction
    n_missed_current = n * ctx.any_serious_pathology_prevalence * 0.15

    cost_current = (
        n * ctx.gp_consultation_eur +
        n_referred_current * ctx.ed_visit_eur +
        n_referred_current * 0.3 * ctx.ambulance_eur +  # 30% arrive by ambulance
        n_referred_current * 0.4 * ctx.overnight_observation_eur  # 40% admitted overnight
    )

    # With POC panel
    # Step 1: HEAR score stratification (without Troponin — C3 fix)
    heart = HEARScoreDistribution()
    n_low = n * heart.low_risk_fraction
    n_mod = n * heart.moderate_risk_fraction
    n_high = n * heart.high_risk_fraction

    # Step 2: POC testing for moderate-risk (HEAR 3-5) patients
    # Low-risk: discharge (no POC needed — low pre-test probability)
    # High-risk: refer immediately (no POC changes decision)
    # Moderate-risk: POC panel determines referral
    #
    # Clinical model: the GP uses HEAR (no T) for stratification,
    # then applies POC panel (including troponin) to moderate-risk.
    # The POC troponin serves DUAL purpose: completes the HEART score
    # (adding the T component) AND contributes to panel coverage.

    # Primary suspicion distribution in moderate-risk chest pain (estimated)
    suspicion_weights = {
        "ACS (STEMI/NSTEMI/UA)":     0.60,  # most common suspect
        "Pulmonary Embolism":        0.15,
        "Pericarditis / Myocarditis": 0.10,
        "Acute Heart Failure":       0.10,
        "Aortic Dissection":         0.03,
        "Pneumothorax (tension)":    0.02,
    }

    # Biomarker routing: which test is used for each suspicion?
    suspicion_biomarker = {
        "ACS (STEMI/NSTEMI/UA)":     "hs-cTnI",
        "Pulmonary Embolism":        "D-dimer",
        "Aortic Dissection":         "D-dimer",
        "Pericarditis / Myocarditis": "CRP",
        "Acute Heart Failure":       "NT-proBNP",
        "Pneumothorax (tension)":    None,  # clinical/imaging diagnosis
    }

    # Weighted effective FP rate across suspicion types
    # For each suspicion: P(FP) = 1 - specificity of the relevant test
    weighted_fp_rate = 0.0
    for pathology, weight in suspicion_weights.items():
        bm = suspicion_biomarker.get(pathology)
        if bm and bm in S.columns:
            spec = S.loc[pathology, bm]
            fp_rate = 1.0 - spec
        else:
            fp_rate = 0.0  # no POC test → no FP
        weighted_fp_rate += weight * fp_rate

    # Average sensitivity across covered pathologies (weighted by prevalence)
    avg_sens = 0.0
    total_prev = 0.0
    for p in PATHOLOGIES:
        prev_p = PATHOLOGY_EPIDEMIOLOGY[p].prevalence
        best_s = max(C.loc[p, bm] for bm in panel_biomarkers)
        avg_sens += best_s * prev_p
        total_prev += prev_p
    avg_sens /= total_prev if total_prev > 0 else 1

    # Moderate-risk referral with POC
    n_mod_true_pos_detected = n_mod * ctx.any_serious_pathology_prevalence * avg_sens
    n_mod_false_pos = n_mod * (1 - ctx.any_serious_pathology_prevalence) * weighted_fp_rate
    n_mod_referred = n_mod_true_pos_detected + n_mod_false_pos

    n_referred_poc = n_high + n_mod_referred  # low-risk discharged, high always referred
    n_unnecessary_poc = n_mod_false_pos  # high-risk FPs not counted (they need referral anyway)

    # Missed cases with POC
    # C4 FIX: Low-risk patients are NOT TESTED.  Their miss rate is
    # the disease prevalence in that HEAR category (all disease missed),
    # NOT panel sensitivity × prevalence (which would only apply if tested).
    n_missed_poc_low = n_low * heart.any_serious_prevalence_low  # untested → all disease missed
    n_missed_poc_mod = n_mod * heart.any_serious_prevalence_moderate * (1 - avg_sens)  # tested → FN only
    n_missed_poc = n_missed_poc_low + n_missed_poc_mod  # high-risk always referred

    cost_poc = (
        n * ctx.gp_consultation_eur +
        n_mod * ctx.poc_panel_4test_cost_eur +  # POC only for moderate-risk
        n_referred_poc * ctx.ed_visit_eur +
        n_referred_poc * 0.2 * ctx.ambulance_eur +  # fewer ambulances (planned referral)
        n_referred_poc * 0.3 * ctx.overnight_observation_eur  # fewer admissions
    )

    return {
        'per_1000_chest_pain_patients': {
            'current_practice': {
                'referred': round(n_referred_current, 1),
                'unnecessary_referrals': round(n_unnecessary_current, 1),
                'missed_serious_pathology': round(n_missed_current, 1),
                'total_cost_eur': round(cost_current, 0),
            },
            'with_poc_panel': {
                'referred': round(n_referred_poc, 1),
                'unnecessary_referrals': round(n_unnecessary_poc, 1),
                'missed_serious_pathology': round(n_missed_poc, 1),
                'total_cost_eur': round(cost_poc, 0),
                'poc_tests_performed': round(n_mod, 0),
            },
        },
        'impact': {
            'referrals_avoided_per_1000': round(n_referred_current - n_referred_poc, 1),
            'unnecessary_referrals_avoided_per_1000': round(
                n_unnecessary_current - n_unnecessary_poc, 1
            ),
            'additional_missed_cases_per_1000': round(
                n_missed_poc - n_missed_current, 2
            ),
            'cost_saving_per_1000_eur': round(cost_current - cost_poc, 0),
            'cost_per_referral_avoided_eur': round(
                (n_mod * ctx.poc_panel_4test_cost_eur) /
                max(n_referred_current - n_referred_poc, 1), 0
            ) if n_referred_poc < n_referred_current else None,
        },
        'per_gp_practice_per_year': {
            'chest_pain_patients': round(
                ctx.avg_patients_per_gp_practice *
                ctx.chest_pain_consultations_per_1000_per_year / 1000, 0
            ),
            'referrals_avoided': round(
                ctx.avg_patients_per_gp_practice *
                ctx.chest_pain_consultations_per_1000_per_year / 1000 *
                (n_referred_current - n_referred_poc) / n, 1
            ),
            'cost_saving_eur': round(
                ctx.avg_patients_per_gp_practice *
                ctx.chest_pain_consultations_per_1000_per_year / 1000 *
                (cost_current - cost_poc) / n, 0
            ),
        },
        'dutch_context': {
            'source': ctx.source,
            'gp_practices_netherlands': 5000,
            'national_referrals_avoided_per_year': round(
                5000 *
                ctx.avg_patients_per_gp_practice *
                ctx.chest_pain_consultations_per_1000_per_year / 1000 *
                (n_referred_current - n_referred_poc) / n, 0
            ),
            'national_cost_saving_eur_per_year': round(
                5000 *
                ctx.avg_patients_per_gp_practice *
                ctx.chest_pain_consultations_per_1000_per_year / 1000 *
                (cost_current - cost_poc) / n, 0
            ),
        },
        'clinical_interpretation': (
            "POC multi-biomarker testing in Dutch primary care, stratified by "
            "HEAR score (without Troponin), could substantially reduce "
            "unnecessary ED referrals. The greatest impact is in HEAR 3-5 "
            "(moderate-risk) patients, who currently face precautionary referral. "
            "Cost savings accrue from avoided ED visits, ambulance transports, and "
            "overnight observations. HEAR-low patients are safely discharged "
            "without testing (honest trade-off: slightly higher miss rate vs "
            "large referral reduction). Safety maintained through high panel "
            "sensitivity for covered pathologies."
        ),
        'limitations': [
            "Model assumes HEAR score and biomarker results are independent (simplified).",
            "Cost estimates based on 2024 NZa tariffs, actual costs vary by region.",
            "HEAR-low discharge: honest increase in missed cases vs current practice "
            "— quantified explicitly as HEAR-low disease prevalence (~3%).",
            "FP rate modelled as suspicion-weighted per-test rate (not independent union).",
            "Serial testing benefit (SISTER ACT core question) not yet modelled "
            "in the cost analysis — this is a single-timepoint estimate.",
            "HEAR score NOT validated in Dutch primary care (Van den Bulk 2024). "
            "Prospective validation is essential — a core SISTER ACT objective.",
        ],
    }


# ─── Comprehensive serial analysis runner ───────────────────────────────────

def run_serial_analysis(
    panel_biomarkers: Optional[List[str]] = None,
    tau: float = 0.90,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Run the full serial testing and CDR analysis suite.

    Steps:
      1. Biomarker kinetic profiles
      2. Serial protocol comparison (0h, 0/1h, 0/3h, 0/1/3h)
      3. Monte Carlo serial testing with uncertainty
      4. HEART score stratified analysis
      5. Dutch GP patient flow and cost model

    Parameters
    ----------
    panel_biomarkers : list of str, optional
        Biomarkers in the panel. Defaults to the optimal 4-test panel.
    tau : float
        Sensitivity threshold.
    output_dir : str, optional
        Directory to write JSON results.

    Returns
    -------
    Dict with all sub-analyses.
    """
    if panel_biomarkers is None:
        panel_biomarkers = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]

    print("\n" + "=" * 90)
    print("SERIAL TESTING & CDR ANALYSIS — SISTER ACT EXTENSION")
    print("=" * 90)

    # 1. Kinetic profiles
    print("\n[SERIAL 1/5] Biomarker kinetic profiles...")
    kinetics_summary = {}
    for bm in panel_biomarkers:
        if bm in BIOMARKER_KINETICS:
            k = BIOMARKER_KINETICS[bm]
            kinetics_summary[bm] = {
                'time_points': k.time_points,
                'multipliers': k.multipliers,
                'multiplier_at_0h': round(k.multiplier_at(0.0), 3),
                'multiplier_at_1h': round(k.multiplier_at(1.0), 3),
                'multiplier_at_3h': round(k.multiplier_at(3.0), 3),
                'source': k.source,
                'note': k.note,
            }
            print(f"  {bm:15s}: 0h={k.multiplier_at(0.0):.0%}  "
                  f"1h={k.multiplier_at(1.0):.0%}  "
                  f"3h={k.multiplier_at(3.0):.0%}  "
                  f"(of peak sensitivity)")

    # 2. Protocol comparison
    print("\n[SERIAL 2/5] Comparing serial testing protocols...")
    protocols = compare_serial_protocols(panel_biomarkers, tau=tau)
    for name, data in protocols['protocols'].items():
        covered = ', '.join(data['pathologies_covered'])
        uncovered = ', '.join(data['pathologies_uncovered']) or 'NONE'
        print(f"  {name:20s}: coverage={data['cumulative_coverage']:.0%}  "
              f"time={data['total_time_to_decision_minutes']:.0f}min  "
              f"gaps={uncovered}")
    print(f"  Optimal: {protocols['optimal_protocol']}")
    if protocols['acs_troponin_0h_1h_delta']:
        print(f"  ACS troponin serial gain (0h→1h): "
              f"+{protocols['acs_troponin_0h_1h_delta']:.0%}")

    # 3. Monte Carlo serial
    print("\n[SERIAL 3/5] Monte Carlo serial testing (n=2000)...")
    mc_serial = monte_carlo_serial(
        panel_biomarkers,
        time_points=[0.0, 1.0, 3.0],
        tau=tau,
        n_samples=2000,
    )
    print(f"  Cumulative coverage (0/1/3h): "
          f"{mc_serial['cumulative_coverage']['mean']:.1%} "
          f"[{mc_serial['cumulative_coverage']['ci_2.5']:.1%}, "
          f"{mc_serial['cumulative_coverage']['ci_97.5']:.1%}]")
    for t_key, t_data in mc_serial['per_timepoint'].items():
        print(f"  At {t_key}: coverage={t_data['cumulative_coverage_mean']:.1%}  "
              f"needing serial={t_data['fraction_needing_serial_mean']:.1%}")

    # 4. HEAR score stratified (without Troponin)
    print("\n[SERIAL 4/5] HEAR score stratified analysis (no T)...")
    hear = hear_score_stratified_analysis(panel_biomarkers, tau=tau)
    for cat_name, cat in hear['per_category'].items():
        npv_str = f"{cat['npv']:.4f}" if cat.get('npv') is not None else 'N/A'
        nb_str = f"{cat['net_benefit']:.4f}" if cat.get('net_benefit') is not None else 'N/A'
        print(f"  {cat_name:15s} (HEAR {cat['hear_range']}): "
              f"prev={cat['acs_prevalence']:.1%}  "
              f"NPV={npv_str}  "
              f"NB={nb_str}  "
              f"POC value={'YES' if cat['poc_adds_clinical_value'] else 'no'}")
    print(f"  Referral reduction: {hear['overall']['referral_reduction_absolute']:.0%} "
          f"({hear['overall']['referral_reduction_relative']:.0%} relative)")

    # 5. Dutch GP patient flow
    print("\n[SERIAL 5/5] Dutch GP patient flow & cost analysis...")
    flow = dutch_patient_flow_analysis(panel_biomarkers, tau=tau)
    curr = flow['per_1000_chest_pain_patients']['current_practice']
    poc = flow['per_1000_chest_pain_patients']['with_poc_panel']
    print(f"  Per 1000 patients:")
    print(f"    Current: {curr['referred']:.0f} referred, "
          f"{curr['unnecessary_referrals']:.0f} unnecessary, "
          f"€{curr['total_cost_eur']:,.0f}")
    print(f"    With POC: {poc['referred']:.0f} referred, "
          f"{poc['unnecessary_referrals']:.0f} unnecessary, "
          f"€{poc['total_cost_eur']:,.0f}")
    print(f"  Impact:")
    print(f"    Referrals avoided: {flow['impact']['referrals_avoided_per_1000']:.0f}/1000")
    print(f"    Cost saving: €{flow['impact']['cost_saving_per_1000_eur']:,.0f}/1000 patients")
    gp = flow['per_gp_practice_per_year']
    print(f"  Per GP practice/year: {gp['referrals_avoided']:.0f} referrals avoided, "
          f"€{gp['cost_saving_eur']:,.0f} saved")
    nl = flow['dutch_context']
    print(f"  Netherlands (5000 practices): "
          f"{nl['national_referrals_avoided_per_year']:,} referrals avoided, "
          f"€{nl['national_cost_saving_eur_per_year']:,.0f}/year saved")

    # Combine all results
    full_results = {
        'kinetics': kinetics_summary,
        'protocol_comparison': protocols,
        'monte_carlo_serial': mc_serial,
        'hear_score_analysis': hear,
        'dutch_patient_flow': flow,
    }

    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "serial_testing_protocols.json"), 'w') as f:
            json.dump(protocols, f, indent=2, default=str)
        with open(os.path.join(output_dir, "monte_carlo_serial.json"), 'w') as f:
            json.dump(mc_serial, f, indent=2, default=str)
        with open(os.path.join(output_dir, "hear_score_analysis.json"), 'w') as f:
            json.dump(hear, f, indent=2, default=str)
        with open(os.path.join(output_dir, "dutch_patient_flow.json"), 'w') as f:
            json.dump(flow, f, indent=2, default=str)

    return full_results


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    results = run_serial_analysis(output_dir=output_dir)
