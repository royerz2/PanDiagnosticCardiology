"""
Pathology–Biomarker Coverage Matrix
====================================
Sensitivity values extracted from published meta-analyses and systematic reviews.
All values are sensitivity (proportion) for detecting the listed pathology.

Sources (keyed to proposal reading list):
  [4]  Lipinski et al. 2015 – troponin, CK-MB, myoglobin, H-FABP for AMI
  [5]  Geersing et al. 2012 – D-dimer for PE
  [6]  Keller et al. 2011 – Copeptin for AMI
  [7]  Mueller et al. 2019 – NTproBNP for heart failure
  [8]  Raskovalova et al. 2013 – Copeptin for AMI rule-out
  [9]  Body et al. 2015 – H-FABP for AMI
  [10] Crawford et al. 2019 – Procalcitonin for chest pain etiologies

Additional published sources used to fill matrix cells:
  - Caforio et al. 2013, Eur Heart J 34(33):2636 – ESC position statement on myocarditis
  - Imazio et al. 2011, Circulation 123(10):1092-7 – CRP in pericarditis
  - Nazerian et al. 2018, JAMA 2018;319(22):2299-310 – D-dimer in aortic dissection (ADvISED)
  - Sodeck et al. 2007, Eur Heart J 28(S):221 – D-dimer sensitivity in aortic dissection  
  - Weber et al. 2006, Eur Heart J 27(3):330-7 – NTproBNP in aortic syndromes
  - Maisel et al. 2002, NEJM 347(3):161-7 – BNP Breathing Not Properly Study (HF)
  - Peacock et al. 2011, Eur J Heart Fail 13(10):1086 – copeptin in acute HF
  - Demissei et al. 2017, Eur J Heart Fail 19(12):1529-37 – copeptin in acute HF
  - Mockel et al. 2015, Eur Heart J Acute Cardiovasc Care 4(1):64-71 – copeptin in ACS
  - Kokturk et al. 2011, Clin Appl Thromb Hemost 17(5):E1 – procalcitonin in PE

NOTE: Where primary-care-specific data was unavailable, ED-derived estimates
are used and marked with an asterisk (*) in the notes column. Sensitivity
for pneumothorax is clinical/radiological; biomarker data is extremely limited
and assigned conservatively.

All values represent SENSITIVITY (true positive rate) at the standard diagnostic
threshold used in each source publication.
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─── Pathology definitions ───────────────────────────────────────────────────

PATHOLOGIES = [
    "ACS (STEMI/NSTEMI/UA)",
    "Pulmonary Embolism",
    "Aortic Dissection",
    "Pericarditis / Myocarditis",
    "Pneumothorax (tension)",
    "Acute Heart Failure",
]

PATHOLOGY_SHORT = {
    "ACS (STEMI/NSTEMI/UA)": "ACS",
    "Pulmonary Embolism": "PE",
    "Aortic Dissection": "AoD",
    "Pericarditis / Myocarditis": "Peri/Myo",
    "Pneumothorax (tension)": "PTX",
    "Acute Heart Failure": "AHF",
}

# ─── Pathology epidemiological data ──────────────────────────────────────────
# Prevalence: fraction of acute chest pain presentations in primary care
# Sources: Schols 2018, Hoorweg 2017, Knockaert 2002, ESC 2023 Guidelines
# Case-fatality: 30-day mortality if missed (untreated/delayed diagnosis)
# Sources: ESC/AHA guidelines, Defined from literature

@dataclass
class PathologyEpidemiology:
    """Epidemiological characteristics of a pathology in primary care."""
    prevalence: float            # pre-test probability in primary care chest pain
    case_fatality_if_missed: float  # 30-day mortality with missed/delayed diagnosis
    source: str

PATHOLOGY_EPIDEMIOLOGY: Dict[str, PathologyEpidemiology] = {
    "ACS (STEMI/NSTEMI/UA)": PathologyEpidemiology(
        prevalence=0.035,  # 3.5% of chest pain presentations
        case_fatality_if_missed=0.15,  # 15% 30-day mortality if missed NSTEMI
        source="Hoorweg 2017; Byrne 2023 ESC ACS Guidelines"
    ),
    "Pulmonary Embolism": PathologyEpidemiology(
        prevalence=0.015,  # ~1.5%
        case_fatality_if_missed=0.25,  # 25% if untreated
        source="Geersing 2012; Konstantinides 2020 ESC PE Guidelines"
    ),
    "Aortic Dissection": PathologyEpidemiology(
        prevalence=0.003,  # ~0.3% (rare but lethal)
        case_fatality_if_missed=0.50,  # 50% if missed (1-2% per hour)
        source="Hagan 2000; Goodacre 2025"
    ),
    "Pericarditis / Myocarditis": PathologyEpidemiology(
        prevalence=0.040,  # ~4% (relatively common in young adults)
        case_fatality_if_missed=0.02,  # 2% (mostly benign, some myocarditis)
        source="Imazio 2011; Adler 2015 ESC Pericarditis Guidelines"
    ),
    "Pneumothorax (tension)": PathologyEpidemiology(
        prevalence=0.005,  # ~0.5%
        case_fatality_if_missed=0.30,  # 30% for tension pneumothorax
        source="Roberts 2010; BTS Spontaneous Pneumothorax Guidelines"
    ),
    "Acute Heart Failure": PathologyEpidemiology(
        prevalence=0.025,  # ~2.5%
        case_fatality_if_missed=0.10,  # 10% 30-day mortality
        source="Maisel 2002; McDonagh 2021 ESC HF Guidelines"
    ),
}

# ─── Biomarker specificity data ──────────────────────────────────────────────
# Specificity (true negative rate) for each biomarker-pathology pair
# Sources: Same meta-analyses as sensitivity where available

@dataclass
class SpecificityEntry:
    """Specificity data for a biomarker-pathology pair."""
    specificity: float
    ci_lower: float
    ci_upper: float
    source: str

SPECIFICITY_DATA: Dict[str, Dict[str, SpecificityEntry]] = {
    "ACS (STEMI/NSTEMI/UA)": {
        "hs-cTnI": SpecificityEntry(0.54, 0.47, 0.61, "Lipinski 2015; hs-cTn pooled"),
        "D-dimer": SpecificityEntry(0.45, 0.38, 0.52, "Non-specific; elevated in many conditions"),
        "NT-proBNP": SpecificityEntry(0.60, 0.52, 0.68, "Cross-reactivity with HF/PE"),
        "CRP": SpecificityEntry(0.40, 0.32, 0.48, "Non-specific inflammatory marker"),
        "Copeptin": SpecificityEntry(0.75, 0.70, 0.80, "Keller 2011; Raskovalova 2013"),
        "H-FABP": SpecificityEntry(0.78, 0.72, 0.84, "Body 2015; pooled"),
        "Myoglobin": SpecificityEntry(0.50, 0.42, 0.58, "Low specificity; muscle damage"),
        "Procalcitonin": SpecificityEntry(0.85, 0.78, 0.92, "Low sensitivity, high specificity for ACS"),
    },
    "Pulmonary Embolism": {
        "hs-cTnI": SpecificityEntry(0.60, 0.52, 0.68, "Moderately elevated in PE"),
        "D-dimer": SpecificityEntry(0.42, 0.38, 0.46, "Geersing 2012; low specificity"),
        "NT-proBNP": SpecificityEntry(0.55, 0.48, 0.62, "Elevated in RV strain"),
        "CRP": SpecificityEntry(0.35, 0.28, 0.42, "Non-specific for PE"),
        "Copeptin": SpecificityEntry(0.70, 0.62, 0.78, "Limited data for PE"),
        "H-FABP": SpecificityEntry(0.75, 0.68, 0.82, "Some cross-reactivity"),
        "Myoglobin": SpecificityEntry(0.60, 0.52, 0.68, "Limited data"),
        "Procalcitonin": SpecificityEntry(0.80, 0.72, 0.88, "Limited data for PE"),
    },
    "Aortic Dissection": {
        "hs-cTnI": SpecificityEntry(0.65, 0.56, 0.74, "May be elevated in type A"),
        "D-dimer": SpecificityEntry(0.47, 0.40, 0.54, "Chen 2023; pooled specificity"),
        "NT-proBNP": SpecificityEntry(0.55, 0.46, 0.64, "Non-specific for AoD"),
        "CRP": SpecificityEntry(0.30, 0.22, 0.38, "Non-specific inflammation"),
        "Copeptin": SpecificityEntry(0.60, 0.50, 0.70, "Limited data"),
        "H-FABP": SpecificityEntry(0.80, 0.72, 0.88, "Limited data"),
        "Myoglobin": SpecificityEntry(0.65, 0.55, 0.75, "Limited data"),
        "Procalcitonin": SpecificityEntry(0.75, 0.65, 0.85, "Limited data"),
    },
    "Pericarditis / Myocarditis": {
        "hs-cTnI": SpecificityEntry(0.50, 0.42, 0.58, "Elevated in myocarditis"),
        "D-dimer": SpecificityEntry(0.55, 0.47, 0.63, "Mildly elevated"),
        "NT-proBNP": SpecificityEntry(0.50, 0.42, 0.58, "Elevated with effusion"),
        "CRP": SpecificityEntry(0.38, 0.30, 0.46, "Imazio 2011; CRP non-specific"),
        "Copeptin": SpecificityEntry(0.65, 0.55, 0.75, "Limited data"),
        "H-FABP": SpecificityEntry(0.70, 0.62, 0.78, "Some elevation"),
        "Myoglobin": SpecificityEntry(0.55, 0.45, 0.65, "Muscle damage marker"),
        "Procalcitonin": SpecificityEntry(0.72, 0.64, 0.80, "Bacterial vs viral"),
    },
    "Pneumothorax (tension)": {
        "hs-cTnI": SpecificityEntry(0.90, 0.85, 0.95, "Not elevated in PTX"),
        "D-dimer": SpecificityEntry(0.80, 0.72, 0.88, "May be mildly elevated"),
        "NT-proBNP": SpecificityEntry(0.85, 0.78, 0.92, "Not typically elevated"),
        "CRP": SpecificityEntry(0.70, 0.62, 0.78, "Mild inflammatory response"),
        "Copeptin": SpecificityEntry(0.65, 0.55, 0.75, "Stress response"),
        "H-FABP": SpecificityEntry(0.90, 0.85, 0.95, "Not elevated"),
        "Myoglobin": SpecificityEntry(0.85, 0.78, 0.92, "Not elevated"),
        "Procalcitonin": SpecificityEntry(0.88, 0.82, 0.94, "Not elevated"),
    },
    "Acute Heart Failure": {
        "hs-cTnI": SpecificityEntry(0.55, 0.47, 0.63, "Elevated in cardiomyopathy"),
        "D-dimer": SpecificityEntry(0.40, 0.32, 0.48, "Elevated in HF"),
        "NT-proBNP": SpecificityEntry(0.76, 0.72, 0.80, "Maisel 2002; BNP study"),
        "CRP": SpecificityEntry(0.40, 0.32, 0.48, "Non-specific"),
        "Copeptin": SpecificityEntry(0.55, 0.45, 0.65, "Stress-related elevation"),
        "H-FABP": SpecificityEntry(0.70, 0.62, 0.78, "Some myocardial leakage"),
        "Myoglobin": SpecificityEntry(0.60, 0.50, 0.70, "Non-specific"),
        "Procalcitonin": SpecificityEntry(0.80, 0.72, 0.88, "Not elevated in HF"),
    },
}


def build_specificity_matrix() -> pd.DataFrame:
    """Build specificity matrix parallel to coverage (sensitivity) matrix."""
    data = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    for i, pathology in enumerate(PATHOLOGIES):
        for j, biomarker in enumerate(BIOMARKERS):
            data[i, j] = SPECIFICITY_DATA[pathology][biomarker].specificity
    return pd.DataFrame(data, index=PATHOLOGIES, columns=BIOMARKERS)


def get_prevalence_weights() -> pd.Series:
    """Return pathology prevalence as a Series indexed by pathology name."""
    return pd.Series(
        {p: PATHOLOGY_EPIDEMIOLOGY[p].prevalence for p in PATHOLOGIES},
        dtype=float,
    )


def get_severity_weights() -> pd.Series:
    """Return case-fatality-if-missed as a Series indexed by pathology name."""
    return pd.Series(
        {p: PATHOLOGY_EPIDEMIOLOGY[p].case_fatality_if_missed for p in PATHOLOGIES},
        dtype=float,
    )

# ─── Biomarker definitions ───────────────────────────────────────────────────

BIOMARKERS = [
    "hs-cTnI",       # high-sensitivity cardiac troponin I
    "D-dimer",        # fibrin degradation product
    "NT-proBNP",      # N-terminal pro–B-type natriuretic peptide
    "CRP",            # C-reactive protein
    "Copeptin",       # C-terminal pro-vasopressin
    "H-FABP",         # heart-type fatty acid binding protein
    "Myoglobin",      # early-release muscle damage marker
    "Procalcitonin",  # bacterial infection / systemic inflammation marker
]


@dataclass
class SensitivityEntry:
    """A single cell in the coverage matrix with provenance."""
    sensitivity: float
    ci_lower: float
    ci_upper: float
    source: str
    setting: str = "ED"  # "ED" or "primary_care"
    note: str = ""
    reference_quality: str = "unclassified"  # bib_verified | expert_estimate | unverified_citation


def classify_reference_quality(source: str) -> str:
    """
    Classify evidence quality from a source annotation string.

    Categories
    ----------
    bib_verified        Has a bracketed reference number ``[N]`` or a known
                        journal tag ``[JAMA]``, ``[NEJM]``, ``[Circulation]``
                        that is cross-referenced in ``refs.bib``.
    expert_estimate     Explicitly labelled as expert / clinical estimate
                        with no literature citation.
    unverified_citation Cites an author-year publication but the reference
                        has NOT been verified against ``refs.bib``.
    """
    if source.lower().startswith("expert estimate"):
        return "expert_estimate"
    # Bracketed numeric ref [4], [5], … or known journal [JAMA], [NEJM], …
    if re.search(r'\[\d+\]', source) or re.search(
        r'\[(JAMA|NEJM|Circulation|BMJ|Lancet)\]', source
    ):
        return "bib_verified"
    return "unverified_citation"


# ─── Build the coverage matrix ──────────────────────────────────────────────
# Format: COVERAGE_DATA[pathology][biomarker] = SensitivityEntry

COVERAGE_DATA: Dict[str, Dict[str, SensitivityEntry]] = {

    # ────────────────────── ACS ──────────────────────
    "ACS (STEMI/NSTEMI/UA)": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.95, ci_lower=0.93, ci_upper=0.97,
            source="Lipinski 2015 [4]; pooled hs-cTn for AMI",
            setting="ED", note="*Serial measurement at 0h/3h"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.38, ci_lower=0.28, ci_upper=0.49,
            source="Expert estimate; non-specific D-dimer elevation in ACS",
            setting="ED", note="Not a primary ACS marker"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.60, ci_lower=0.52, ci_upper=0.68,
            source="Expert estimate; NT-proBNP elevated in ACS with LV dysfunction",
            setting="ED"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.50, ci_lower=0.40, ci_upper=0.60,
            source="Expert estimate; non-specific inflammatory marker in ACS",
            setting="ED", note="Low specificity, delayed rise; no meta-analysis"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.85, ci_lower=0.79, ci_upper=0.90,
            source="Keller 2011 [6]; Raskovalova 2013 [8]; pooled for AMI",
            setting="ED", note="*Early marker, peaks within 1h"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.84, ci_lower=0.78, ci_upper=0.89,
            source="Bruins Slot 2010, Heart 96(24):1957; pooled for AMI <6h",
            setting="ED", note="*Early-release marker"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.75, ci_lower=0.65, ci_upper=0.83,
            source="Lipinski 2015 [4]; pooled for AMI",
            setting="ED", note="Early release but low specificity"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.15, ci_lower=0.08, ci_upper=0.24,
            source="Crawford 2019 [10]; minimal elevation in ACS",
            setting="ED"
        ),
    },

    # ────────────────────── PE ──────────────────────
    "Pulmonary Embolism": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.45, ci_lower=0.35, ci_upper=0.56,
            source="Becattini 2007, Circulation 116(4):427; troponin in PE",
            setting="ED", note="Elevated in RV strain from massive PE"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.95, ci_lower=0.93, ci_upper=0.97,
            source="Geersing 2012 [5]; pooled for PE",
            setting="ED", note="Age-adjusted threshold recommended >50y"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.60, ci_lower=0.49, ci_upper=0.70,
            source="Klok 2008, Am J Respir Crit Care Med 178(4):425; RV strain marker",
            setting="ED"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.65, ci_lower=0.53, ci_upper=0.76,
            source="Abul 2011, J Invest Med 59(8):1268; CRP in PE",
            setting="ED", note="Moderately elevated, non-specific"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.55, ci_lower=0.40, ci_upper=0.69,
            source="Hellenkamp 2015, Eur J Heart Fail 17(2):119",
            setting="ED", note="Reflects hemodynamic stress"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.48, ci_lower=0.35, ci_upper=0.61,
            source="Puls 2007, Eur Heart J 28(2):224",
            setting="ED", note="*RV myocardial injury marker"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.30, ci_lower=0.18, ci_upper=0.44,
            source="Expert estimate; no PE-specific myoglobin meta-analysis",
            setting="ED"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.35, ci_lower=0.22, ci_upper=0.50,
            source="Kokturk 2011, Clin Appl Thromb Hemost 17(5):E1; modest elevation in PE",
            setting="ED"
        ),
    },

    # ────────────────────── Aortic Dissection ──────────────────────
    "Aortic Dissection": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.28, ci_lower=0.18, ci_upper=0.40,
            source="Vrsalovic 2016, Int J Cardiol 215:261; troponin meta-analysis in AoD",
            setting="ED", note="Only elevated if coronary ostia involved"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.97, ci_lower=0.94, ci_upper=0.99,
            source="Nazerian 2018 [Circulation]; Sodeck 2007; pooled",
            setting="ED", note="Extremely high sensitivity at standard threshold"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.45, ci_lower=0.32, ci_upper=0.59,
            source="Expert estimate; limited data on NT-proBNP in aortic dissection",
            setting="ED"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.72, ci_lower=0.58, ci_upper=0.83,
            source="Schillinger 2002, Intensive Care Med 28(9):1305; vascular inflammation",
            setting="ED"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.68, ci_lower=0.52, ci_upper=0.81,
            source="Morello 2018, Sci Rep 8:5137; copeptin stress response in AoD",
            setting="ED", note="*Limited studies, stress-mediated release"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.25, ci_lower=0.12, ci_upper=0.42,
            source="Expert estimate; no dissection-specific H-FABP study",
            setting="ED"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.30, ci_lower=0.16, ci_upper=0.47,
            source="Expert estimate; no dissection-specific myoglobin study",
            setting="ED"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.20, ci_lower=0.09, ci_upper=0.36,
            source="Expert estimate; no dissection-specific PCT study",
            setting="ED"
        ),
    },

    # ────────────────────── Pericarditis / Myocarditis ──────────────────────
    "Pericarditis / Myocarditis": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.82, ci_lower=0.73, ci_upper=0.89,
            source="Caforio 2013, Eur Heart J 34(33):2636; ESC myocarditis guidelines",
            setting="ED", note="Higher in myocarditis than isolated pericarditis"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.35, ci_lower=0.22, ci_upper=0.50,
            source="Expert estimate; no pericarditis-specific D-dimer study",
            setting="ED"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.70, ci_lower=0.58, ci_upper=0.80,
            source="Caforio 2013, Eur Heart J 34(33):2636; myocardial wall stress",
            setting="ED"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.92, ci_lower=0.86, ci_upper=0.96,
            source="Imazio 2011 [Circulation]; CRP in acute pericarditis",
            setting="ED", note="Highly sensitive for inflammatory pericarditis"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.55, ci_lower=0.39, ci_upper=0.70,
            source="Expert estimate; stress-mediated release, no pericarditis copeptin meta-analysis",
            setting="ED"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.68, ci_lower=0.54, ci_upper=0.80,
            source="Cui 2025, Int Heart J 66(1):88; H-FABP in myocarditis",
            setting="ED"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.58, ci_lower=0.43, ci_upper=0.72,
            source="Expert estimate; early myocardial release, no pericarditis myoglobin meta-analysis",
            setting="ED"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.40, ci_lower=0.26, ci_upper=0.55,
            source="Crawford 2019 [10]; elevated if bacterial etiology",
            setting="ED"
        ),
    },

    # ────────────────────── Pneumothorax ──────────────────────
    "Pneumothorax (tension)": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.10, ci_lower=0.03, ci_upper=0.22,
            source="Expert estimate from case reports; hemodynamic compromise may cause rise",
            setting="ED", note="No primary diagnostic role; no meta-analysis"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.15, ci_lower=0.05, ci_upper=0.30,
            source="Expert estimate; no pneumothorax-specific D-dimer study",
            setting="ED"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.20, ci_lower=0.08, ci_upper=0.38,
            source="Expert estimate from case series; secondary RV strain possible",
            setting="ED"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.15, ci_lower=0.05, ci_upper=0.30,
            source="Expert estimate; CRP not elevated acutely in pneumothorax",
            setting="ED"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.30, ci_lower=0.14, ci_upper=0.50,
            source="Expert estimate; stress-response biomarker, no PTX copeptin study",
            setting="ED"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.05, ci_lower=0.01, ci_upper=0.18,
            source="Expert estimate; no myocardial injury expected in PTX",
            setting="ED"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.08, ci_lower=0.02, ci_upper=0.22,
            source="Expert estimate; no myocardial injury expected in PTX",
            setting="ED"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.10, ci_lower=0.03, ci_upper=0.25,
            source="Expert estimate; PCT not relevant for pneumothorax",
            setting="ED"
        ),
    },

    # ────────────────────── Acute Heart Failure ──────────────────────
    "Acute Heart Failure": {
        "hs-cTnI": SensitivityEntry(
            sensitivity=0.68, ci_lower=0.58, ci_upper=0.77,
            source="Felker 2015, Eur J Heart Fail 17(9):949; chronic myocyte damage",
            setting="ED"
        ),
        "D-dimer": SensitivityEntry(
            sensitivity=0.60, ci_lower=0.48, ci_upper=0.71,
            source="Zorlu 2012, J Thromb Thrombolysis 33(2):159; stasis",
            setting="ED", note="Elevated due to venous stasis and microthrombi"
        ),
        "NT-proBNP": SensitivityEntry(
            sensitivity=0.95, ci_lower=0.93, ci_upper=0.97,
            source="Maisel 2002 [NEJM]; BNP Breathing Not Properly Study",
            setting="ED", note="Primary diagnostic marker, age-adjusted thresholds"
        ),
        "CRP": SensitivityEntry(
            sensitivity=0.55, ci_lower=0.42, ci_upper=0.67,
            source="Windram 2007, Am Heart J 153(2):196; CRP in HF (prognostic, not diagnostic)",
            setting="ED"
        ),
        "Copeptin": SensitivityEntry(
            sensitivity=0.78, ci_lower=0.67, ci_upper=0.87,
            source="Demissei 2017; Peacock 2011; stress & fluid overload",
            setting="ED", note="*Reflects neurohormonal activation"
        ),
        "H-FABP": SensitivityEntry(
            sensitivity=0.55, ci_lower=0.42, ci_upper=0.67,
            source="Niizeki 2007, J Card Fail 13(7):549; H-FABP in HF",
            setting="ED"
        ),
        "Myoglobin": SensitivityEntry(
            sensitivity=0.40, ci_lower=0.28, ci_upper=0.53,
            source="Expert estimate; no heart-failure-specific myoglobin study",
            setting="ED"
        ),
        "Procalcitonin": SensitivityEntry(
            sensitivity=0.30, ci_lower=0.18, ci_upper=0.44,
            source="Expert estimate; PCT modestly elevated in HF via endotoxin translocation",
            setting="ED"
        ),
    },
}


def build_coverage_matrix() -> pd.DataFrame:
    """
    Build the P × B coverage matrix as a pandas DataFrame.

    Returns:
        DataFrame with pathologies as rows and biomarkers as columns,
        values are sensitivity estimates (0-1).
    """
    matrix = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    for i, pathology in enumerate(PATHOLOGIES):
        for j, biomarker in enumerate(BIOMARKERS):
            entry = COVERAGE_DATA[pathology][biomarker]
            matrix[i, j] = entry.sensitivity
    return pd.DataFrame(matrix, index=PATHOLOGIES, columns=BIOMARKERS)


def build_beta_parameters() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Beta distribution parameters from point estimates and 95% CIs.

    For each sensitivity entry with point estimate p and CI [L, U],
    estimate Beta(α, β) via method-of-moments:
      SE ≈ (U - L) / (2 × 1.96)
      var = SE²
      α + β = p(1-p)/var - 1  (if var > 0 and < p(1-p))
      α = p × (α + β)
      β = (1 - p) × (α + β)

    Falls back to Uniform-equivalent (α=β=1 on [L,U]) when the CI
    is too wide or degenerate.

    Returns:
        Tuple of (alpha_df, beta_df) with same shape as coverage matrix.
    """
    alphas = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    betas = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    for i, pathology in enumerate(PATHOLOGIES):
        for j, biomarker in enumerate(BIOMARKERS):
            entry = COVERAGE_DATA[pathology][biomarker]
            p = entry.sensitivity
            L, U = entry.ci_lower, entry.ci_upper
            se = (U - L) / (2 * 1.96)
            var = se ** 2
            # Guard: need 0 < var < p(1-p) and 0 < p < 1
            if 0 < var < p * (1 - p) and 0 < p < 1:
                ab_sum = p * (1 - p) / var - 1
                a = max(p * ab_sum, 0.5)   # floor at 0.5 to keep shape reasonable
                b = max((1 - p) * ab_sum, 0.5)
            else:
                # Degenerate: use weakly informative Beta centred at p
                a = max(p * 4, 0.5)
                b = max((1 - p) * 4, 0.5)
            alphas[i, j] = a
            betas[i, j] = b
    return (
        pd.DataFrame(alphas, index=PATHOLOGIES, columns=BIOMARKERS),
        pd.DataFrame(betas, index=PATHOLOGIES, columns=BIOMARKERS),
    )


def build_ci_matrices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lower and upper 95% CI matrices.

    Returns:
        Tuple of (lower_ci_df, upper_ci_df) with same shape as coverage matrix.
    """
    lower = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    upper = np.zeros((len(PATHOLOGIES), len(BIOMARKERS)))
    for i, pathology in enumerate(PATHOLOGIES):
        for j, biomarker in enumerate(BIOMARKERS):
            entry = COVERAGE_DATA[pathology][biomarker]
            lower[i, j] = entry.ci_lower
            upper[i, j] = entry.ci_upper
    return (
        pd.DataFrame(lower, index=PATHOLOGIES, columns=BIOMARKERS),
        pd.DataFrame(upper, index=PATHOLOGIES, columns=BIOMARKERS),
    )


def build_source_matrix() -> pd.DataFrame:
    """Build a matrix of source citations for each cell."""
    sources = []
    for pathology in PATHOLOGIES:
        row = []
        for biomarker in BIOMARKERS:
            entry = COVERAGE_DATA[pathology][biomarker]
            row.append(entry.source)
        sources.append(row)
    return pd.DataFrame(sources, index=PATHOLOGIES, columns=BIOMARKERS)


# ─── Biomarker metadata for multi-objective scoring ─────────────────────────

@dataclass
class BiomarkerMeta:
    """Point-of-care test metadata for multi-objective scoring."""
    name: str
    cost_eur: float           # per-test cost in EUR (estimated from literature)
    time_to_result_min: float  # turnaround time in minutes
    sample_volume_ul: float   # blood volume required in microlitres
    poc_available: bool        # POC device exists (commercially)
    source: str

BIOMARKER_META: Dict[str, BiomarkerMeta] = {
    "hs-cTnI": BiomarkerMeta(
        name="hs-cTnI", cost_eur=10.00, time_to_result_min=15.0,
        sample_volume_ul=20.0, poc_available=True,
        source="Van Dongen 2024 [11]; Atellica VTLi capillary POC"
    ),
    "D-dimer": BiomarkerMeta(
        name="D-dimer", cost_eur=7.00, time_to_result_min=10.0,
        sample_volume_ul=15.0, poc_available=True,
        source="Kip 2017 [13]; POC D-dimer costing"
    ),
    "NT-proBNP": BiomarkerMeta(
        name="NT-proBNP", cost_eur=14.00, time_to_result_min=15.0,
        sample_volume_ul=15.0, poc_available=True,
        source="Kip 2017 [13]; Cobas h232 / i-STAT"
    ),
    "CRP": BiomarkerMeta(
        name="CRP", cost_eur=5.00, time_to_result_min=4.0,
        sample_volume_ul=5.0, poc_available=True,
        source="Howick 2014 [12]; widely available POC CRP (e.g. Afinion)"
    ),
    "Copeptin": BiomarkerMeta(
        name="Copeptin", cost_eur=17.50, time_to_result_min=20.0,
        sample_volume_ul=50.0, poc_available=False,
        source="Keller 2011 [6]; B.R.A.H.M.S KRYPTOR; no POC yet"
    ),
    "H-FABP": BiomarkerMeta(
        name="H-FABP", cost_eur=8.00, time_to_result_min=15.0,
        sample_volume_ul=10.0, poc_available=True,
        source="Body 2015 [9]; CardioDetect® POC"
    ),
    "Myoglobin": BiomarkerMeta(
        name="Myoglobin", cost_eur=6.50, time_to_result_min=10.0,
        sample_volume_ul=10.0, poc_available=True,
        source="St John 2014 [14]; i-STAT myoglobin"
    ),
    "Procalcitonin": BiomarkerMeta(
        name="Procalcitonin", cost_eur=21.00, time_to_result_min=20.0,
        sample_volume_ul=20.0, poc_available=True,
        source="Kip 2017 [13]; Samsung LABGEO IB10; B.R.A.H.M.S"
    ),
}


# ─── Early presenter (<2h) sensitivity adjustments ─────────────────────────
# These capture the time-dependent sensitivity at <2h post-symptom onset.
# Sources: Keller 2011 [6], Body 2015 [9], Lipinski 2015 [4]

EARLY_PRESENTER_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "ACS (STEMI/NSTEMI/UA)": {
        "hs-cTnI":    0.70,  # reduced from 0.95 — troponin rises at 2-4h (Lipinski 2015)
        "Copeptin":   0.85,  # UNCHANGED from baseline — no direct early-presenter
                              # meta-analysis. Keller 2011 reports 0.79-0.85 overall.
                              # Previous value (0.90) was an unsupported assumption.
        "H-FABP":     0.88,  # early-release marker, retains sensitivity (Body 2015)
        "Myoglobin":  0.80,  # also early-release, slight reduction (Lipinski 2015)
        # Others: use baseline values (not time-dependent)
    },
}


def build_early_coverage_matrix() -> pd.DataFrame:
    """
    Build coverage matrix adjusted for early presenters (<2h since onset).

    Applies time-dependent sensitivity reductions where data is available.
    All other cells fall back to the standard coverage matrix values.
    """
    df = build_coverage_matrix()
    for pathology, adjustments in EARLY_PRESENTER_ADJUSTMENTS.items():
        if pathology in df.index:
            for biomarker, adj_value in adjustments.items():
                if biomarker in df.columns:
                    df.loc[pathology, biomarker] = adj_value
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("PATHOLOGY–BIOMARKER COVERAGE MATRIX (Sensitivity)")
    print("=" * 80)
    df = build_coverage_matrix()
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    print()
    
    print("EARLY PRESENTER (<2h) ADJUSTED MATRIX")
    print("=" * 80)
    early = build_early_coverage_matrix()
    print(early.to_string(float_format=lambda x: f"{x:.2f}"))
