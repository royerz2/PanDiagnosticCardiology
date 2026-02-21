# PanDiagnosticCardiology — Plain-Language Cheatsheet

## What is this paper about? (One sentence)

> **Given 8 blood tests and 6 deadly chest conditions, what is the _smallest, cheapest, fastest_ combination of tests that can detect _all_ of them?**

---

## The Problem

A patient walks into a GP clinic with chest pain. It could be **any** of six life-threatening diseases:

| # | Pathology | What it is |
|---|-----------|-----------|
| 1 | **ACS** (Acute Coronary Syndrome) | Heart attack |
| 2 | **PE** (Pulmonary Embolism) | Blood clot in the lungs |
| 3 | **Aortic Dissection** | Aorta is tearing apart |
| 4 | **Pericarditis / Myocarditis** | Inflammation of the heart lining/muscle |
| 5 | **Pneumothorax** | Collapsed lung |
| 6 | **Acute Heart Failure** | Heart can't pump enough |

Today, doctors test for **one disease at a time** (HEART score for heart attacks, Wells score for PE). Nobody has a principled method to cover **all six at once**.

---

## The Idea

Borrow a mathematical trick from **cancer drug design** (the ALIN paper):

- **In cancer**: find the smallest group of drugs that blocks _every_ way a tumour can survive.  
- **Here**: find the smallest group of blood tests that _detects_ every deadly chest condition.

Both are the same math problem — a **"minimal hitting set"** (or **"weighted set cover"**).

---

## The Inputs

### 8 candidate blood tests (biomarkers)

| Biomarker | What it detects | Cost | Time | Blood needed |
|-----------|----------------|------|------|-------------|
| hs-cTnI (troponin) | Heart muscle damage | £8.50 | 15 min | 20 µL |
| D-dimer | Blood clots | £6.00 | 10 min | 15 µL |
| NT-proBNP | Heart strain | £12.00 | 15 min | 10 µL |
| CRP | Inflammation | £4.00 | 5 min | 5 µL |
| Copeptin | Stress hormone | £15.00 | 10 min | 10 µL |
| H-FABP | Early heart damage | £10.00 | 10 min | 5 µL |
| Myoglobin | Muscle breakdown | £8.00 | 8 min | 5 µL |
| Procalcitonin (PCT) | Bacterial infection | £12.00 | 15 min | 10 µL |

### The coverage matrix

Each cell = "how good is this test at detecting this disease?" (sensitivity, 0–1).

|  | hs-cTnI | D-dimer | NT-proBNP | CRP | Copeptin | H-FABP | Myo | PCT |
|--|---------|---------|-----------|-----|----------|--------|-----|-----|
| **ACS** | **0.95** | 0.38 | 0.60 | 0.50 | 0.85 | 0.84 | 0.75 | 0.15 |
| **PE** | 0.45 | **0.95** | 0.60 | 0.65 | 0.55 | 0.48 | 0.30 | 0.35 |
| **Aortic Dissection** | 0.28 | **0.97** | 0.45 | 0.72 | 0.68 | 0.25 | 0.30 | 0.20 |
| **Pericarditis/Myo** | 0.82 | 0.35 | 0.70 | **0.92** | 0.55 | 0.68 | 0.58 | 0.40 |
| **Pneumothorax** | 0.10 | 0.15 | 0.20 | 0.15 | 0.30 | 0.05 | 0.08 | 0.10 |
| **Acute HF** | 0.68 | 0.60 | **0.95** | 0.55 | 0.78 | 0.55 | 0.40 | 0.30 |

**Bold** = sensitivity ≥ 0.90 (our threshold). Notice pneumothorax row: _nothing works_ → it can only be diagnosed by physical examination or imaging, not blood.

---

## The Equations (Explained)

### Equation 1 — "Does this test cover this disease?"

$$
a_{p,b} = \begin{cases} 1 & \text{if } M_{p,b} \geq \tau \\ 0 & \text{otherwise} \end{cases}
$$

**Plain English:** Look up the sensitivity of test $b$ for disease $p$ in the matrix. If it's at least $\tau$ (we use 0.90), mark it as "covered" (1). Otherwise "not covered" (0).

**Example:** D-dimer for PE has sensitivity 0.95 ≥ 0.90 → $a_{\text{PE, D-dimer}} = 1$ ✅  
**Example:** CRP for ACS has sensitivity 0.50 < 0.90 → $a_{\text{ACS, CRP}} = 0$ ❌

After applying the threshold, the big matrix collapses to a simple yes/no grid:

|  | hs-cTnI | D-dimer | NT-proBNP | CRP | rest... |
|--|---------|---------|-----------|-----|---------|
| ACS | ✅ | ❌ | ❌ | ❌ | ❌ |
| PE | ❌ | ✅ | ❌ | ❌ | ❌ |
| Aortic Diss. | ❌ | ✅ | ❌ | ❌ | ❌ |
| Peri/Myo | ❌ | ❌ | ❌ | ✅ | ❌ |
| Pneumothorax | ❌ | ❌ | ❌ | ❌ | ❌ |
| Acute HF | ❌ | ❌ | ✅ | ❌ | ❌ |

---

### Equation 2 — "What to minimise" (The Objective)

$$
\min_{\mathbf{x}} \sum_{b \in \mathcal{B}} \Big[ \underbrace{w_{\text{size}}}_{=10} + \underbrace{w_{\text{cost}} \cdot c_b}_{=1 \times \text{cost}} + \underbrace{w_{\text{time}} \cdot t_b}_{=0.5 \times \text{time}} + \underbrace{w_{\text{sample}} \cdot v_b}_{=0.1 \times \text{volume}} \Big] \cdot x_b
$$

where $x_b \in \{0, 1\}$ means "include test $b$ in the panel (1) or not (0)."

**Plain English:** For each test you include, you pay a _penalty_:

| Penalty term | What it means | Weight | Why? |
|-------------|---------------|--------|------|
| $w_{\text{size}} = 10$ | "Just for existing" — fixed cost per test | 10 | Fewer tests = simpler for the GP |
| $w_{\text{cost}} \cdot c_b$ | Reagent cost in £ | 1 | Cheaper = better |
| $w_{\text{time}} \cdot t_b$ | Turnaround minutes | 0.5 | Faster = better |
| $w_{\text{sample}} \cdot v_b$ | Blood volume in µL | 0.1 | Less blood = better |

The solver picks the subset of tests with the **lowest total penalty** that still covers all detectable diseases.

**Example — scoring D-dimer:**  
$\text{penalty}(\text{D-dimer}) = 10 + 1 \times 6 + 0.5 \times 10 + 0.1 \times 15 = 10 + 6 + 5 + 1.5 = 22.5$

**Example — scoring NT-proBNP:**  
$\text{penalty}(\text{NT-proBNP}) = 10 + 1 \times 12 + 0.5 \times 15 + 0.1 \times 10 = 10 + 12 + 7.5 + 1 = 30.5$

The solver adds up penalties for all selected tests and finds the set with the minimum total.

---

### Equation 3 — "Make sure every disease is covered" (The Constraint)

$$
\forall\, p \in \mathcal{P}^* : \quad \sum_{b \in \mathcal{B}} a_{p,b} \cdot x_b \geq 1
$$

**Plain English:** For _every_ detectable disease $p$, at least one selected test must cover it.

$\mathcal{P}^* = \{$ACS, PE, Aortic Dissection, Pericarditis, Acute HF$\}$ — the 5 "coverable" pathologies. Pneumothorax is excluded because no test reaches 0.90 for it.

**Worked example for PE:**  
$a_{\text{PE,hs-cTnI}} \cdot x_{\text{hs-cTnI}} + a_{\text{PE,D-dimer}} \cdot x_{\text{D-dimer}} + \ldots \geq 1$  
$= 0 \cdot x_{\text{hs-cTnI}} + 1 \cdot x_{\text{D-dimer}} + 0 + 0 + \ldots \geq 1$  
→ **D-dimer must be selected** (it's the _only_ test that covers PE).

---

### Putting it all together

The solver solves this as an **Integer Linear Program (ILP)** — a standard optimisation technique. Think of it as:

> "Turn on/off each of the 8 blood tests (0 or 1). Find the cheapest/simplest combination where every disease has at least one test that detects it with ≥90% sensitivity."

---

## The Answer

### Optimal Panel (τ = 0.90)

| Test | Covers | Cost | Time | Blood |
|------|--------|------|------|-------|
| **hs-cTnI** | ACS (heart attack) | £8.50 | 15 min | 20 µL |
| **D-dimer** | PE + Aortic Dissection | £6.00 | 10 min | 15 µL |
| **NT-proBNP** | Acute Heart Failure | £12.00 | 15 min | 10 µL |
| **CRP** | Pericarditis/Myocarditis | £4.00 | 5 min | 5 µL |
| **TOTAL** | **5/6 diseases (83.3%)** | **£30.50** | **15 min** | **55 µL** |

**Pneumothorax** is uncoverable — needs stethoscope + chest X-ray, not blood.

---

## Key Findings (at a glance)

### 1. D-dimer is irreplaceable
- Removing D-dimer loses **2 diseases** (PE + Aortic Dissection) = –33.3% coverage  
- It appears in **100% of bootstrap iterations**  
- No other test comes close for those two conditions

### 2. Pareto Frontier — "What if I can't afford 4 tests?"

| Tests | Panel | Coverage | Cost |
|-------|-------|----------|------|
| 1 | D-dimer alone | 33% | £6 |
| 2 | D-dimer + CRP | 50% | £10 |
| 3 | D-dimer + CRP + hs-cTnI | 67% | £18.50 |
| 4 | All four | 83% | £30.50 |

Each extra test adds _exactly one more disease_ for increasing cost.

### 3. Early presenters (< 2 hours from symptom onset)
- Troponin is **too slow** to detect very early heart attacks (sensitivity drops 0.95 → 0.70)
- Solution: **swap troponin for copeptin** (stress hormone released in minutes)
- New panel: **Copeptin + D-dimer + NT-proBNP + CRP** (same coverage, costs £37)

### 4. Threshold sensitivity
- At τ = 0.80–0.82: only 3 tests needed  
- At τ = 0.85–0.92: 4 tests needed (the standard regime)  
- At τ = 0.95: drops back to 3 tests but _loses_ pericarditis coverage

### 5. Bootstrap stability
- 1,000 random resamplings → D-dimer always selected
- CRP selected 62.4% of the time, hs-cTnI 37.6% (they alternate depending on which diseases dominate the random sample)

---

## Key Terminology

| Term | Meaning |
|------|---------|
| **Sensitivity** | Probability the test is positive when the disease is truly present (higher = better at detecting) |
| **Specificity** | Probability the test is negative when the disease is truly absent (not modelled here) |
| **τ (tau)** | The sensitivity threshold — test must detect the disease at least this well to "count" (default: 0.90 = 90%) |
| **Coverage** | Fraction of the 6 pathologies covered by at least one test above τ |
| **MHS** | Minimal Hitting Set — smallest group that "hits" every target |
| **Set Cover** | Pick the fewest sets (tests) whose union covers the entire universe (diseases) |
| **ILP** | Integer Linear Program — optimisation where variables are whole numbers (0 or 1 = include/exclude) |
| **Pareto optimal** | A solution where you can't improve one thing (e.g., cost) without worsening another (e.g., coverage) |
| **Ablation** | Removing one component and measuring the damage |
| **Bootstrap** | Randomly resample the data many times to test how stable the answer is |
| **POC** | Point-of-care — testing done at the bedside/clinic, not sent to a central lab |

---

## The Big Insight

> **Single-axis testing (one test for one disease) is how we've always done it. But chest pain has 6 possible causes. By borrowing a math trick from cancer drug design, we can find the optimal multi-disease panel with just 4 cheap, fast, fingerprick blood tests.**

This is the same as asking: "What's the minimum number of antibiotics to cover every possible bacterial strain?" — a covering problem.

---

## Visual Summary

```
PATIENT: "My chest hurts"
         │
         ▼
   ┌─────────────────────────────────────┐
   │   OPTIMAL 4-TEST PANEL (£30.50)     │
   │   One fingerprick, 55µL, 15 min     │
   ├─────────────────────────────────────┤
   │                                     │
   │  hs-cTnI ────► ACS (heart attack)   │
   │  D-dimer ────► PE (lung clot)       │
   │          └───► Aortic Dissection    │
   │  NT-proBNP ──► Acute Heart Failure  │
   │  CRP ────────► Pericarditis/Myo     │
   │                                     │
   │  ❌ Pneumothorax = needs imaging    │
   └─────────────────────────────────────┘
```

---

## What this paper does NOT do

- ❌ Does NOT use any new patient data — everything comes from published studies
- ❌ Does NOT model specificity (false alarms are not penalised)
- ❌ Does NOT model serial testing (test-now-then-test-again strategies)
- ❌ Does NOT account for correlations between tests in the same patient
- ❌ Does NOT claim the panel is clinically validated — it's a hypothesis to be tested in the SISTER ACT trial
