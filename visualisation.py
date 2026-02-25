"""
Visualization Module — Publication-Quality Figures
====================================================
Generates all figures for the manuscript:
  Fig 1: Coverage heatmap
  Fig 2: Pareto frontier (coverage vs cost) 
  Fig 3: Marginal value curve
  Fig 4: Ablation impact
  Fig 5: Bootstrap stability
  Fig 6: Threshold sensitivity
  Fig 7: Biomarker kinetics (serial testing)
  Fig 8: Serial protocol comparison
  Fig 9: Sensitivity–FP trade-off (quantitative LR)
  Fig 10: Cost-effectiveness plane & CEAC
  Fig 11: Copula correlation & FP decomposition
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from biomarker_coverage_matrix import (
    BIOMARKERS, PATHOLOGIES, PATHOLOGY_SHORT, 
    build_coverage_matrix, build_early_coverage_matrix,
)
from diagnostic_panel_solver import (
    DiagnosticPanelSolver, marginal_value_analysis, current_approach_comparison,
)
from pareto_ablation_analysis import (
    compute_pareto_frontier, get_reference_approaches,
    ablation_analysis, threshold_sensitivity,
    bootstrap_panel_stability,
)
from serial_testing_model import (
    BIOMARKER_KINETICS, compare_serial_protocols,
    build_time_coverage_matrix,
)

# ── Style ──
sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {
    'primary': '#2563EB',
    'secondary': '#DC2626',
    'tertiary': '#059669',
    'quaternary': '#D97706',
    'pareto': '#7C3AED',
    'reference': '#6B7280',
    'optimal': '#2563EB',
}

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def fig1_coverage_heatmap():
    """Figure 1: Pathology × Biomarker sensitivity heatmap with annotated values."""
    C = build_coverage_matrix()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap: red (low) → yellow (mid) → green (high)
    cmap = LinearSegmentedColormap.from_list(
        'diagnostic', ['#DC2626', '#FBBF24', '#059669'], N=256
    )

    sns.heatmap(
        C, annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor='white',
        xticklabels=BIOMARKERS,
        yticklabels=[PATHOLOGY_SHORT.get(p, p) for p in PATHOLOGIES],
        ax=ax,
        cbar_kws={'label': 'Sensitivity', 'shrink': 0.8},
    )

    # Highlight cells ≥ 0.90 with a bold box
    for i in range(len(PATHOLOGIES)):
        for j in range(len(BIOMARKERS)):
            if C.iloc[i, j] >= 0.90:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, 
                    edgecolor='black', linewidth=2.5
                ))

    ax.set_title('Pathology–Biomarker Coverage Matrix (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Point-of-Care Biomarker', fontsize=12)
    ax.set_ylabel('Acute Thoracic Pathology', fontsize=12)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig1_coverage_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_pareto_frontier():
    """Figure 2: Coverage–Cost Pareto frontier with reference approaches overlaid."""
    solver = DiagnosticPanelSolver()
    pareto_df = compute_pareto_frontier(solver, tau=0.90)
    refs = get_reference_approaches(solver, tau=0.90)

    fig, ax = plt.subplots(figsize=(10, 7))

    # All panels (grey)
    non_pareto = pareto_df[~pareto_df['pareto_optimal']]
    ax.scatter(
        non_pareto['cost_eur'], non_pareto['coverage'] * 100,
        c='#E5E7EB', s=20, alpha=0.5, zorder=1, label='_nolegend_'
    )

    # Pareto-optimal panels
    pareto_opt = pareto_df[pareto_df['pareto_optimal']].sort_values('cost_eur')
    ax.scatter(
        pareto_opt['cost_eur'], pareto_opt['coverage'] * 100,
        c=COLORS['pareto'], s=60, zorder=3, label='Pareto-optimal panels',
        edgecolors='white', linewidth=0.5
    )
    # Connect pareto front
    ax.plot(
        pareto_opt['cost_eur'], pareto_opt['coverage'] * 100,
        c=COLORS['pareto'], linewidth=1.5, alpha=0.6, zorder=2
    )

    # Annotate top pareto panels
    for _, row in pareto_opt.iterrows():
        if row['coverage'] >= 0.8 or row['n_tests'] <= 2:
            label = row['biomarkers']
            if len(label) > 35:
                label = f"{row['n_tests']}-test panel"
            ax.annotate(
                label, (row['cost_eur'], row['coverage'] * 100),
                fontsize=7, ha='left', va='bottom',
                xytext=(5, 5), textcoords='offset points',
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5),
            )

    # Reference approaches (red markers)
    markers = ['s', 'D', '^']
    for i, (_, ref) in enumerate(refs.iterrows()):
        ax.scatter(
            ref['cost_eur'], ref['coverage'] * 100,
            c=COLORS['secondary'], s=120, marker=markers[i % len(markers)],
            zorder=4, edgecolors='black', linewidth=1,
            label=ref['approach']
        )

    # τ = 0.90 coverage threshold line
    ax.axhline(y=100, color='grey', linestyle='--', alpha=0.3, label='Full coverage')

    ax.set_xlabel('Panel Cost (€)', fontsize=12)
    ax.set_ylabel('Pathology Coverage (%)', fontsize=12)
    ax.set_title('Coverage–Cost Pareto Frontier (τ = 0.90)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-5, 110)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig2_pareto_frontier.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig3_marginal_value():
    """Figure 3: Marginal diagnostic value vs panel size."""
    solver = DiagnosticPanelSolver()
    marginal = marginal_value_analysis(solver, tau=0.90)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: cumulative coverage
    ax1.bar(
        marginal['panel_size'], marginal['coverage'] * 100,
        color=COLORS['primary'], alpha=0.8, edgecolor='white'
    )
    ax1.axhline(y=100, color=COLORS['secondary'], linestyle='--', alpha=0.5, label='Full coverage')
    ax1.set_xlabel('Panel Size (number of biomarkers)', fontsize=11)
    ax1.set_ylabel('Best Achievable Coverage (%)', fontsize=11)
    ax1.set_title('(A) Cumulative Coverage', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, len(BIOMARKERS) + 1))
    ax1.set_ylim(0, 115)
    ax1.legend()

    # Annotate panel composition
    for _, row in marginal.iterrows():
        if row['panel_size'] <= 4:
            ax1.text(
                row['panel_size'], row['coverage'] * 100 + 3,
                row['best_panel'].replace(', ', '\n'), 
                fontsize=7, ha='center', va='bottom'
            )

    # Right: marginal gain
    colors = [COLORS['tertiary'] if g > 0.05 else '#9CA3AF' 
              for g in marginal['marginal_coverage_gain']]
    ax2.bar(
        marginal['panel_size'], marginal['marginal_coverage_gain'] * 100,
        color=colors, alpha=0.8, edgecolor='white'
    )
    ax2.set_xlabel('Panel Size (number of biomarkers)', fontsize=11)
    ax2.set_ylabel('Marginal Coverage Gain (%)', fontsize=11)
    ax2.set_title('(B) Marginal Diagnostic Value', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, len(BIOMARKERS) + 1))

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig3_marginal_value.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig4_ablation():
    """Figure 4: Ablation impact — coverage change when each biomarker is removed."""
    ablation = ablation_analysis(tau=0.90)

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = []
    for _, row in ablation.iterrows():
        if row['in_optimal']:
            colors.append(COLORS['secondary'] if row['coverage_change'] < 0 else COLORS['quaternary'])
        else:
            colors.append('#9CA3AF')

    bars = ax.barh(
        ablation['removed'], ablation['coverage_change'] * 100,
        color=colors, edgecolor='white', height=0.6
    )

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Coverage Change (percentage points)', fontsize=11)
    ax.set_ylabel('Biomarker Removed', fontsize=11)
    ax.set_title('Ablation Analysis: Coverage Impact of Removing Each Biomarker (τ = 0.90)',
                 fontsize=13, fontweight='bold')

    # Add new gap annotations — placed inside bars with enough padding
    for i, (_, row) in enumerate(ablation.iterrows()):
        if row['new_gaps'] != 'None':
            # Shorten long gap descriptions
            gap_text = row['new_gaps']
            if 'Aortic Dissection' in gap_text and 'Pulmonary' in gap_text:
                gap_text = 'PE + AoD'
            elif 'Pericarditis' in gap_text:
                gap_text = 'Peri/Myo'
            elif 'STEMI' in gap_text:
                gap_text = 'ACS'
            elif 'Acute Heart' in gap_text:
                gap_text = 'AHF'
            bar_val = row['coverage_change'] * 100
            # Place text just to the right of the bar end (inside white space)
            ax.text(
                bar_val + 0.5, i,
                f"Gap: {gap_text}", fontsize=8, va='center', ha='left',
                color=COLORS['secondary']
            )

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['secondary'], label='In optimal panel (coverage loss)'),
        mpatches.Patch(color='#9CA3AF', label='Not in optimal panel'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig5_bootstrap_stability():
    """Figure 5: Bootstrap biomarker inclusion rates and panel stability."""
    bootstrap = bootstrap_panel_stability(n_bootstrap=1000, tau=0.90)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: biomarker inclusion rates
    rates = bootstrap['biomarker_inclusion_rate']
    sorted_rates = sorted(rates.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_rates]
    values = [x[1] * 100 for x in sorted_rates]

    colors = [COLORS['primary'] if v > 50 else '#9CA3AF' for v in values]
    ax1.barh(names, values, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlabel('Inclusion Rate (%)', fontsize=11)
    ax1.set_ylabel('Biomarker', fontsize=11)
    ax1.set_title('(A) Bootstrap Inclusion Rate (n=1000)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 105)

    for i, v in enumerate(values):
        ax1.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9)

    # Right: top panel frequencies
    top_panels = bootstrap['panel_frequency'][:10]
    panel_labels = [', '.join(p['panel']) for p in top_panels]
    panel_freqs = [p['frequency'] * 100 for p in top_panels]

    # Truncate long labels
    panel_labels_short = []
    for label in panel_labels:
        if len(label) > 40:
            label = label[:37] + '...'
        panel_labels_short.append(label)

    ax2.barh(panel_labels_short[::-1], panel_freqs[::-1],
             color=COLORS['pareto'], edgecolor='white', height=0.6)
    ax2.set_xlabel('Frequency (%)', fontsize=11)
    ax2.set_title('(B) Most Frequent Optimal Panels', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig5_bootstrap_stability.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig6_threshold_sensitivity():
    """Figure 6: Minimum panel size vs sensitivity threshold τ."""
    thresh = threshold_sensitivity()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: panel size vs threshold
    ax1.plot(
        thresh['threshold'], thresh['min_panel_size'],
        'o-', color=COLORS['primary'], linewidth=2, markersize=8
    )
    ax1.set_xlabel('Sensitivity Threshold (τ)', fontsize=11)
    ax1.set_ylabel('Minimum Panel Size', fontsize=11)
    ax1.set_title('(A) Panel Size vs Threshold', fontsize=12, fontweight='bold')
    ax1.set_xticks(thresh['threshold'])

    # Annotate panel compositions — only at key thresholds to avoid overlap
    annotated_thresholds = {0.80, 0.90, 0.95}  # subset to prevent text collision
    for _, row in thresh.iterrows():
        if not pd.isna(row['min_panel_size']) and row['threshold'] in annotated_thresholds:
            ax1.annotate(
                row['optimal_panel'].replace(', ', '\n'),
                (row['threshold'], row['min_panel_size']),
                fontsize=6, ha='center', va='bottom',
                xytext=(0, 10), textcoords='offset points',
            )

    # Right: cost vs threshold
    ax2.plot(
        thresh['threshold'], thresh['cost_eur'],
        's-', color=COLORS['tertiary'], linewidth=2, markersize=8
    )
    ax2.set_xlabel('Sensitivity Threshold (τ)', fontsize=11)
    ax2.set_ylabel('Minimum Panel Cost (€)', fontsize=11)
    ax2.set_title('(B) Panel Cost vs Threshold', fontsize=12, fontweight='bold')
    ax2.set_xticks(thresh['threshold'])

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig6_threshold_sensitivity.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Generate all figures ───────────────────────────────────────────────────

def fig7_biomarker_kinetics():
    """Figure 7: Time-dependent sensitivity multipliers for panel biomarkers.

    Shows how each biomarker's sensitivity changes over 0-12h since symptom
    onset.  This motivates the serial testing protocol (SISTER ACT core).
    """
    panel = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    kinetics_colors = {
        'hs-cTnI': COLORS['secondary'],
        'D-dimer': COLORS['tertiary'],
        'NT-proBNP': COLORS['primary'],
        'CRP': COLORS['quaternary'],
    }

    time_fine = np.linspace(0, 12, 200)
    for bm in panel:
        if bm in BIOMARKER_KINETICS:
            k = BIOMARKER_KINETICS[bm]
            mults = [k.multiplier_at(t) for t in time_fine]
            ax.plot(time_fine, mults, linewidth=2.2,
                    color=kinetics_colors.get(bm, '#888'),
                    label=bm, zorder=3)

    # Mark serial protocol time-points
    for t, label in [(0, '0h'), (1, '1h'), (3, '3h')]:
        ax.axvline(t, color='#9CA3AF', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(t + 0.1, 1.04, label, fontsize=9, color='#6B7280',
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Hours since symptom onset", fontsize=11)
    ax.set_ylabel("Sensitivity multiplier (relative to peak)", fontsize=11)
    ax.set_title("Biomarker Kinetics — Serial Testing Rationale", fontsize=13, fontweight='bold')
    ax.set_xlim(0, 12)
    ax.set_ylim(0.4, 1.08)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.axhline(0.90, color='#DC2626', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.text(11.5, 0.905, 'τ=0.90', fontsize=8, color='#DC2626', ha='right')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURE_DIR, f"fig7_kinetics.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved fig7_kinetics.pdf/png")


def fig8_serial_protocol_comparison():
    """Figure 8: Serial testing protocol comparison — cumulative coverage.

    Bar chart showing coverage at single (0h), 0/1h, 0/3h, and 0/1/3h
    protocols for the optimal 4-test panel.
    """
    panel = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
    protocols = compare_serial_protocols(panel, tau=0.90)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [3, 2]})

    # Panel A: Coverage comparison
    ax = axes[0]
    names = list(protocols['protocols'].keys())
    coverages = [protocols['protocols'][n]['cumulative_coverage'] for n in names]
    times = [protocols['protocols'][n]['total_time_to_decision_minutes'] for n in names]

    bar_colors = [COLORS['reference'], COLORS['primary'], COLORS['tertiary'],
                  COLORS['pareto'], COLORS['quaternary']]

    bars = ax.bar(range(len(names)), coverages, color=bar_colors[:len(names)],
                  edgecolor='white', linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Cumulative Panel Coverage", fontsize=11)
    ax.set_title("A. Protocol Coverage Comparison", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.axhline(0.833, color='#DC2626', linestyle=':', linewidth=1.0, alpha=0.5)
    ax.text(len(names) - 0.5, 0.84, 'single 0h baseline',
            fontsize=8, color='#DC2626', ha='right')

    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{cov:.0%}", ha='center', fontsize=9, fontweight='bold')

    # Panel B: Time-to-decision
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(names)), times, color=bar_colors[:len(names)],
                     edgecolor='white', linewidth=1.5, zorder=3)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Weighted time to decision (min)", fontsize=11)
    ax2.set_title("B. Time to Decision", fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    for bar, t in zip(bars2, times):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{t:.0f} min", ha='left', va='center', fontsize=9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURE_DIR, f"fig8_serial_protocols.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved fig8_serial_protocols.pdf/png")


def fig9_sensitivity_fp_tradeoff():
    """Figure 9: Sensitivity vs FP trade-off curve for quantitative LR interpretation."""
    results_path = os.path.join(os.path.dirname(__file__), "results", "quantitative_interpretation.json")
    if not os.path.exists(results_path):
        from quantitative_panel_interpretation import run_full_analysis
        results = run_full_analysis(n_healthy=200_000, n_disease=50_000)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    else:
        import json
        with open(results_path) as f:
            results = json.load(f)

    sweep = results['sensitivity_fp_tradeoff']
    sens_vals = [s['target_sensitivity'] for s in sweep]
    fp_any_cop = [s['any_pathology_fp_copula'] for s in sweep]
    fp_ed_cop = [s['ed_only_fp_copula'] for s in sweep]
    fp_any_ind = [s['any_pathology_fp_indep'] for s in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        [s * 100 for s in sens_vals], [f * 100 for f in fp_any_ind],
        'o--', color=COLORS['reference'], linewidth=1.5, markersize=6,
        label='Any pathology (independent)', alpha=0.6,
    )
    ax.plot(
        [s * 100 for s in sens_vals], [f * 100 for f in fp_any_cop],
        's-', color=COLORS['primary'], linewidth=2, markersize=7,
        label='Any pathology (copula)',
    )
    ax.plot(
        [s * 100 for s in sens_vals], [f * 100 for f in fp_ed_cop],
        'D-', color=COLORS['secondary'], linewidth=2, markersize=7,
        label='ED-only (copula)',
    )

    # Reference line: binary OR
    binary_fp = results['binary_or_rule']['panel_fp_rate'] * 100
    ax.axhline(binary_fp, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(81, binary_fp + 1.5, f'Binary OR: {binary_fp:.1f}%',
            fontsize=9, color='gray', style='italic')

    # Working points
    ax.axvline(95, color=COLORS['tertiary'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(95.3, 25, '95% target', fontsize=9, color=COLORS['tertiary'], rotation=90)

    ax.set_xlabel('Per-pathology sensitivity target (%)', fontsize=12)
    ax.set_ylabel('Panel-level false-positive rate (%)', fontsize=12)
    ax.set_title('Sensitivity–FP Trade-off: Quantitative LR Interpretation', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(78, 100)
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURE_DIR, f"fig9_sensitivity_fp_tradeoff.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved fig9_sensitivity_fp_tradeoff.pdf/png")


def fig10_cost_effectiveness_plane():
    """Figure 10: Cost-effectiveness plane + CEAC from health-economic PSA."""
    results_path = os.path.join(os.path.dirname(__file__), "results", "health_economics.json")
    if not os.path.exists(results_path):
        print("  [SKIP] health_economics.json not found — run pipeline first")
        return

    import json
    with open(results_path) as f:
        he = json.load(f)

    psa = he.get('psa', {})
    ceac = psa.get('ceac', {})
    scatter = psa.get('scatter_data', {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: CE Plane
    if scatter:
        for strategy_key, label, color in [
            ('panel_vs_current', 'Optimal Panel vs Current', COLORS['primary']),
            ('sister_act_vs_current', 'SISTER ACT vs Current', COLORS['secondary']),
        ]:
            if strategy_key in scatter:
                d_cost = scatter[strategy_key].get('delta_costs', [])
                d_qaly = scatter[strategy_key].get('delta_qalys', [])
                if d_cost and d_qaly:
                    ax1.scatter(d_qaly, d_cost, alpha=0.15, s=8, color=color, label=label)

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_xlabel(r'$\Delta$ QALYs (per cohort)', fontsize=11)
        ax1.set_ylabel(r'$\Delta$ Cost (\texteuro{} per cohort)', fontsize=11)
        ax1.set_title('A. Cost-Effectiveness Plane', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Quadrant labels
        ax1.text(0.95, 0.95, 'More costly,\nmore effective', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=8, color='gray', alpha=0.6)
        ax1.text(0.05, 0.05, 'Less costly,\nless effective', transform=ax1.transAxes,
                 ha='left', va='bottom', fontsize=8, color='gray', alpha=0.6)
        ax1.text(0.05, 0.95, 'More costly,\nless effective', transform=ax1.transAxes,
                 ha='left', va='top', fontsize=8, color='gray', alpha=0.6)
        ax1.text(0.95, 0.05, 'DOMINANT\n(cheaper + better)', transform=ax1.transAxes,
                 ha='right', va='bottom', fontsize=9, color=COLORS['tertiary'],
                 fontweight='bold', alpha=0.8)
    else:
        ax1.text(0.5, 0.5, 'Scatter data not available', transform=ax1.transAxes,
                 ha='center', va='center')

    # Panel B: CEAC
    if ceac:
        strategy_colors = {
            'Current care (HEART + cTnI)': COLORS['reference'],
            'Optimal 4-test panel': COLORS['primary'],
            'SISTER ACT': COLORS['secondary'],
        }
        for strategy_name, wtp_dict in ceac.items():
            wtps = sorted([int(w) for w in wtp_dict.keys()])
            probs = [wtp_dict[str(w)] for w in wtps]
            color = strategy_colors.get(strategy_name, 'gray')
            ax2.plot(
                [w / 1000 for w in wtps], probs,
                '-', linewidth=2, color=color, label=strategy_name,
            )

        ax2.set_xlabel(r'Willingness-to-pay (k\texteuro{}/QALY)', fontsize=11)
        ax2.set_ylabel('Probability cost-effective', fontsize=11)
        ax2.set_title('B. Cost-Effectiveness Acceptability Curve', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='center right')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'CEAC data not available', transform=ax2.transAxes,
                 ha='center', va='center')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURE_DIR, f"fig10_cost_effectiveness.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved fig10_cost_effectiveness.pdf/png")


def fig11_copula_correlation():
    """Figure 11: Biomarker correlation heatmap (copula model) with FP decomposition."""
    from correlation_dependence_model import PAIRWISE_CORRELATIONS

    biomarkers = ["hs-cTnI", "D-dimer", "NT-proBNP", "CRP"]
    n = len(biomarkers)
    corr_matrix = np.eye(n)

    for entry in PAIRWISE_CORRELATIONS:
        b1, b2, r = entry.biomarker_a, entry.biomarker_b, entry.correlation
        if b1 in biomarkers and b2 in biomarkers:
            i, j = biomarkers.index(b1), biomarkers.index(b2)
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel A: Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    short_labels = ['hs-cTnI', 'D-dimer', 'NT-proBNP', 'CRP']
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
        vmin=-0.3, vmax=0.8, center=0, square=True,
        xticklabels=short_labels, yticklabels=short_labels,
        ax=ax1, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
    )
    ax1.set_title('A. Panel Biomarker Correlations', fontsize=12, fontweight='bold')

    # Panel B: FP rate decomposition bar chart
    fp_data = {
        'Binary (indep.)': 93.4,
        'Binary (copula)': 86.5,
        'Quant. LR (copula)': 66.5,
        'ED-only (copula)': 63.0,
        'HEAR + Quant.\n(per 1000)': 37.1,
    }
    labels = list(fp_data.keys())
    values = list(fp_data.values())
    colors_bar = [COLORS['reference'], COLORS['quaternary'], COLORS['primary'],
                  COLORS['secondary'], COLORS['tertiary']]

    bars = ax2.barh(labels, values, color=colors_bar, edgecolor='white', height=0.65)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('False-positive / unnecessary referral rate (%)', fontsize=11)
    ax2.set_title('B. Layered FP Reduction', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 110)
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURE_DIR, f"fig11_copula_fp_decomposition.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Saved fig11_copula_fp_decomposition.pdf/png")


def generate_all_figures():
    """Generate all publication figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    print("\n[1/8] Coverage heatmap...")
    fig1_coverage_heatmap()

    print("\n[2/8] Pareto frontier...")
    fig2_pareto_frontier()

    print("\n[3/8] Marginal diagnostic value...")
    fig3_marginal_value()

    print("\n[4/8] Ablation analysis...")
    fig4_ablation()

    print("\n[5/8] Bootstrap stability...")
    fig5_bootstrap_stability()

    print("\n[6/8] Threshold sensitivity...")
    fig6_threshold_sensitivity()

    print("\n[7/8] Biomarker kinetics (serial testing)...")
    fig7_biomarker_kinetics()

    print("\n[8/11] Serial protocol comparison...")
    fig8_serial_protocol_comparison()

    print("\n[9/11] Sensitivity–FP trade-off (quantitative LR)...")
    fig9_sensitivity_fp_tradeoff()

    print("\n[10/11] Cost-effectiveness plane & CEAC...")
    fig10_cost_effectiveness_plane()

    print("\n[11/11] Copula correlation & FP decomposition...")
    fig11_copula_correlation()

    print(f"\nAll figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
