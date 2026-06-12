"""Regenerate the 4 headline CPPro figures into ../figures/ (committed to git).

These are the curated, paper-grade graphs embedded in the README. Everything is
read from committed result CSVs/JSON in ../results/ (nothing here is hand-authored),
so `python scripts/make_key_figures.py` always reproduces the figures from data.

Figures (all on the v6 held-out test, n=570, threshold 0.5):
  fig1_benchmark.png        best-of-each backbone vs previous SOTA pLM4CPPs
  fig2_head_sweep_600m.png  classifier-head sweep on frozen ESM-C 600M (open-weight)
  fig3_head_sweep_6b.png    classifier-head sweep on frozen ESM-C 6B
  fig4_hnm_6b.png           hard-negative-mining progress (6B + DeepSet)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGS = ROOT / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

HEAD_COLORS = {'deepset': '#d62728', 'cnn': '#ff7f0e', 'mlp': '#1f77b4',
               'transformer': '#2ca02c', 'seqcnn': '#9467bd'}


def _save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f'{stem}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote figures/{stem}.png')


def fig1_benchmark():
    """Deployed model per backbone (HNM ensembles) vs previous SOTA, threshold 0.5 on v6 test."""
    plm = json.load(open(RESULTS / 'plm4cpps_640_on_v6_test.json'))['test_at_05']['mcc']
    m6b = json.load(open(ROOT / 'checkpoints' / 'hnm_round2' / 'metrics.json'))['test_mcc_ensemble']
    p600 = ROOT / 'checkpoints' / 'frozen_600m_seqcnn_hnm' / 'metrics.json'
    m600 = json.load(open(p600))['test_mcc_ensemble'] if p600.exists() else None

    bars = [('pLM4CPPs\n(ESM2-150M + CNN)\nprevious SOTA', plm, '#9aa0a6')]
    if m600 is not None:
        bars.append(('CPPro-600M + HNM\n(open: ESM-C 600M + seqcnn)', m600, '#2ca02c'))
    bars.append(('CPPro-6B + HNM\n(ESM-C 6B + DeepSet)', m6b, '#1f77b4'))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor='black', width=0.6)
    for x, (_, m, _c) in zip(xs, bars):
        ax.text(x, m + 0.015, f'{m:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylabel('MCC on v6 held-out test (n = 570, threshold 0.5)', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('CPP classifier benchmark: deployed model per backbone\n'
                 '(5-seed ensemble + hard-negative mining)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    _save(fig, 'fig1_benchmark')


def _head_sweep(summary_csv, tag, title):
    s = pd.read_csv(summary_csv).sort_values('test_mcc_mean', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(s))
    colors = [HEAD_COLORS.get(h, '#888') for h in s['head']]
    ax.bar(xs, s.test_mcc_mean, yerr=s.test_mcc_std, color=colors, edgecolor='black', capsize=4, width=0.6)
    for x, m, sd in zip(xs, s.test_mcc_mean, s.test_mcc_std):
        ax.text(x, m + sd + 0.008, f'{m:.3f}\n±{sd:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(s['head'], fontsize=11)
    ax.set_ylabel('MCC on v6 test (3 seeds, threshold 0.5)', fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    _save(fig, tag)


def fig_hnm(rounds_csv, stem, title, max_round=None):
    """HNM progress as % change from baseline (round 0): FP rate falls steeply while recall
    barely moves. Ensemble metrics. Absolute % annotated at the endpoints."""
    d = pd.read_csv(rounds_csv).sort_values('round')
    if max_round is not None:
        d = d[d['round'] <= max_round]
    d = d.reset_index(drop=True)
    rounds = d['round'].values
    fp = d['fp_rate_screen@0.5'].values                  # FP rate on fixed 10k held-out screen
    tp = d['tpr_test@0.5'].values                        # recall on v6 test
    fp_pct = (fp - fp[0]) / fp[0] * 100
    tp_pct = (tp - tp[0]) / tp[0] * 100
    best = int(np.argmin(fp_pct))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(0, color='grey', lw=1, ls='--', alpha=0.6)
    ax.plot(rounds, fp_pct, 'o-', color='#d62728', lw=2.6, ms=10, label='False-positive rate')
    ax.plot(rounds, tp_pct, 's-', color='#1f77b4', lw=2.6, ms=10, label='True-positive rate (recall)')

    def lab(x, ypct, absval, color, weight='normal', dy=11):
        ax.annotate(f'{absval*100:.1f}%', (x, ypct), textcoords='offset points',
                    xytext=(0, dy), ha='center', fontsize=9.5, color=color, fontweight=weight)
    lab(rounds[0], fp_pct[0], fp[0], '#d62728', dy=-15)
    lab(rounds[best], fp_pct[best], fp[best], '#d62728', weight='bold', dy=15)
    lab(rounds[0], tp_pct[0], tp[0], '#1f77b4', dy=11)
    lab(rounds[best], tp_pct[best], tp[best], '#1f77b4', weight='bold', dy=11)

    ax.set_ylim(min(fp_pct.min(), -6) - 6, 8)
    ax.set_xticks(rounds)
    ax.set_xlabel('Hard-negative-mining round', fontsize=12)
    ax.set_ylabel('Change from baseline (%)', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=10, frameon=False)
    fig.text(0.5, 0.005, 'Ensemble of 5 seeds. FP rate on a fixed 10k held-out negative '
             'screen; recall on the v6 test; threshold 0.5.', ha='center', fontsize=8, color='#555')
    _save(fig, stem)


if __name__ == '__main__':
    fig1_benchmark()
    _head_sweep(RESULTS / 'head_sweep_600m_summary.csv', 'fig2_head_sweep_600m',
                'Head sweep on frozen ESM-C 600M (open-weight)')
    _head_sweep(RESULTS / 'head_sweep_6b_summary.csv', 'fig3_head_sweep_6b',
                'Head sweep on frozen ESM-C 6B')
    fig_hnm(RESULTS / 'hnm_rounds.csv', 'fig4_hnm_6b',
            'Hard-negative mining (frozen ESM-C 6B + DeepSet)', max_round=3)
    if (RESULTS / 'hnm_rounds_600m.csv').exists():
        fig_hnm(RESULTS / 'hnm_rounds_600m.csv', 'fig5_hnm_600m',
                'Hard-negative mining (frozen ESM-C 600M + seqcnn)')
    print(f'\nKey figures written to {FIGS}/')
