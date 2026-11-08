# -*- coding: utf-8 -*-
"""540_build_deep_inflation_figure.py (final version)

New figure: performance inflation attributable to the position of feature
selection (in-fold vs. selection on the full cohort).
  (a) Paired dumbbell: open marker = in-fold selection, filled marker =
      full-cohort selection, arrow = size of the inflation
  (b) Horizontal bars: inflation for each of the four arms
Style: Times New Roman, no grid, lowercase bold panel letters outside the axes,
four formats (PNG 600 dpi / TIFF LZW / PDF / SVG).
Self-checks (abort if any fails): (1) the y ticks must equal the four arm names
(guards against silently reverting to vertical bars); (2) labels must not overlap;
(3) panel letters must lie fully outside the axes; (4) the panel b labels must not
collide with panel a.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

M = os.path.join(BASE, 'results', 'models')
OUT = os.path.join(BASE, 'results', 'figures_journal')

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0, 'legend.fontsize': 9.0,
    'axes.grid': False, 'axes.linewidth': 0.9, 'legend.frameon': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
})

d = pd.read_csv(os.path.join(M, '401_deep_protocol.csv'), encoding='utf-8-sig')
d.columns = [c.strip() for c in d.columns]
REF = float(d['ref_swing_handcrafted'].iloc[0])

LBL = {'深度·DWI': 'DWI embedding\n(512 candidates)',
       '深度·DCE': 'DCE embedding\n(512 candidates)',
       '深度·DWI+DCE': 'Concatenated embedding\n(1024 candidates)'}
COL = {'深度·DWI': '#2C7FB8', '深度·DCE': '#F0A202', '深度·DWI+DCE': '#D7301F'}
SHORT = ['Handcrafted', 'DWI', 'DCE', 'Concatenated']

fig, AX = plt.subplots(1, 2, figsize=(7.6, 3.5))
fig.subplots_adjust(left=0.205, right=0.985, top=0.86, bottom=0.16, wspace=0.34)
a0, a1 = AX[0], AX[1]

# ── (a) in-fold vs. full cohort: paired horizontal dumbbell ──
ys = np.arange(len(d))[::-1]
for y, (_, r) in zip(ys, d.iterrows()):
    c = COL[r['arm']]
    a0.annotate('', xy=(r['C_leaky'], y), xytext=(r['C_infold'], y),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.7, shrinkA=0, shrinkB=0))
    a0.plot([r['C_infold']], [y], 'o', ms=6.4, mfc='white', mec=c, mew=1.6, zorder=4)
    a0.plot([r['C_leaky']], [y], 'o', ms=6.4, mfc=c, mec=c, zorder=4)
    a0.text(r['C_leaky'] + 0.012, y, '+%.3f' % r['swing'], va='center', ha='left',
            fontsize=9.0, color='black')
a0.axvline(0.50, color='black', ls='--', lw=0.9, zorder=1)
a0.text(0.50, -0.62, 'chance', ha='center', va='bottom', fontsize=8.4, color='black')
a0.set_yticks(ys); a0.set_yticklabels([LBL[a] for a in d['arm']], fontsize=8.6, linespacing=1.25)
a0.set_ylim(-0.75, len(d) - 0.25); a0.set_xlim(0.46, 0.735)
a0.set_xlabel('$C$-index (mean over 20 partitions)', labelpad=4)
a0.text(0.0, 1.015, 'a', transform=a0.transAxes, ha='left', va='bottom',
        fontsize=21, fontweight='bold')
a0.plot([], [], 'o', ms=6.4, mfc='white', mec='black', mew=1.6, label='in-fold selection')
a0.plot([], [], 'o', ms=6.4, mfc='black', mec='black', label='full-cohort selection')
a0.legend(loc='lower right', bbox_to_anchor=(1.02, -0.30), ncol=2, fontsize=8.6,
          handletextpad=0.35, columnspacing=1.0)

# ── (b) size of the inflation: horizontal bars ──
vals = [REF] + [float(d.loc[d.arm == a, 'swing'].iloc[0]) for a in ('深度·DWI', '深度·DCE', '深度·DWI+DCE')]
cols = ['#7F8C8D', COL['深度·DWI'], COL['深度·DCE'], COL['深度·DWI+DCE']]
yb = np.arange(len(vals))[::-1]
a1.barh(yb, vals, height=0.58, color=cols, edgecolor='black', linewidth=0.7, zorder=3)
for y, v in zip(yb, vals):
    a1.text(v + max(vals) * 0.025, y, '%.3f' % v, va='center', ha='left',
            fontsize=8.8, color='black')
a1.set_yticks(yb); a1.set_yticklabels(SHORT, fontsize=8.6)
a1.set_ylim(-0.75, len(vals) - 0.25)
a1.set_xlim(0, max(vals) * 1.22)
a1.set_xlabel('$\\Delta$ $C$-index (full-cohort $-$ in-fold)', labelpad=4)
a1.text(0.0, 1.015, 'b', transform=a1.transAxes, ha='left', va='bottom',
        fontsize=21, fontweight='bold')

for _a in fig.axes:
    _a.grid(False)

# ── self-checks ──
fig.canvas.draw()
r_ = fig.canvas.get_renderer()
ylab = [t.get_text() for t in a1.get_yticklabels() if t.get_text().strip()]
assert ylab == SHORT, 'panel b y ticks are not the four arm names (still vertical bars?): %s' % ylab      # (1)
print('[layout] panel b y ticks = %s  OK (horizontal bars)' % ylab)
tl = [t.get_window_extent(renderer=r_) for t in a1.get_yticklabels() if t.get_text().strip()]
bad = [(i, j) for i in range(len(tl)) for j in range(i + 1, len(tl)) if tl[i].overlaps(tl[j])]
print('[ytick] %d labels, overlapping pairs = %d %s' % (len(tl), len(bad), 'OK' if not bad else '!!'))   # (2)
ra = a0.get_window_extent(renderer=r_)
gap = min(t.x0 for t in tl) - ra.x1
print('[spacing] panel b label left edge - panel a right edge = %.1f px %s' % (gap, 'OK' if gap > 2 else '!! collision'))  # (4)
for nm, ax in (('a', a0), ('b', a1)):
    bb = ax.get_window_extent(renderer=r_)
    for t in ax.texts:
        if t.get_text() in ('a', 'b'):
            tb = t.get_window_extent(renderer=r_)
            assert tb.y0 >= bb.y1 - 1, 'panel %s letter overlaps the axes area' % nm
            print('[panel %s] letter outside the axes area OK (%.0f >= %.0f)' % (nm, tb.y0, bb.y1))          # (3)
assert not bad and gap > 2

for e in ('png', 'pdf', 'tiff', 'svg'):
    p = os.path.join(OUT, 'SHR7_selection_leakage.' + e)
    if e == 'png':
        fig.savefig(p, dpi=600, bbox_inches='tight')
    elif e == 'tiff':
        fig.savefig(p, dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    else:
        fig.savefig(p, bbox_inches='tight')
    print('wrote %s  %.1f KB' % (os.path.basename(p), os.path.getsize(p) / 1024))
print('values: handcrafted (5)=%.4f  DWI=%.4f  DCE=%.4f  concatenated=%.4f' % tuple(vals))
