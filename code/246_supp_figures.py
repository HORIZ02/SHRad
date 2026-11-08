# -*- coding: utf-8 -*-
"""202_newfig_chain.py — new analysis-result figures (second batch)
   N3  attribution chain: four-step exclusion chain (one panel per cohort, Δ ± 95% CI forest)
   N4  exhaustive family-expansion test: A/B/C pool comparison + adjudication on fixed additions
   N5  redundancy spectrum and elimination test: max|ρ| distribution + volcano plot (FDR verdict marked)
   N6  calibration and decision curves (two panels per cohort)
Conventions as in 200: Times New Roman / black text / 6.90 in double column / PNG+TIFF(LZW)+PDF+SVG
      with PNG and TIFF converted to RGB / panel letters a,b at the top left, one size / no truncated-axis
      bar charts / text is audited for neither overlapping nor escaping its box
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # ★ sibling module qa_text_line
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# NOGRID: switch grids off globally
plt.rcParams['axes.grid'] = False
from matplotlib import rcParams
from matplotlib.transforms import Bbox
import warnings
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
PROSTATE_ROOT = os.environ.get("PROSTATE_ROOT", os.path.abspath(os.getcwd()))
warnings.filterwarnings('ignore')

H2 = BASE
PRO = os.path.join(PROSTATE_ROOT, '')
M = os.path.join(H2, 'results', 'models')
CD = os.path.join(PRO, 'results', 'cross_dataset')
OUT = os.path.join(H2, 'results', 'figures_journal')
os.makedirs(OUT, exist_ok=True)

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 10,
                 'axes.linewidth': 0.9, 'legend.frameon': False, 'savefig.facecolor': 'white',
                 'figure.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                 'svg.fonttype': 'none'})
for _k in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    rcParams[_k] = 'black'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['mathtext.fontset'] = 'custom'
for k, v in [('mathtext.rm', 'Times New Roman'), ('mathtext.it', 'Times New Roman:italic'),
             ('mathtext.bf', 'Times New Roman:bold'), ('mathtext.sf', 'Times New Roman'),
             ('mathtext.tt', 'Times New Roman'), ('mathtext.cal', 'Times New Roman:italic')]:
    rcParams[k] = v
rcParams['mathtext.default'] = 'it'

C1, C2, C3 = '#3498DB', '#2ECC71', '#E74C3C'      # (1) whole tumour / (2) habitat / (3) spatial
CCLIN = '#7F8C8D'
C_NEW, C_CS, C_DS = '#8E44AD', '#E67E22', '#16A085'   # novel spatial / classical spatial statistics / distribution shape


def _rgb(path, ext, dpi=600):
    """★ PNG and TIFF must be RGB and must carry the correct DPI metadata (submission systems read it).
    Unless dpi is given explicitly, PIL's save writes TIFF as (1,1) and drops pHYs from PNG."""
    if ext not in ('png', 'tiff'):
        return
    try:
        from PIL import Image
        im = Image.open(path)
        need = (im.mode != 'RGB') or (tuple(im.info.get('dpi') or (0, 0)) != (dpi, dpi))
        if need:
            kw = {'dpi': (dpi, dpi)}
            if ext == 'tiff':
                kw['compression'] = 'tiff_lzw'
            im.convert('RGB').save(path, **kw)
    except Exception as e:
        print('  !! RGB/DPI fix failed %s: %s' % (path, e), flush=True)


def _text_has_ink(fig, bb, thr=0.01):
    """★ Post-render ink test: text whose bounding box holds fewer than thr non-white pixels counts
    as not drawn.

    get_visible() and alpha are not enough: text drawn below an opaque axes background (lower
    zorder) and white text both pass those checks yet leave not one pixel (4 false positives on SHR6).
    """
    try:
        import numpy as _np
        buf = _np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        return True                      # do not raise a false alarm when the buffer is unavailable
    H, W = buf.shape[0], buf.shape[1]
    y0 = max(int(H - bb.y1), 0); y1 = min(int(H - bb.y0) + 1, H)
    x0 = max(int(bb.x0), 0); x1 = min(int(bb.x1) + 1, W)
    if y1 <= y0 or x1 <= x0:
        return False
    sub = buf[y0:y1, x0:x1, :3]
    return float((sub.min(axis=2) < 200).mean()) >= thr


def save(fig, name, dpi=600):
    _MM_COH = {'S1_study_flow_and_centre_heterogeneity': 'Cohort A',
               'S5_classifier_sweep_cohort_b': 'Cohort B',
               'S6_feature_selection_sweep_cohort_b': 'Cohort B',
               'S7_sample_definition_sensitivity': 'Cohort A'}
    _MM_NARROW = {'S5_classifier_sweep_cohort_b', 'S6_feature_selection_sweep_cohort_b',
                  'S7_sample_definition_sensitivity'}
    if name in _MM_COH:
        _Y = 0.004 if name.startswith('S1_') else 0.996
        if name in _MM_NARROW:
            # place the cohort tag just left of the y-tick labels, and tighten the left margin
            _XL = min(0.160, max(0.020, 0.55 * min(_ax.get_position().x0 for _ax in fig.axes)))
        else:
            _XL = 0.004
        fig.text(_XL, _Y, _MM_COH[name], ha='left', va='bottom' if _Y < 0.5 else 'top', fontsize=9.5, fontweight='bold', color='black')

    from matplotlib import font_manager as _fm
    from matplotlib.ft2font import FT2Font as _FT
    try:
        _cm = _FT(_fm.findfont(_fm.FontProperties(family='Times New Roman'))).get_charmap()
        _bad = {}
        for _t in fig.findobj(match=lambda o: hasattr(o, 'get_text')):
            try: _s = _t.get_text()
            except Exception: continue
            if not _s: continue
            for _c in _s:
                if ord(_c) >= 128 and ord(_c) not in _cm:
                    _bad[_c] = _bad.get(_c, 0) + 1
        if _bad:
            print('  !! missing glyph %s: ' % name + ', '.join('%r(U+%04X)x%d' % (c, ord(c), n) for c, n in _bad.items()), flush=True)
    except Exception:
        pass

    def _tight():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        return bb, bb.width, bb.height
    fig.canvas.draw(); _r0 = fig.canvas.get_renderer()
    _wt = []
    for _t in fig.findobj(match=lambda o: hasattr(o, 'get_window_extent') and hasattr(o, 'get_text')):
        try: _ss = _t.get_text()
        except Exception: continue
        if not _ss or not _ss.strip(): continue
        try: _bb = _t.get_window_extent(_r0)
        except Exception: continue
        if _bb.width > 0: _wt.append((_ss, _bb.width / fig.dpi))
    bb, w, h = _tight()
    target = 3.35 if w <= 4.81 else 6.90
    _tr = []
    for _ in range(12):
        _tr.append(round(w, 3))
        if w <= target + 1e-6: break
        fig.set_size_inches(*[x * (target / w) for x in fig.get_size_inches()])
        bb, w, h = _tight()
    if w > target + 0.01:
        print('  ~~ narrowing did not converge %s: %.3f > %.2f in (the widest text did not shrink with the figure)' % (name, w, target), flush=True)
        for _s, _wi in sorted(_wt, key=lambda t: -t[1])[:3]:
            print('     widest text %.2f in: %r' % (_wi, _s[:58]), flush=True)
    if h > 8.60:
        s = 8.60 / h
        fig.set_size_inches(*[x * s for x in fig.get_size_inches()]); bb, w, h = _tight()
        if w > target + 1e-6:
            s2 = target / w
            fig.set_size_inches(*[x * s2 for x in fig.get_size_inches()]); bb, w, h = _tight()
    pad = max((target - w) / 2.0, 0.0)
    if name in {'S5_classifier_sweep_cohort_b', 'S6_feature_selection_sweep_cohort_b',
                'S7_sample_definition_sensitivity'}:
        pad *= 0.18          # these three panels had a wide empty left band
    PADY = 0.05
    nb = Bbox.from_extents(bb.x0 - pad, bb.y0 - PADY, bb.x1 + pad, bb.y1 + PADY)
    if not (1.0 < nb.width < 7.0 and 0.8 < nb.height < 9.0):
        raise RuntimeError('figure size out of range %s: %.2f x %.2f in' % (name, nb.width, nb.height))
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(rr) for ax in fig.axes if ax.get_visible()]
    _txt, _seen = [], set()
    for _t in fig.findobj(match=lambda o: hasattr(o, 'get_window_extent') and hasattr(o, 'get_text')):
        try: _s = _t.get_text()
        except Exception: continue
        if not _s or not _s.strip():
            continue
        # ★ Skip invisible text: alpha=0 probe strings and invisible placeholders take no part in the
        #   visual overlap test, otherwise the probe used to measure text width is reported as an
        #   overlap (this produced 2 false positives on SHR6).
        try:
            if not _t.get_visible() or _t.get_alpha() == 0:
                continue
        except Exception:
            pass
        try: tb = _t.get_window_extent(rr)
        except Exception: continue
        if not (tb.width > 0 and tb.height > 0):
            continue
        if not _text_has_ink(fig, tb):
            continue                     # ★ no ink = not drawn (see the _text_has_ink docstring)
        key = (_s, round(tb.x0, 1), round(tb.y0, 1))
        if key in _seen: continue
        _seen.add(key); _txt.append((_s, tb, _t))   # ★ keep the Text object itself: panel ownership is decided exactly from t.axes
    _ov = []

    def _na(bb, art):
        """Panel ownership: ★ decided from the text's own axes (Text.axes), never guessed from
        position - in a 2x2 layout a positional guess mixes up the panels of the same column in the
        two rows (when both have y = -1 it falls back to guessing by x)."""
        try:
            return list(fig.axes).index(art.axes)
        except Exception:
            pass
        if not boxes:
            return -1
        cx_ = (bb.x0 + bb.x1) / 2.0
        return int(np.argmin([abs((b.x0 + b.x1) / 2.0 - cx_) for b in boxes]))

    for _i in range(len(_txt)):
        for _j in range(_i + 1, len(_txt)):
            if not _txt[_i][1].overlaps(_txt[_j][1]): continue
            a, b = _txt[_i][1], _txt[_j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox * oy > 4.0:
                _ov.append((_txt[_i][0][:24], _txt[_j][0][:24], round(ox * oy, 1), a, b,
                            _txt[_i][2], _txt[_j][2]))
    if _ov:
        print('  !! text overlap %s (%d pairs): ' % (name, len(_ov)) +
              ' | '.join('%r[ax%d %s]~%r[ax%d %s]@%s' %
                         (x, _na(pa, oa), tuple(int(v) for v in (pa.x0, pa.y0, pa.x1, pa.y1)),
                          y, _na(pb, ob), tuple(int(v) for v in (pb.x0, pb.y0, pb.x1, pb.y1)), z)
                         for x, y, z, pa, pb, oa, ob in _ov), flush=True)
    for s, tb, _t in _txt:
        if len([b for b in boxes if tb.overlaps(b)]) > 1:
            print('  !! text escapes its box %s: %r' % (name, s[:30]), flush=True)
    # ★ Third gate: text vs. line segments (journals require that numbers do not sit on a line)
    try:
        from qa_text_line import audit as _tl_audit
        _tl_audit(fig, name)
    except ImportError:
        print('  !! qa_text_line not found, skipping the text-on-line audit', flush=True)
    for ext, kw in [('png', dict(dpi=dpi)), ('tiff', dict(dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})),
                    ('pdf', {}), ('svg', {})]:
        p = os.path.join(OUT, '%s.%s' % (name, ext))
        fig.savefig(p, bbox_inches=nb, **kw)
        _rgb(p, ext, dpi)
    plt.close(fig)
    print('  OK %s' % name, flush=True)


def pu(s):
    """Parse a CI string of the form 'a~b'."""
    a, b = str(s).split('~'); return float(a), float(b)


def forest(ax, rows, xlabel, xlim, letter, note=None, xticks=None):
    """rows: list of dict(label, d, lo, hi, p, head=False)"""
    ys = list(range(len(rows)))
    for y, r in zip(ys, rows):
        if r.get('head'):
            continue
        ax.errorbar([r['d']], [y], xerr=[[max(r['d'] - r['lo'], 0)], [max(r['hi'] - r['d'], 0)]],
                    fmt='o', ms=4.8, lw=1.1, capsize=2.4, color=r.get('c', '#2C3E50'), zorder=3)
    # ★ The annotation column and the right x limit must follow the data: the limit has to hold both
    #   the longest error bar and the annotation column, otherwise the error-bar cap runs through the
    #   numbers ("digit on a line"). The text width is measured with the renderer, not hard-coded.
    _pb = ax.text(0, 0, '0.000   P = 0.000', fontsize=6.6, alpha=0.0)
    try:
        ax.figure.canvas.draw(); _rr = ax.figure.canvas.get_renderer()
        _w = _pb.get_window_extent(_rr).width * (xlim[1] - xlim[0]) / ax.get_window_extent(_rr).width
    except Exception:
        _w = 0.32 * (xlim[1] - xlim[0])
    _pb.remove()
    _br = [r['d'] + max(r['hi'] - r['d'], 0) for r in rows if not r.get('head')]
    xann = (max(_br) if _br else 0.0) + 0.016 + _w
    _XR = max(xlim[1], xann + 0.016)
    for y, r in zip(ys, rows):
        if r.get('head'):
            continue
        ax.text(xann, y, '%.3f' % r['d'] + ('' if r.get('p') is None else '   P = %.3f' % r['p']),
                ha='right', va='center', fontsize=6.6, color='black')
    ax.axvline(0, color='black', lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r['label'] for r in rows], fontsize=7.0)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r.get('head'):
            t.set_fontweight('bold'); t.set_fontsize(7.2)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(xlim[0], _XR)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    # NOGRID ax.grid(axis='x', color='#DDDDDD', lw=0.5); ax.set_axisbelow(True)
    ax.text(0.012, 0.985, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=14, fontweight='bold')
    if note:
        ax.text(0.992, 1.035, note, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=6.8, color='black')



# ══════════════════════════════════════════════════════════════════════════════
#  Supplementary figures S1 / S3 / S4 / S6 / S7 / S8 (S2 is in 248, S5 in 249)
#  One shared save() and three gates (no text overlap / no text outside its box / no text on a line)
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyBboxPatch
C_EFF, C_WEAK, C_COM = '#E74C3C', '#95A5A6', '#9B59B6'
GREY = '#7F8C8D'

# ══════════════════════ S1  study flow and centre heterogeneity ══════════════════════
fig, AX = plt.subplots(1, 2, figsize=(6.90, 3.6))
fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.108, wspace=0.156)
a0, a1 = AX[0], AX[1]

# ── (a) flow chart ──
# ★ Changed 2026-10-13: (1) the first two boxes are merged (the former "Archived...cohort" box had no
#   patient count and duplicated "Analysis cohort 345"); (2) the arrows now point down (the original
#   drew them the wrong way round, with the head on the upper box); (3) arrow ends sit exactly on the
#   box edges with equal width, height and spacing and a single arrow size and line width, so nothing
#   is slanted or of unequal length.
a0.set_xlim(0, 1); a0.set_ylim(0, 1); a0.axis('off')
_BOXES = [
    ('Three-centre MRI cohort\n345 patients, 51 DFS events (14.8%)', '#EEF3F8'),
    ('Habitat segmentation: DCE voxel-wise clustering,\nK = 3 by the gap-statistic 1-SE rule', '#F3F0F8'),
    ('Feature extraction: 84 spatial habitat descriptors\n+ 856 conventional radiomics features', '#F3F0F8'),
    ('Modellable: 81 spatial (3 DWI fractal-dimension\ncolumns excluded) and 856 conventional', '#EEF7EF'),
]
_h, _g = 0.132, 0.070
_x0, _w = 0.075, 0.85
_edges = []
_total = len(_BOXES) * _h + (len(_BOXES) - 1) * _g
_y = 0.5 + _total / 2 - _h            # centre the box stack vertically in the panel, aligned with the (b) plot area
for txt, fc in _BOXES:
    a0.add_patch(FancyBboxPatch((_x0, _y), _w, _h, boxstyle='round,pad=0.006',
                                fc=fc, ec='#666666', lw=0.85))
    a0.text(0.50, _y + _h / 2, txt, ha='center', va='center', fontsize=6.6, linespacing=1.30)
    _edges.append((_y, _y + _h))
    _y -= (_h + _g)
from matplotlib.patches import FancyArrowPatch
# ★ 2026-10-18: the boxes use boxstyle='round,pad=0.006', so the drawn border extends 0.006 beyond
#   the given rectangle; taking the rectangle edge (y, y+h) as the end point would push the arrow
#   0.006 into the box and onto its border and fill. End points therefore start from the visual outer
#   edge plus a _GAP gap: the tail below the upper box's outer edge, the head above the lower one's.
_PAD = 0.006
_GAP = 0.005
for i in range(len(_BOXES) - 1):
    y_tail = _edges[i][0] - _PAD - _GAP        # visual bottom edge of the upper box minus the gap
    y_head = _edges[i + 1][1] + _PAD + _GAP    # visual top edge of the lower box plus the gap
    a0.add_patch(FancyArrowPatch((0.50, y_tail), (0.50, y_head),
                                 arrowstyle='-|>,head_length=4.2,head_width=2.0',
                                 lw=0.9, color='#333333', shrinkA=0, shrinkB=0,
                                 joinstyle='miter', capstyle='butt'))
a0.text(0.0, 1.015,  'a', transform=a0.transAxes, ha='left', va='bottom',
        fontsize=21, fontweight='bold')

# ── (b) heterogeneity across centres ──
CN = ['Center 1\n(n = 193)', 'Center 2\n(n = 78)', 'Center 3\n(n = 74)']
EV = [9.3, 14.1, 29.7]
PN = [18.1, 42.3, 71.6]
xs = np.arange(3)
a1.bar(xs - 0.19, EV, width=0.36, color=C_WEAK, edgecolor='none', label='DFS event rate')
a1.bar(xs + 0.19, PN, width=0.36, color=C_COM, edgecolor='none', label='Node-positive (pN ≥ 1)')
for x, v in zip(xs - 0.19, EV):
    a1.text(x, v + 1.4, '%.1f' % v, ha='center', va='bottom', fontsize=6.5, color='black')
for x, v in zip(xs + 0.19, PN):
    a1.text(x, v + 1.4, '%.1f' % v, ha='center', va='bottom', fontsize=6.5, color='black')
a1.set_xticks(xs); a1.set_xticklabels(CN, fontsize=7.0)
a1.set_ylim(0, 88); a1.set_yticks([0, 20, 40, 60, 80])
a1.set_ylabel('Proportion of patients (%)', labelpad=8)
a1.set_xlabel('Centre', labelpad=4)
# NOGRID a1.grid(axis='y', color='#DDDDDD', lw=0.5); a1.set_axisbelow(True)
a1.text(0.0, 1.015,  'b', transform=a1.transAxes, ha='left', va='bottom',
        fontsize=21, fontweight='bold')
# ★ Legend inside the axes at the top left: the tallest bar is 71.6 against a limit of 88, so there
# is room above; outside the axes it would compete with the footnote for the same bottom space
a1.legend(loc='upper left', bbox_to_anchor=(0.005, 0.995), ncol=2, fontsize=7.0, handletextpad=0.4)
# LONG TEXT MOVED TO CAPTION
# fig.text(0.5, 0.012, 'Percentages are of each centre\'s own patients; the three centres differ in event rate and in nodal burden alike.',
# ha='center', va='bottom', fontsize=6.4, color='black')
save(fig, 'S1_study_flow_and_centre_heterogeneity')

# ══════════════════════ S3  identifiability of K ══════════════════════
gp = pd.read_csv(os.path.join(M, 'gap_statistic.csv'), encoding='utf-8-sig')
km = pd.read_csv(os.path.join(M, 'k_optimality_metrics.csv'), encoding='utf-8-sig')
ks = pd.read_csv(os.path.join(M, 'k_stability_summary.csv'), encoding='utf-8-sig')
bc = pd.read_csv(os.path.join(M, 'bic_recheck.csv'), encoding='utf-8-sig')
fig, AX = plt.subplots(1, 3, figsize=(6.90, 3.95))
fig.subplots_adjust(left=0.095, right=0.972, top=0.882, bottom=0.150, wspace=0.78)
MOD = [('DCE', C3, 'o'), ('DWI', C1, 's')]

# ── (a) gap statistic, marking K = 3 as chosen by the 1-SE rule ──
for tag, col, mk in MOD:
    g = gp[gp.modality.str.lower() == tag.lower()]   # ★ the tables disagree on case (gap uses 'DCE', k_optimality/k_stability use 'dce')
    ms, ss = [], []
    for K in range(2, 8):
        v = g['gap_%d' % K].astype(float)
        ms.append(v.mean()); ss.append(g['s_%d' % K].astype(float).mean())
    AX[0].errorbar(range(2, 8), ms, yerr=ss, fmt=mk + '-', ms=4.2, lw=1.1, capsize=2.0,
                   color=col, label=tag, zorder=3)
AX[0].axvline(3, color=C3, ls='--', lw=0.9, zorder=1)
AX[0].annotate('1-SE rule\nselects $K$ = 3', xy=(3, 1.02), xycoords=('data', 'axes fraction'),
               xytext=(0, 3.5), textcoords='offset points', ha='left', va='bottom',
               fontsize=6.3, color=C3)
AX[0].set_xticks(range(2, 8)); AX[0].set_xlabel('Number of habitats $K$', labelpad=4)
AX[0].set_ylabel('Gap statistic', labelpad=9)
AX[0].set_ylim(0.4, 1.75)
# NOGRID AX[0].grid(color='#DDDDDD', lw=0.5); AX[0].set_axisbelow(True)
AX[0].text(0.0, 1.015,  'a', transform=AX[0].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
# MOVLEG AX[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.195), ncol=2, fontsize=6.6, handletextpad=0.35)

# ── (b) silhouette vs. K (boundary-limited) ──
for tag, col, mk in MOD:
    g = km[km.modality.str.lower() == tag.lower()]
    ms = [g[g.K == K].silhouette.astype(float).mean() for K in range(2, 8)]
    AX[1].plot(range(2, 8), ms, mk + '-', ms=3.8, lw=1.1, color=col, label=tag, zorder=3)
# ★ The annotation must stay narrow: text at a fixed size does not wrap, so a long line crosses the
# panel's right edge and covers the neighbouring panel's tick labels
AX[1].text(0.33, 0.945, 'boundary-limited:\nno interior optimum',
           transform=AX[1].transAxes, ha='left', va='top', fontsize=6.6, color='black')
AX[1].set_xticks(range(2, 8)); AX[1].set_xlabel('Number of habitats $K$', labelpad=4)
AX[1].set_ylabel('Silhouette coefficient\n(mean over cases)', labelpad=6)
# NOGRID AX[1].grid(color='#DDDDDD', lw=0.5); AX[1].set_axisbelow(True)
AX[1].text(0.0, 1.015,  'b', transform=AX[1].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
# MOVLEG AX[1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.195), ncol=2, fontsize=6.6, handletextpad=0.35)

# ── (c) partition stability + BIC mode ──
for tag, col, mk in MOD:
    g = ks[ks.modality.str.lower() == tag.lower()]
    Ks = sorted(g.K.unique())
    AX[2].errorbar(Ks, [g[g.K == K]['mean'].astype(float).values[0] for K in Ks],
                   yerr=[g[g.K == K]['std'].astype(float).values[0] for K in Ks],
                   fmt=mk + '-', ms=3.8, lw=1.1, capsize=2.0, color=col, label=tag, zorder=3)
_bw = [int(bc[bc.modality == tag].bestK_w.mode().values[0]) for tag, _, _ in MOD]
# ★ Panel c has no safe empty space (the DCE/DWI stability curves fill it) and the point is already
#   carried by the footnote; a second annotation would only cross the curves, so none is added.
AX[2].set_xticks(range(2, 8)); AX[2].set_xlabel('Number of habitats $K$', labelpad=4)
AX[2].set_ylabel('Partition stability (ARI)', labelpad=10)
# NOGRID AX[2].grid(color='#DDDDDD', lw=0.5); AX[2].set_axisbelow(True)
AX[2].text(0.0, 1.015,  'c', transform=AX[2].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
# MOVLEG AX[2].legend(loc='upper left', bbox_to_anchor=(0.0, -0.195), ncol=2, fontsize=6.6, handletextpad=0.35)
# LONG TEXT MOVED TO CAPTION (requirement: no long text inside figures)
# fig.text(0.5, 0.015, 'Only the gap statistic has an interior optimum, and only in DCE.\n'
# 'Silhouette, partition stability and Gaussian-mixture BIC are boundary-limited and cannot arbitrate $K$.',
# ha='center', va='bottom', fontsize=6.2, color='black', linespacing=1.4)

#save(fig, 'S3_identifiability_of_k')  # content merged into main-text Figure 3

# ══════════════════════ S4  sensitivity of the model comparison to K ══════════════════════
kss = pd.read_csv(os.path.join(M, 'k_sensitivity.csv'), encoding='utf-8-sig')
ksf = pd.read_csv(os.path.join(M, 'k_spatial_full.csv'), encoding='utf-8-sig')
fig, AX = plt.subplots(1, 2, figsize=(6.90, 2.9))
fig.subplots_adjust(left=0.085, right=0.988, top=0.885, bottom=0.158, wspace=0.234)

# ── (a) discrimination vs. K ──
for mdl, col, mk in [('Clinical 3', GREY, 's'), ('Clinical 3 + spatial', C3, 'o')]:
    g = kss[kss.model == mdl].sort_values('K')
    AX[0].errorbar(g.K, g.C_mean, yerr=g.C_sd, fmt=mk + '-', ms=4.2, lw=1.1, capsize=2.0,
                   color=col, label=mdl, zorder=3)
AX[0].set_xticks(range(2, 7)); AX[0].set_xlabel('Number of habitats $K$', labelpad=4)
AX[0].set_ylabel('Concordance index', labelpad=6)
AX[0].set_ylim(0.655, 0.760)
# NOGRID AX[0].grid(color='#DDDDDD', lw=0.5); AX[0].set_axisbelow(True)
AX[0].text(0.0, 1.015,  'a', transform=AX[0].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
AX[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.187), ncol=1, fontsize=6.8, handletextpad=0.35)

# ── (b) spatial habitat pool vs. K (three modality sets) ──
for ms, col, mk, lab in [('DCE', C3, 'o', 'DCE'), ('DWI', C1, 's', 'DWI'), ('DWI+DCE', C_COM, '^', 'DWI + DCE')]:
    g = ksf[(ksf.modset == ms) & (ksf.pool == 'all')].sort_values('K')
    g = g[g.K >= 2]
    AX[1].errorbar(g.K, g['mean'], yerr=g['std'], fmt=mk + '-', ms=4.2, lw=1.1, capsize=2.0,
                   color=col, label=lab, zorder=3)
    r0 = ksf[(ksf.modset == ms + ' [clin ref]') & (ksf.K == 0)]
    if len(r0):
        AX[1].axhline(float(r0['mean'].values[0]), color=col, ls=':', lw=0.9, zorder=1)
AX[1].set_xticks(range(2, 7)); AX[1].set_xlabel('Number of habitats $K$', labelpad=4)
AX[1].set_ylabel('Concordance index\n(spatial pool, all descriptors)', labelpad=6)
AX[1].set_ylim(0.63, 0.775)
# NOGRID AX[1].grid(color='#DDDDDD', lw=0.5); AX[1].set_axisbelow(True)
AX[1].text(0.0, 1.015,  'b', transform=AX[1].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
AX[1].text(0.988, 1.045, 'dotted: clinical reference', transform=AX[1].transAxes,
           ha='right', va='bottom', fontsize=6.5, color='black')
AX[1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.187), ncol=3, fontsize=6.8, handletextpad=0.35)
#save(fig, 'S4_sensitivity_to_k')  # content merged into main-text Figure 3

# ══════════════════════ S6  20 classifiers in cohort B ══════════════════════
ac = pd.read_csv(os.path.join(PRO, 'results', 'all_classifiers', 'results.csv'), encoding='utf-8-sig')
# ★ Learners that did not converge (TabPFN) have NaN AUC/CI and must be dropped - the caption states
#   "did not converge and is not shown"; keeping them paints a 'nan' row label and a long off-scale line.
_n0 = len(ac)
ac = ac.dropna(subset=['AUC', 'CI_low', 'CI_high']).sort_values('AUC')
print('  S6: %d non-converged learners dropped (%d -> %d)' % (_n0 - len(ac), _n0, len(ac)))
fig, ax = plt.subplots(figsize=(6.90, 4.5))
fig.subplots_adjust(left=0.285, right=0.985, top=0.945, bottom=0.084)
ys = np.arange(len(ac))
COLS = [C_EFF if 'Ridge' in c or 'LDA' in c or c.startswith('LR') else C_WEAK for c in ac.Classifier]
ax.errorbar(ac.AUC, ys, xerr=[ac.AUC - ac.CI_low, ac.CI_high - ac.AUC],
            fmt='none', ecolor='#BBBBBB', elinewidth=1.0, capsize=2.0, zorder=2)
ax.scatter(ac.AUC, ys, s=26, c=COLS, zorder=3, edgecolors='none')
# ★ The values sit just outside the rightmost error bar (the largest observed CI bound is 0.802) in a
#   single aligned column: attached to each point they would be crossed by that point's upper cap and
#   the gate would report "digit on a line".
XL = 0.810
# ★ Values are rounded half-up on purpose: Python's default '%.3f' (half-even) prints TabNet's 0.6745
#   as 0.674 whereas both the text and the caption give 0.675, which would make figure and text
#   disagree. Rounding half-up here keeps the two consistent.
from decimal import Decimal, ROUND_HALF_UP as _RHE
_f3 = lambda v: str(Decimal(repr(float(v))).quantize(Decimal('0.001'), rounding=_RHE))
for y, v in zip(ys, ac.AUC):
    ax.text(XL, y, _f3(v), ha='left', va='center', fontsize=6.4, color='black')
ax.axvline(float(ac[ac.Classifier.str.contains('Ridge')].AUC.values[0]), color=GREY, ls=':', lw=0.9, zorder=1)
ax.set_yticks(ys); ax.set_yticklabels(ac.Classifier, fontsize=6.9)
ax.set_xlim(0.60, 0.870); ax.set_ylim(-0.7, len(ac) - 0.3)
ax.set_xlabel('Out-of-fold AUC', labelpad=4)
# NOGRID ax.grid(axis='x', color='#DDDDDD', lw=0.5); ax.set_axisbelow(True)
ax.text(0.988, 1.012, 'red: linear learners', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=6.6, color='black')
save(fig, 'S5_classifier_sweep_cohort_b')

# ══════════════════════ S7  five feature-selection rules in cohort B ══════════════════════
fs = pd.read_csv(os.path.join(PRO, 'results', 'feature_selection', 'results.csv'), encoding='utf-8-sig')
fs = fs.sort_values('AUC')
NREF = 0.7503
fig, ax = plt.subplots(figsize=(6.90, 2.55))
fig.subplots_adjust(left=0.235, right=0.985, top=0.885, bottom=0.146)
ys = np.arange(len(fs))
ax.scatter(fs.AUC, ys, s=34, c=[C_EFF if 'Univariate' in m else GREY for m in fs.Method],
           zorder=3, edgecolors='none')
for y, v in zip(ys, fs.AUC):
    ax.text(v - 0.004, y, '%.3f' % v, ha='right', va='center', fontsize=6.6, color='black')
ax.axvline(NREF, color=C_EFF, ls=':', lw=1.0, zorder=1)
ax.annotate('the 9-variable model 0.750', xy=(NREF, len(fs) - 0.55), xycoords='data',
            xytext=(-4, 0), textcoords='offset points', ha='right', va='center',
            fontsize=6.4, color=C_EFF)
ax.set_yticks(ys); ax.set_yticklabels(fs.Method, fontsize=7.0)
ax.set_xlim(0.55, 0.79); ax.set_ylim(-0.7, len(fs) - 0.3)
ax.set_xlabel('Out-of-fold AUC (matched pool size: 8 variables)', labelpad=4)
# NOGRID ax.grid(axis='x', color='#DDDDDD', lw=0.5); ax.set_axisbelow(True)
save(fig, 'S6_feature_selection_sweep_cohort_b')

# ══════════════════════ S8  sensitivity to the sample definition (345 vs. 288) ══════════════════════
t2 = pd.read_csv(os.path.join(M, 'tableS2_sample_definition_345.csv'), encoding='utf-8-sig')
_order = ['M0 clinical 11', 'M4c original paper (DCE_H 107 + clinical 11)', 'clinical + P2', 'M4c + P1 (novel spatial 53)',
          'M4c + P2 (all spatial 81)', 'P2_only (spatial only)']
_lab = {'M0 clinical 11': 'Clinical 11', 'M4c original paper (DCE_H 107 + clinical 11)': 'Whole-tumour + clinical',
        'clinical + P2': 'Clinical + all spatial', 'M4c + P1 (novel spatial 53)': 'Novel spatial pool',
        'M4c + P2 (all spatial 81)': 'All spatial pool', 'P2_only (spatial only)': 'Spatial only'}
fig, AX = plt.subplots(1, 2, figsize=(6.90, 3.1))
fig.subplots_adjust(left=0.245, right=0.985, top=0.885, bottom=0.152, wspace=0.406)
rows = [m for m in _order if m in set(t2.model)]
ys = np.arange(len(rows))
# (a) discrimination
for sub, col, mk in [('n345', C3, 'o'), ('n288', C1, 's')]:
    v = [float(t2[(t2.subset == sub) & (t2.model == m)].C_OOF.values[0]) for m in rows]
    AX[0].scatter(v, ys, s=30, marker=mk, color=col, zorder=3,
                  label='%s (n = %d)' % ('all patients' if sub == 'n345' else 'adequate DWI range',
                                         int(t2[t2.subset == sub].n.values[0])))
for y, m in zip(ys, rows):
    v1 = float(t2[(t2.subset == 'n345') & (t2.model == m)].C_OOF.values[0])
    v2 = float(t2[(t2.subset == 'n288') & (t2.model == m)].C_OOF.values[0])
    AX[0].plot([v1, v2], [y, y], color='#CCCCCC', lw=1.0, zorder=2)
AX[0].set_yticks(ys); AX[0].set_yticklabels([_lab[m] for m in rows], fontsize=6.9)
AX[0].set_ylim(len(rows) - 0.4, -0.6); AX[0].set_xlim(0.50, 0.80)
AX[0].set_xlabel('Out-of-fold concordance index', labelpad=4, fontsize=7.4)
AX[0].tick_params(axis='x', labelsize=7.0)
# NOGRID AX[0].grid(axis='x', color='#DDDDDD', lw=0.5); AX[0].set_axisbelow(True)
AX[0].text(0.0, 1.015,  'a', transform=AX[0].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
AX[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.215), ncol=1, fontsize=6.6, handletextpad=0.35)
# (b) increment Δ with its CI
SUB = [('n345', C3, 'o', 'All 345'), ('n288', C1, 's', 'Adequate DWI (288)')]
for sub, col, mk, lab in SUB:
    for y, m in zip(ys, rows):
        r = t2[(t2.subset == sub) & (t2.model == m)]
        if not len(r) or pd.isna(r.dC.values[0]):
            continue
        d = float(r.dC.values[0]); lo = float(r.dC_lo.values[0]); hi = float(r.dC_hi.values[0])
        off = -0.17 if sub == 'n345' else 0.17
        AX[1].errorbar([d], [y + off], xerr=[[max(d - lo, 0)], [max(hi - d, 0)]],
                       fmt=mk, ms=3.8, lw=0.95, capsize=1.8, color=col, label=lab if y == 0 else None,
                       zorder=3)
AX[1].axvline(0, color='black', lw=0.8)
AX[1].set_yticks(ys); AX[1].set_yticklabels([_lab[m] for m in rows], fontsize=6.9)
AX[1].set_ylim(len(rows) - 0.4, -0.6); AX[1].set_xlim(-0.24, 0.09)
AX[1].set_xlabel('$\\Delta$ concordance index vs clinical reference', labelpad=4, fontsize=7.4)
AX[1].tick_params(axis='x', labelsize=7.0)
# NOGRID AX[1].grid(axis='x', color='#DDDDDD', lw=0.5); AX[1].set_axisbelow(True)
AX[1].text(0.0, 1.015,  'b', transform=AX[1].transAxes, ha='left', va='bottom',
           fontsize=21, fontweight='bold')
AX[1].legend(loc='upper left', bbox_to_anchor=(0.0, -0.215), ncol=1, fontsize=6.6, handletextpad=0.35)
save(fig, 'S7_sample_definition_sensitivity')
