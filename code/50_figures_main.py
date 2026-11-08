import os
# -*- coding: utf-8 -*-
"""
50_figures_main.py — main figures F1–F6 (corrected version)

Corrections:
  F1 text overflow -> taller boxes + shorter lines
  F2 annotations crossing lines / offset diagonal -> annotations re-laid out + common axis limits
  F3 misaligned CI -> paired-bootstrap 95% CI of ΔC (the C-index CI was used by mistake)
  F3 model table -> the leaky version is replaced by the parsimonious model with unbiased in-fold selection
"""
import importlib.util, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT, CL, MAX_MISSING = m.BASE, m.OUT, m.CLIN_VARS, m.MAX_MISSING
RES = os.path.join(BASE, 'results', 'models')
FIG = os.path.join(BASE, 'results', 'figures_journal')
os.makedirs(FIG, exist_ok=True)

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 9
for k in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    rcParams[k] = 'black'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['savefig.facecolor'] = 'white'
rcParams['figure.facecolor'] = 'white'
rcParams['legend.frameon'] = False

# ★ Font compliance: $...$ mathtext must use the same serif as the body text (Times New Roman)
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.sf'] = 'Times New Roman'
rcParams['mathtext.tt'] = 'Times New Roman'
rcParams['mathtext.cal'] = 'Times New Roman:italic'
rcParams['mathtext.default'] = 'it'

C_CLIN, C_IMG, C_SPAT, C_PARS = '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'

def save(fig, name, dpi=600):
    """Journal specification: output width exactly 3.35 in (single column) or 6.90 in (double column); height <= 8.60 in; includes a glyph audit."""
    from matplotlib.transforms import Bbox
    # ---- Glyph audit: any visible text whose glyph is missing from Times New Roman is reported ----
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
                    _bad.setdefault(_c, 0)
                    _bad[_c] += 1
        if _bad:
            print(f'  !! missing glyph {name}: ' + ', '.join(f'{c!r}(U+{ord(c):04X})x{n}' for c, n in _bad.items()), flush=True)
    except Exception:
        pass
    def _tight():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())   # note: matplotlib already returns inches
        return bb, bb.width, bb.height
    bb, w, h = _tight()
    target = 3.35 if w <= 4.81 else 6.90    # cut-off = sqrt(3.35 x 6.90); the old value 3.36 padded 3.4 in content up to 6.90 and left a wide white band    # snap to the nearest: <= 3.36 -> single column, otherwise -> double column (no intermediate widths)
    for _ in range(5):                       # iterative narrowing (text sizes are fixed, several passes may be needed)
        if w <= target + 1e-6: break
        fig.set_size_inches(*[x * (target / w) for x in fig.get_size_inches()])
        bb, w, h = _tight()
    if h > 8.60:
        s = 8.60 / h
        fig.set_size_inches(*[x * s for x in fig.get_size_inches()])
        bb, w, h = _tight()
        if w > target + 1e-6:
            s2 = target / w
            fig.set_size_inches(*[x * s2 for x in fig.get_size_inches()])
            bb, w, h = _tight()
    pad = max((target - w) / 2.0, 0.0)
    PADY = 0.05                                   # leave 0.05 in top and bottom so titles and tick labels do not touch the edge
    nb = Bbox.from_extents(bb.x0 - pad, bb.y0 - PADY, bb.x1 + pad, bb.y1 + PADY)
    if not (1.0 < (nb.width) < 7.0 and 0.8 < nb.height < 9.0):
        raise RuntimeError(f'figure size out of range {name}: {nb.width:.2f} x {nb.height:.2f} in')
    dests = []
    for g in ('OUT', 'FIG', 'FIGJ'):
        v = globals().get(g)
        if v and v not in dests:
            dests.append(v)
    for di, d in enumerate(dests):
        os.makedirs(d, exist_ok=True)
        for ext, kw in [('png', dict(dpi=dpi)), ('tiff', dict(dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})),
                        ('pdf', {}), ('svg', {})]:
            if ext == 'svg' and di > 0:
                continue
            fig.savefig(os.path.join(d, f'{name}.{ext}'), bbox_inches=nb, **kw)
    plt.close(fig)
    print(f'  OK {name}', flush=True)

# ================= data =================
a = pd.read_csv(os.path.join(m.BASE, 'analysis', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
y = m.make_surv(a); ev, tm = y['event'].astype(bool), y['time'].astype(float)
order = a['Case_No.']
CEN = sorted(a.Center.unique())
_rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
_rad['Case_No.'] = _rad['Case_No.'].astype(str)
RF = [c for c in _rad.columns if c not in ('Case_No.', 'Habitat')]
radH = _rad[_rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.').loc[order, RF].reset_index(drop=True)
sp = pd.read_csv(os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv'))
sp['Case_No.'] = sp['Case_No.'].astype(str)
dw, dc = sp[sp.modality == 'DWI'].set_index('Case_No.'), sp[sp.modality == 'DCE'].set_index('Case_No.')
spw = pd.concat([dw[[c for c in sp.columns if c.startswith('dwi_')]].loc[order].reset_index(drop=True),
                 dc[[c for c in sp.columns if c.startswith('dce_')]].loc[order].reset_index(drop=True)], axis=1)
ms = spw.isna().mean(); spw = spw[ms[ms <= MAX_MISSING].index]
C3 = a[['NAC', 'N', 'ALN']].reset_index(drop=True).astype(float)
C11 = a[CL].reset_index(drop=True).astype(float)
POOL = pd.concat([radH.add_prefix('R_'), spw.add_prefix('S_')], axis=1)
red = pd.read_csv(os.path.join(RES, 'FINAL_redundancy_4center.csv'))
NOVEL = set(red[red.max_abs_r < 0.80].feature)
P1 = [c for c in spw.columns if c in NOVEL]; P2 = list(spw.columns)
DCEN = [c for c in P1 if c.startswith('dce_')]
print(f'{len(a)} cases / {ev.sum()} events; spatial {spw.shape[1]}; P1={len(P1)}')

def screen_topk(pool_tr, ytr, k):
    sc = []
    for c in pool_tr.columns:
        v = pool_tr[c].values
        if not np.isfinite(v).all() or np.std(v, ddof=1) == 0: continue
        cc = concordance_index_censored(ytr['event'], ytr['time'], v)[0]
        sc.append((c, abs(cc - 0.5), 1 if cc >= 0.5 else -1))
    sc.sort(key=lambda t: -t[1]); return [(c, s) for c, _, s in sc[:k]]

def nested_lasso(X, seed=42):
    """Call 23b's nested_oof directly so the definition matches the summary table."""
    return m.nested_oof(X.fillna(0).astype(float), y, seed=seed)

def nested_insfold(k, seed=42):
    o = np.full(len(a), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(C3.values):
        Mtr, Mte = C3.iloc[tr].copy(), C3.iloc[te].copy(); cols = list(C3.columns)
        if k > 0:
            for c, sgn in screen_topk(POOL.iloc[tr], y[tr], k):
                Mtr[c] = sgn * POOL.iloc[tr][c].values; Mte[c] = sgn * POOL.iloc[te][c].values; cols.append(c)
        sc = StandardScaler(); A = sc.fit_transform(Mtr[cols].values); B = sc.transform(Mte[cols].values)
        o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr]).predict(B)
    return o

# ---- out-of-fold predictions of all models ----
X = {
    'M0 clinical 11': (C11, 'lasso'),
    'M4c': (pd.concat([C11, radH.add_prefix('DCEH_')], axis=1), 'lasso'),
    'M4c+P1': (pd.concat([C11, radH.add_prefix('DCEH_'), spw[P1]], axis=1), 'lasso'),
    'M4c+P2': (pd.concat([C11, radH.add_prefix('DCEH_'), spw[P2]], axis=1), 'lasso'),
    'M4c+DCEspat': (pd.concat([C11, radH.add_prefix('DCEH_'), spw[DCEN]], axis=1), 'lasso'),
    'Clin+P2': (pd.concat([C11, spw[P2]], axis=1), 'lasso'),
    'Clin+DCEspat': (pd.concat([C11, spw[DCEN]], axis=1), 'lasso'),
    'Spatial_only': (pd.concat([spw[P2]], axis=1), 'lasso'),
    'Parsim_k0': (None, 'ins0'),
    'Parsim_k1': (None, 'ins1'),
}
OOF = {}
for k, (XX, kind) in X.items():
    if kind == 'lasso': OOF[k] = nested_lasso(XX.fillna(0).astype(float))
    else: OOF[k] = nested_insfold(int(kind[-1]))
    print(f'  {k}: C = {concordance_index_censored(ev, tm, OOF[k])[0]:.4f}')
M4C = OOF['M4c']

# ============================================================
# F1 — flow + centre heterogeneity
# ============================================================
# F1 — flow chart + centre heterogeneity
#   Layout requirements (2026-09-28): a/b horizontally aligned | panel letters without parentheses | boxes must not overlap
# ============================================================
fig = plt.figure(figsize=(6.9, 3.90))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)

# ---------- a patient flow (all in axes-fraction coordinates for geometric control) ----------
ax = fig.add_subplot(gs[0, 0])
ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
_BOX = []          # collect the box rectangles for the hard "no overlap" assertion

def bx(x0, y0, w, h, txt, fc, ec, fs=7.4):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle='round,pad=0.004,rounding_size=0.02',
                                linewidth=1.0, facecolor=fc, edgecolor=ec,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha='center', va='center',
            fontsize=fs, color='#000000', linespacing=1.45, transform=ax.transAxes)
    _BOX.append((x0, y0, x0 + w, y0 + h))

def arw(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=6, lw=0.8, color='#000000',
                                 transform=ax.transAxes, clip_on=False))

# five-row vertical layout (gaps between rows; arrows live in the gaps so they never cross a box)
L1Y, L1H = 0.865, 0.135
L2Y, L2H = 0.688, 0.115
L3Y, L3H = 0.468, 0.158
L4Y, L4H = 0.248, 0.158
L5Y, L5H = 0.018, 0.165

bx(0.18, L1Y, 0.64, L1H, 'HER2-low breast cancer\n3 centers, 345 patients',
   '#EAECEE', '#000000', 8.0)

_CX = [0.16, 0.50, 0.84]
for _i, _k in enumerate(CEN):
    bx(_CX[_i] - 0.14, L2Y, 0.28, L2H,
       f'{_k.split("-")[0]}\nn = {int((a.Center == _k).sum())}', '#EBF5FB', '#3498DB', 7.8)
    arw(0.50, L1Y - 0.002, _CX[_i], L2Y + L2H + 0.002)

bx(0.06, L3Y, 0.88, L3H,
   'DWI + DCE MRI\nK-means habitat segmentation\n(K = 3: L / M / H, per case)',
   '#EAFAF1', '#2ECC71', 7.6)
for _x in _CX:
    arw(_x, L2Y - 0.002, _x, L3Y + L3H + 0.002)

bx(0.005, L4Y, 0.475, L4H, 'Conventional radiomics\n856 features\n(107 x 4 regions x 2)',
   '#EBF5FB', '#3498DB', 6.6)
bx(0.520, L4Y, 0.475, L4H, 'Spatial habitat radiomics\n84 descriptors, 81 modelled\n(14 types / 4 families)',
   '#EAFAF1', '#2ECC71', 6.6)
arw(0.25, L3Y - 0.002, 0.25, L4Y + L4H + 0.002)
arw(0.75, L3Y - 0.002, 0.75, L4Y + L4H + 0.002)

bx(0.06, L5Y, 0.88, L5H, 'Analysis cohort\n345 patients, 51 DFS events (14.8%)\nmedian follow-up 43 months',
   '#F4ECF7', '#9B59B6', 7.6)
arw(0.25, L4Y - 0.002, 0.40, L5Y + L5H + 0.002)
arw(0.75, L4Y - 0.002, 0.60, L5Y + L5H + 0.002)

# ★ Boxes must not overlap (hard journal layout requirement; any intersection aborts the run)
for _i in range(len(_BOX)):
    for _j in range(_i + 1, len(_BOX)):
        x0a, y0a, x1a, y1a = _BOX[_i]
        x0b, y0b, x1b, y1b = _BOX[_j]
        _ox = min(x1a, x1b) - max(x0a, x0b)
        _oy = min(y1a, y1b) - max(y0a, y0b)
        assert not (_ox > 1e-9 and _oy > 1e-9), f'overlapping boxes {_BOX[_i]} vs {_BOX[_j]}'

# Panel letter a: axes coordinates; same y as b so the two stay strictly aligned; no parentheses
_LBL_Y = 1.035
ax.text(0.0, _LBL_Y, 'a', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=10, fontweight='bold', color='#000000')

# ★ Text must lie entirely inside its box (hard journal layout requirement).
#   Method: draw once, then compare the text bbox with the patch bbox; anything outside aborts.
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_patches = [c for c in ax.get_children() if isinstance(c, FancyBboxPatch)]
_texts = [c for c in ax.get_children() if isinstance(c, plt.Text)]
_outs = []
for _p in _patches:
    pb = _p.get_window_extent(_r)
    for _t in _texts:
        if not _t.get_text() or _t.get_text() in ('a', 'b'):
            continue
        tb = _t.get_window_extent(_r)
        cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
        if pb.x0 <= cx <= pb.x1 and pb.y0 <= cy <= pb.y1:      # this text belongs to this box
            slack = 0.6                                            # pixel tolerance
            if tb.x0 < pb.x0 - slack or tb.x1 > pb.x1 + slack or \
               tb.y0 < pb.y0 - slack or tb.y1 > pb.y1 + slack:
                _outs.append((_t.get_text().split(chr(10))[0][:26],
                              round(tb.x1 - pb.x1, 1), round(pb.y0 - tb.y0, 1)))
            break
assert not _outs, 'box text overflow: ' + str(_outs)

# ---------- b centre heterogeneity (percentages with their denominators, not counts) ----------
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(CEN)); w = 0.32
# ★ Definition guard: N = TNM N stage (0-3), ALN = binary axillary nodal status (0/1).
#   The node-positive rate may only be computed as P(N >= 1) or ALN.mean(), never as mean(N)*100.
_np = (a['N'] >= 1).mean() * 100
_al = a.ALN.mean() * 100
assert abs(_np - _al) < 5, f'node-positive by N ({_np:.1f}%) and by ALN ({_al:.1f}%) should agree - the definition may be wrong'
assert a['N'].min() >= 0 and a['N'].max() <= 3, 'N must be the TNM N stage, 0-3'
assert set(a.ALN.dropna().unique()) <= {0, 1}, 'ALN must be binary 0/1'
# Two rates with different meanings (in the first version the third bar, N positive, was clinically
# synonymous with ALN positive and both were exactly 35.1% across the 345 cases)
d = [[a[a.Center == k].DFS.mean() * 100 for k in CEN],
     [(a[a.Center == k]['N'] >= 1).mean() * 100 for k in CEN]]
for j, (vals, col, lab) in enumerate(zip(d, [C_CLIN, C_IMG],
                                         ['DFS event rate', 'Node positive (pN \u2265 1)'])):
    ax.bar(x + (j - 0.5) * w, vals, w, color=col, edgecolor='#000000', linewidth=0.7, label=lab)
    for xi, v_ in zip(x, vals):
        ax.text(xi + (j - 0.5) * w, v_ + 1.8, f'{v_:.1f}%', ha='center', va='bottom', fontsize=6.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{k.split("-")[0]}\n(n = {int((a.Center == k).sum())})' for k in CEN], fontsize=7.4)
ax.set_ylabel('Within-center proportion (%)', fontsize=8.6)
ax.set_ylim(0, 100); ax.tick_params(labelsize=7.6)
ax.legend(fontsize=7.2, loc='upper left', handlelength=1.1, borderpad=0.2, labelspacing=0.35)

# ★ Remove the ambiguity (requested in review): the panel shows rates whose denominators differ by
#   centre, so they must not be summed. The footnote is generated from the data rather than typed by
#   hand, so it cannot drift away from the plotted values.
_note = ('Within-center proportions (denominators differ, n = '
         + ' / '.join(str(int((a.Center == k).sum())) for k in CEN)
         + ') do not sum to the cohort total.\n'
         + 'Counts — DFS events '
         + ' / '.join(str(int(a[a.Center == k].DFS.sum())) for k in CEN)
         + ';  node-positive '
         + ' / '.join(str(int((a[a.Center == k]['N'] >= 1).sum())) for k in CEN) + '.')
ax.text(0.0, _LBL_Y, 'b', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=10, fontweight='bold', color='#000000')

# ★ The footnote is now centred across the whole figure: hung under the right panel alone it left a
#   large empty area (about 1 in) below the flow chart on the left. Its position is computed - take
#   the lowest pixel of the panel b tick labels and put the note 8 px below, so nothing is covered.
_note_t = fig.text(0.5, 0.30, _note, ha='center', va='top',
                   fontsize=6.8, color='#333333', linespacing=1.55)
fig.canvas.draw(); _rr = fig.canvas.get_renderer()
_tick_y0 = min(_tk.get_window_extent(_rr).y0 for _tk in ax.get_xticklabels())
_nh = _note_t.get_window_extent(_rr).height
_note_t.set_position((0.5, (_tick_y0 - 8 - _nh) / fig.bbox.height))
fig.canvas.draw()
_nb = _note_t.get_window_extent(_rr)
for _tk in ax.get_xticklabels():
    _tb = _tk.get_window_extent(_rr)
    assert _nb.y1 <= _tb.y0 + 0.6, (
        f'footnote overlaps the tick labels: note.y1={_nb.y1:.1f} tick.y0={_tb.y0:.1f}')
print(f'  footnote position auto-calibrated (8 px below the tick labels), height {_nh:.1f}px')
save(fig, 'Figure1_flow_and_heterogeneity')
if os.environ.get('FIG_ONLY_F1'):      # for iterating on F1: skip the slow F2/F3 (nested CV)
    print('[only F1 done, exiting on FIG_ONLY_F1]'); raise SystemExit(0)

# ============================================================
# F2 — redundancy (corrected: annotations clear of lines, common axis limits, correct diagonal)
# ============================================================
allr = _rad[_rad['Case_No.'].isin(order)]
wide = {}
# ★ Correction: h[:2] collapsed habitat_L/M/H to the single prefix 'ha' so they overwrote each other,
#   and the scatter resolved to 'Wh_' (whole tumour) instead of the same-habitat comparator, giving
#   rho = 0.650 in the figure and contradicting the 0.999 claimed in the manuscript
for h in ['Whole_tumor', 'habitat_L', 'habitat_M', 'habitat_H']:
    sb = allr[allr.Habitat == h].drop_duplicates('Case_No.').set_index('Case_No.').loc[order, RF]
    pre = {'Whole_tumor': 'Wh', 'habitat_L': 'L', 'habitat_M': 'M', 'habitat_H': 'H'}[h]
    for cc in RF: wide[f'{pre}_{cc}'] = sb[cc].values
W = pd.DataFrame(wide)
# ★ Read the authoritative redundancy results instead of recomputing them, so that figure and
#   manuscript share one definition: the original implementation recomputed them here from only 428
#   DCE conventional features, dropping every feature that contained a NaN (78 of them), which gave
#   12/6/60, whereas the manuscript uses all 856 conventional features and all 84 descriptors (22/6/56).
_red = pd.read_csv(os.path.join(OUT, 'spatial_redundancy_summary_k3.csv'))
_red = _red[_red['max_abs_r'].notna()].copy()
sr = _red.set_index('feature')['max_abs_r'].abs().sort_values(ascending=False)
pt = dict(zip(_red['feature'], _red['partner']))

fig = plt.figure(figsize=(6.9, 2.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.22, 0.90, 0.90, 1.02], wspace=0.50)
ax = fig.add_subplot(gs[0, 0])
cols = [C_CLIN if v >= 0.9 else (C_IMG if v >= 0.8 else C_SPAT) for v in sr.values]
ax.bar(range(len(sr)), sr.values, color=cols, edgecolor='none', width=0.92)
ax.axhline(0.9, color='#000000', ls='--', lw=0.9); ax.axhline(0.8, color='#000000', ls=':', lw=0.9)
ax.set_xlabel('Spatial habitat features (ranked)', fontsize=8.5)
ax.set_ylabel('Maximum |Spearman ρ|\nvs. conventional radiomics', fontsize=8.5)
ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(labelsize=7.5)  # headroom for the legend
_nR = int((sr.values >= 0.90).sum()); _nI = int(((sr.values >= 0.80) & (sr.values < 0.90)).sum())
_nN = int((sr.values < 0.80).sum())
ax.legend(handles=[plt.Line2D([], [], color=C_CLIN, lw=5, label=f'Redundant (|ρ| ≥ 0.90): {_nR}'),
                   plt.Line2D([], [], color=C_IMG, lw=5, label=f'Intermediate (0.80–0.90): {_nI}'),
                   plt.Line2D([], [], color=C_SPAT, lw=5, label=f'Novel (|ρ| < 0.80): {_nN}')],
          fontsize=6.6, loc='upper right', handlelength=1.0, borderpad=0.2, labelspacing=0.3)
ax.set_title('(a) Feature redundancy', fontsize=9, pad=6)

for j, (f1, sfx, ttl) in enumerate([('dce_H_sphericity', 'Sphericity', '(b) Sphericity'),
                                    ('dce_H_surface', 'SurfaceArea', '(c) Surface area')]):
    ax = fig.add_subplot(gs[0, 1 + j])
    xv = spw[f1].values
    key = f'H_original_shape_{sfx}'          # ★ same-habitat (habitat H) comparator, not the whole tumour
    yv = W[key].values
    ax.scatter(xv, yv, s=5, c=C_IMG, alpha=0.55, edgecolors='none')
    rho = spearmanr(xv, yv)[0]
    lo = min(xv.min(), yv.min()); hi = max(xv.max(), yv.max())
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#000000', ls='--', lw=0.9)
    ax.set_xlabel(f'Spatial-derived\n{f1}', fontsize=7.8)
    ax.set_ylabel(f'PyRadiomics\noriginal_shape_{sfx}', fontsize=7.8)
    ax.text(0.05, 0.93, f'ρ = {rho:.3f}', transform=ax.transAxes, fontsize=8, va='top')
    ax.tick_params(labelsize=7.2); ax.set_title(ttl, fontsize=9, pad=6)

# (d) redundancy composition by descriptor family - matches the three right-hand columns of Table 1
ax = fig.add_subplot(gs[0, 3])
_DESCS = ['n_voxels', 'n_components', 'largest_frac', 'euler', 'surface', 'sphericity',
          'fractal_dim', 'centroid_dist', 'radial_mean', 'radial_sd', 'edge_dist_mean',
          'interface_norm', 'interface']
_FAM = {'n_voxels': 'Morphology', 'surface': 'Morphology', 'sphericity': 'Morphology',
        'largest_frac': 'Morphology', 'n_components': 'Topology', 'euler': 'Topology',
        'fractal_dim': 'Topology', 'centroid_dist': 'Spatial distribution',
        'radial_mean': 'Spatial distribution', 'radial_sd': 'Spatial distribution',
        'edge_dist_mean': 'Spatial distribution', 'interface': 'Inter-habitat',
        'interface_norm': 'Inter-habitat'}


def _fam_of(f_):
    """Assign the descriptor family. ★ Pairwise centroid distances (two habitat prefixes) must go to
    "inter-habitat"; matching on the suffix alone, as the original implementation did, counts the six
    pairwise centroid distances as "spatial distribution" by mistake."""
    s_ = f_.split('_', 1)[1] if '_' in f_ else f_
    for d_ in _DESCS:
        if s_.endswith('_' + d_):
            head = s_[:-(len(d_) + 1)]
            if d_ == 'centroid_dist' and len(head.split('_')) == 2:
                return 'Inter-habitat'
            return _FAM[d_]
    return None


_FO = ['Morphology', 'Topology', 'Spatial distribution', 'Inter-habitat']
_TICK = ['Morphology', 'Topology', 'Spatial\ndistribution', 'Inter-\nhabitat']
_cnt = {f_: {'n': 0, 'i': 0, 'r': 0} for f_ in _FO}
for f_, v_ in sr.items():
    _fm = _fam_of(f_)
    if _fm is None:
        continue
    _cnt[_fm]['r' if v_ >= 0.90 else ('i' if v_ >= 0.80 else 'n')] += 1
# Horizontal stacked bars: the four long labels sit on the y axis so they cannot overlap (the
# vertical version rendered as one run-on string in practice)
_xs = np.arange(len(_FO))
_bn = np.array([_cnt[k_]['n'] for k_ in _FO], float)
_bi = np.array([_cnt[k_]['i'] for k_ in _FO], float)
_br = np.array([_cnt[k_]['r'] for k_ in _FO], float)
ax.barh(_xs, _bn, 0.62, color=C_SPAT, edgecolor='#000000', linewidth=0.6)
ax.barh(_xs, _bi, 0.62, left=_bn, color=C_IMG, edgecolor='#000000', linewidth=0.6)
ax.barh(_xs, _br, 0.62, left=_bn + _bi, color=C_CLIN, edgecolor='#000000', linewidth=0.6)
for _xi, _n, _i, _r in zip(_xs, _bn, _bi, _br):
    if _n: ax.text(_n / 2, _xi, '%d' % _n, ha='center', va='center', fontsize=6.6)
    if _i: ax.text(_n + _i / 2, _xi, '%d' % _i, ha='center', va='center', fontsize=6.6)
    if _r: ax.text(_n + _i + _r / 2, _xi, '%d' % _r, ha='center', va='center', fontsize=6.6)
ax.set_yticks(_xs)
ax.set_yticklabels(_TICK, fontsize=6.8)          # two-line labels so they do not eat into the plot area
ax.invert_yaxis()
ax.set_xlabel('Spatial habitat features (n)', fontsize=8.5)
ax.set_xlim(0, 27); ax.tick_params(labelsize=7.2)   # longest bar = 24 (spatial distribution)
ax.set_title('(d) Redundancy by descriptor family', fontsize=9, pad=6)
save(fig, 'Figure2_feature_redundancy')

# ============================================================
# F3 — forest plot (corrected: paired-bootstrap CI of ΔC)
# ============================================================
rows = []
for k in ['Parsim_k1', 'Parsim_k0', 'M0 clinical 11', 'Clin+DCEspat', 'M4c+DCEspat',
          'Clin+P2', 'M4c+P1', 'M4c+P2', 'Spatial_only']:
    dlo, dhi, dd, pp = m.paired_boot(ev, tm, OOF[k], M4C)
    rows.append(dict(model=k, dC=dd, lo=dlo, hi=dhi, p=pp,
                     C=concordance_index_censored(ev, tm, OOF[k])[0]))
fr = pd.DataFrame(rows).sort_values('dC')
LBL = {'Parsim_k1': 'Parsimonious, 4 variables', 'Parsim_k0': 'Clinical only, 3 variables',
       'M0 clinical 11': 'Clinical, 11 variables', 'Clin+DCEspat': 'Clinical + DCE spatial (novel)',
       'M4c+DCEspat': 'M4c + DCE spatial (novel)', 'Clin+P2': 'Clinical + all spatial',
       'M4c+P1': 'M4c + novel spatial (53)', 'M4c+P2': 'M4c + all spatial (81)',
       'Spatial_only': 'Spatial features only'}
PNL = 8.2          # common font size for panel letters and titles
fig = plt.figure(figsize=(6.9, 3.45))
gs = fig.add_gridspec(2, 3, height_ratios=[1.30, 1.0], hspace=0.78, wspace=0.46)

# ---------- (a) forest: ΔC and paired-bootstrap 95% CI for the nine models ----------
ax = fig.add_subplot(gs[0, :])
xmax = 0.125
for i, (_, r) in enumerate(fr.iterrows()):
    col = C_PARS if r.model.startswith('Parsim') else (C_CLIN if r.model == 'M0 clinical 11' else
          (C_SPAT if r.model == 'Spatial_only' else C_IMG))
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.5, solid_capstyle='butt', zorder=2)
    for v_ in (r.lo, r.hi):
        ax.plot(v_, i, marker='|', color=col, ms=5, mew=1.3, zorder=2)
    ax.scatter(r.dC, i, s=36, color=col, edgecolors='#000000', linewidths=0.6, zorder=3)
ax.axvline(0, color='#000000', lw=1.0, ls='--', zorder=1)
ax.set_yticks(range(len(fr))); ax.set_yticklabels([LBL[s] for s in fr.model], fontsize=7.4)
ax.set_xlabel('Δ C-index vs. reference M4c (paired bootstrap, 95% CI)', fontsize=8.2)
ax.set_xlim(-0.26, xmax); ax.tick_params(labelsize=7.4)
ax.set_ylim(-0.7, len(fr) - 0.25)
for i, (_, r) in enumerate(fr.iterrows()):
    ax.text(xmax * 0.985, i, f'p = {r.p:.3f}', fontsize=6.8, va='center', ha='right')
ax.set_title('(a) Incremental value over the reference model', fontsize=PNL, loc='left', pad=4)

# ---------- (b) ΔC per partition: the null result is not an accident of one partition ----------
d2 = pd.read_csv(os.path.join(FIG, 'FigureS2_data.csv'), encoding='utf-8-sig')
_b2 = d2['M4c'].values
CMP2 = [('M4c + spatial all', 'M4c +\nspatial', C_IMG), ('Spatial only', 'Spatial\nonly', C_SPAT),
        ('Parsimonious 4', 'Parsim\n4', C_PARS), ('Clinical 11', 'Clinical\n11', C_CLIN)]
axb = fig.add_subplot(gs[1, 0])
_rng = np.random.default_rng(0)
_bstat = []
for _i, (_c, _l, _col) in enumerate(CMP2):
    dv = d2[_c].values - _b2
    axb.scatter(_rng.normal(_i, 0.072, len(dv)), dv, s=8, color=_col, alpha=0.80,
                edgecolors='none', zorder=3)
    mu, sd = float(dv.mean()), float(dv.std(ddof=1))
    _bstat.append((_c, mu, sd, float(dv.min()), float(dv.max())))
    axb.plot([_i - 0.30, _i + 0.30], [mu, mu], color='#000000', lw=1.5, zorder=4)
    axb.plot([_i, _i], [mu - sd, mu + sd], color='#000000', lw=1.0, zorder=4)
    for _e in (mu - sd, mu + sd):
        axb.plot([_i - 0.13, _i + 0.13], [_e, _e], color='#000000', lw=1.0, zorder=4)
axb.axhline(0, color='#000000', lw=0.9, ls='--', zorder=1)
axb.set_xticks(range(len(CMP2))); axb.set_xticklabels([c[1] for c in CMP2], fontsize=6.6)
axb.set_ylabel('Δ C-index, per partition', fontsize=7.6)
axb.tick_params(labelsize=7.0)
axb.set_title('(b) Across 20 partitions', fontsize=PNL, loc='left', pad=4)

# ---------- (c) K sensitivity: 30 modality x K x pool combinations ----------
axc = fig.add_subplot(gs[1, 1])
kd = pd.read_csv(os.path.join(RES, 'k_spatial_full_delta.csv'), encoding='utf-8-sig')
MCOL3 = {'DWI': '#E67E22', 'DCE': '#3498DB', 'DWI+DCE': '#8E44AD'}
for _ms in ['DWI', 'DCE', 'DWI+DCE']:
    for _p, _mk, _fc in [('novel', 'o', True), ('all', 's', False)]:
        g = kd[(kd.modset == _ms) & (kd.pool == _p)].sort_values('K')
        axc.errorbar(g.K, g.dC, yerr=[g.dC - g.CI_lo, g.CI_hi - g.dC],
                     fmt=_mk, ms=3.4, lw=0.8, capsize=1.4, elinewidth=0.7,
                     color=MCOL3[_ms], markerfacecolor=MCOL3[_ms] if _fc else 'white',
                     markeredgecolor=MCOL3[_ms], markeredgewidth=0.8, zorder=3)
axc.axhline(0, color='#000000', lw=0.9, ls='--', zorder=1)
_sdw = float(pd.read_csv(os.path.join(RES, 'FINAL_stability_4center.csv'),
                         encoding='utf-8-sig').set_index('Model').loc['M4c original paper (DCE_H 107 + clinical 11)', 'sd'])
axc.axhspan(-_sdw, _sdw, color='#B0B0B0', alpha=0.22, zorder=0)
axc.set_xticks([2, 3, 4, 5, 6]); axc.tick_params(labelsize=7.0)
axc.set_xlabel('Number of habitats, K', fontsize=7.6)
axc.set_ylabel('Δ C-index (spatial added)', fontsize=7.6)
axc.set_title('(c) 30 modality × K combinations', fontsize=PNL, loc='left', pad=4)
axc.set_ylim(-0.062, 0.018)
# ★ Plot the headline number of the paper (upper 95% bound of the increment, +0.010) so that the
# figure can be checked against the results on its own
axc.axhline(0.010, color='#E74C3C', lw=0.9, ls=':', zorder=2)
axc.text(0.985, 0.965, 'largest upper 95%\nbound = +0.010', transform=axc.transAxes,
         fontsize=5.5, ha='right', va='top', color='#E74C3C', linespacing=1.25)
# ★ Keep only the three modality colour keys, placed in the empty lower-left area (the former 2x2
#   legend plus grey-band note box overlapped each other at the bottom centre); the meaning of the
#   filled/open markers and of the grey band is given in the caption, to keep in-figure text sparse
_handles = [plt.Line2D([], [], marker='o', ls='none', ms=3.6, color=MCOL3[k_],
                       markeredgecolor=MCOL3[k_], label=k_) for k_ in ['DWI', 'DCE', 'DWI+DCE']]
axc.legend(handles=_handles, fontsize=5.9, loc='lower left', ncol=1, handletextpad=0.25,
           borderpad=0.18, labelspacing=0.22)

# ---------- (d) change of learner: the conclusion does not depend on Cox ----------
axd = fig.add_subplot(gs[1, 2])
sa = pd.read_csv(os.path.join(RES, 'survival_algorithms_summary.csv'), encoding='utf-8-sig')
_add = sa[sa.model == 'M4c+P2 (199)'].set_index('alg')
_lasso = fr[fr.model == 'M4c+P2'].iloc[0]
ALG = [('LASSO-Cox', float(_lasso.dC), float(_lasso.lo), float(_lasso.hi), C_IMG),
       ('RSF', float(_add.loc['RSF', 'dC_oof']), float(_add.loc['RSF', 'dC_lo']),
        float(_add.loc['RSF', 'dC_hi']), '#16A085'),
       ('GBSA', float(_add.loc['GBSA', 'dC_oof']), float(_add.loc['GBSA', 'dC_lo']),
        float(_add.loc['GBSA', 'dC_hi']), '#D35400'),
       ('ExtraTrees', float(_add.loc['ExtraTrees', 'dC_oof']), float(_add.loc['ExtraTrees', 'dC_lo']),
        float(_add.loc['ExtraTrees', 'dC_hi']), '#7F8C8D')]
for i, (nm, dv, lo, hi, col) in enumerate(ALG):
    axd.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle='butt', zorder=2)
    for v_ in (lo, hi):
        axd.plot(v_, i, marker='|', color=col, ms=5, mew=1.3, zorder=2)
    axd.scatter(dv, i, s=34, color=col, edgecolors='#000000', linewidths=0.6, zorder=3)
axd.axvline(0, color='#000000', lw=1.0, ls='--', zorder=1)
axd.set_yticks(range(len(ALG))); axd.set_yticklabels([a[0] for a in ALG], fontsize=7.0)
axd.set_xlabel('Δ C-index, spatial added\n(patient-level 95% CI)', fontsize=7.4)
axd.set_ylim(-0.55, len(ALG) - 0.45); axd.tick_params(labelsize=7.0)
axd.set_title('(d) Across survival learners', fontsize=PNL, loc='left', pad=4)

save(fig, 'Figure3_model_forest')
pd.DataFrame(_bstat, columns=['model', 'dC_mean', 'dC_sd', 'dC_min', 'dC_max']).to_csv(
    os.path.join(FIG, 'Figure3_partition_data.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame([dict(algorithm=a[0], dC=a[1], lo=a[2], hi=a[3]) for a in ALG]).to_csv(
    os.path.join(FIG, 'Figure3_algorithm_data.csv'), index=False, encoding='utf-8-sig')
fr.to_csv(os.path.join(FIG, 'Figure3_data.csv'), index=False, encoding='utf-8-sig')
print('\n[F1-F3 done]')
