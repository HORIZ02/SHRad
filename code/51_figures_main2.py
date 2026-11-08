import os
# -*- coding: utf-8 -*-
"""
51_figures_main2.py — main figures F4 / F5 / F6
  F4 parsimony quartet (C-index / Brier@60 / AUC@60 / DCA@0.20), one common definition over 20 seeds
  F5 centre effect (within-centre vs. cross-centre; pooled vs. within-centre)
  F6 risk-stratified KM (4-variable model, median split of the OOF risk)
"""
import importlib.util, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
rcParams['font.family'] = 'serif'; rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 9
for k in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']: rcParams[k] = 'black'
rcParams['axes.linewidth'] = 0.9; rcParams['xtick.major.width'] = 0.9; rcParams['ytick.major.width'] = 0.9
rcParams['axes.spines.top'] = False; rcParams['axes.spines.right'] = False
rcParams['savefig.facecolor'] = 'white'; rcParams['figure.facecolor'] = 'white'
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
def _pf(p):
    """Journal P-value format: <0.001, never 0.000."""
    p = float(p)
    return '<0.001' if p < 0.001 else format(p, '.3f')


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

a = pd.read_csv(os.path.join(m.BASE, 'analysis', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
y = m.make_surv(a); ev, tm = y['event'].astype(bool), y['time'].astype(float)
order = a['Case_No.']; CEN = sorted(a.Center.unique())
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
SEEDS20 = [42] + list(range(19))
TIMES = [36, 60]

def screen_topk(pool_tr, ytr, k):
    sc = []
    for c in pool_tr.columns:
        v = pool_tr[c].values
        if not np.isfinite(v).all() or np.std(v, ddof=1) == 0: continue
        cc = concordance_index_censored(ytr['event'], ytr['time'], v)[0]
        sc.append((c, abs(cc - 0.5), 1 if cc >= 0.5 else -1))
    sc.sort(key=lambda t: -t[1]); return [(c, s) for c, _, s in sc[:k]]

def run_model(kind, seed):
    """Return (oof_lp, S36, S60)."""
    lp = np.full(len(a), np.nan); s36 = np.full(len(a), np.nan); s60 = np.full(len(a), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(C11.values):
        if kind == 'M0':
            Mtr, Mte, cols = C11.iloc[tr].copy(), C11.iloc[te].copy(), CL
        elif kind == 'M4c':
            Mtr = pd.concat([C11.iloc[tr].reset_index(drop=True), radH.iloc[tr].reset_index(drop=True)], axis=1)
            Mte = pd.concat([C11.iloc[te].reset_index(drop=True), radH.iloc[te].reset_index(drop=True)], axis=1)
            cols = list(Mtr.columns)
        else:
            Mtr, Mte = C3.iloc[tr].copy(), C3.iloc[te].copy(); cols = list(C3.columns)
            for c, sgn in screen_topk(POOL.iloc[tr], y[tr], 1):
                Mtr[c] = sgn * POOL.iloc[tr][c].values; Mte[c] = sgn * POOL.iloc[te][c].values; cols.append(c)
        sc = StandardScaler(); A = sc.fit_transform(Mtr[cols].values); B = sc.transform(Mte[cols].values)
        if kind == 'M4c':
            _, ba, _ = m.run_lasso_cv(Mtr[cols], y[tr])
            cp = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=20000, fit_baseline_model=True).fit(A, y[tr])
            lp[te] = np.asarray(cp.predict(B)).ravel()
        else:
            cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr]); lp[te] = cp.predict(B)
        sf = cp.predict_survival_function(B, return_array=True); tt = cp.unique_times_
        for t, arr in [(36, s36), (60, s60)]:
            j = max(int(np.searchsorted(tt, t, side='right') - 1), 0)
            arr[te] = sf[:, j]
    return lp, s36, s60

KINDS = {'M0': 'Clinical, 11 variables', 'M4c': 'M4c, 118 variables', 'P4': 'Parsimonious, 4 variables'}
STORE = {}
for kd in KINDS:
    L_, S36, S60 = [], [], []
    for sd in SEEDS20:
        l_, s36, s60 = run_model(kd, sd)
        L_.append(l_); S36.append(s36); S60.append(s60)
    STORE[kd] = (L_, S36, S60)
    print(f'  {kd} 20 seeds done')

km_c = KaplanMeierFitter().fit(tm, 1 - ev.astype(int))
Gfun = lambda s: max(float(km_c.predict(s)), 1e-6)
def nb(risk, t, pt):
    tp = fp = 0.0; n = len(risk); Gt = Gfun(t)
    for i in range(n):
        if tm[i] <= t and ev[i]:
            if risk[i] >= pt: tp += 1.0 / Gfun(tm[i])
        elif tm[i] > t:
            if risk[i] >= pt: fp += 1.0 / Gt
    return tp / n - fp / n * (pt / (1 - pt))

MET = {}
for kd, (L_, S36, S60) in STORE.items():
    cc, b36, b60, a36, a60, dd = [], [], [], [], [], []
    for i in range(len(SEEDS20)):
        lp = L_[i]; s36 = S36[i]; s60 = S60[i]
        cc.append(concordance_index_censored(ev, tm, lp)[0])
        for t, s_, acc in [(36, s36, b36), (60, s60, b60)]:
            ok = np.isfinite(s_)
            try: acc.append(brier_score(y[ok], y[ok], s_[ok], [t])[1][0])
            except Exception: pass
        try:
            au, _ = cumulative_dynamic_auc(y, y, lp, TIMES); a36.append(au[0]); a60.append(au[1])
        except Exception: pass
        dd.append(nb(1 - s36, 36, 0.20))
    MET[kd] = dict(C=np.mean(cc), Csd=np.std(cc, ddof=1), B36=np.mean(b36), B36sd=np.std(b36, ddof=1),
                   B60=np.mean(b60), B60sd=np.std(b60, ddof=1), A36=np.mean(a36), A36sd=np.std(a36, ddof=1),
                   A60=np.mean(a60), A60sd=np.std(a60, ddof=1), D=np.mean(dd), Dsd=np.std(dd, ddof=1))
    print(f"  {kd}: C={MET[kd]['C']:.4f} B60={MET[kd]['B60']:.4f} A60={MET[kd]['A60']:.4f} DCA={MET[kd]['D']:.4f}")

# ============================================================
# ★ Definition alignment (important): the LASSO path of this script's own run_model is
#   not identical to 23b.nested_oof, so the C of M0/M4c would disagree with
#   Table 4 / Figure4_authoritative.csv (this is what split them historically).
#   -> Always take the output of 134_figure4_authoritative.py as the reference, overwrite
#      the MET computed here and print the comparison; warn explicitly if a model's C
#      deviates by more than 0.002.
# ============================================================
_AUT = os.path.join(FIG, 'Figure4_authoritative.csv')
if os.path.exists(_AUT):
    _au = pd.read_csv(_AUT, encoding='utf-8-sig').set_index('model')
    print('\n-- definition alignment: Figure4_authoritative.csv is authoritative --')
    for _k in list(MET.keys()):
        if _k not in _au.index:
            print(f'  [!] {_k} is not in the authoritative table, keeping the value computed here')
            continue
        _d = abs(MET[_k]['C'] - float(_au.loc[_k, 'C']))
        print(f"  {_k}: local C={MET[_k]['C']:.4f} -> authoritative C={float(_au.loc[_k,'C']):.4f}"
              + (f"  [!] |Δ|={_d:.4f} >0.002 overwritten" if _d > 0.002 else '  ok'))
        MET[_k] = {c: float(_au.loc[_k, c]) for c in _au.columns}
    pd.DataFrame(MET).T.to_csv(os.path.join(FIG, 'Figure4_data.csv'), encoding='utf-8-sig')
    print('  ok Figure4_data.csv rewritten with the authoritative values')
else:
    print(f'\n[!] {_AUT} not found; F4 uses the values computed here (run 134_figure4_authoritative.py first)')

# ============================================================
# F4 — parsimony quartet
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.5))
order = ['P4', 'M0', 'M4c']
cols = [C_PARS, C_CLIN, C_IMG]
panels = [('C', 'Csd', 'C-index', False), ('B60', 'B60sd', 'Brier score at 60 mo', True),
          ('A60', 'A60sd', 'AUC at 60 mo', False), ('D', 'Dsd', 'Net benefit (36 mo, pt = 0.20)', False)]
for ax, (mkey, skeys, title, lower) in zip(axes, panels):
    vals = [MET[k][mkey] for k in order]; errs = [MET[k][skeys] for k in order]
    bars = ax.bar(range(3), vals, yerr=errs, color=cols, edgecolor='#000000', linewidth=0.7,
                  capsize=2.5, error_kw=dict(elinewidth=0.8, ecolor='#000000'), width=0.62)
    for i, (v_, e_) in enumerate(zip(vals, errs)):
        ax.text(i, v_ + e_ + (max(vals) * 0.035), f'{v_:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['4\nvariables', '11\nvariables', '118\nvariables'], fontsize=7.5)
    ax.set_title(title, fontsize=8.2, pad=6); ax.tick_params(labelsize=7.2)
    lo = min(v_ - e_ for v_, e_ in zip(vals, errs)); hi = max(v_ + e_ for v_, e_ in zip(vals, errs))
    rng = hi - lo
    ax.set_ylim(lo - rng * (0.55 if lower else 0.28), hi + rng * 0.32)
    if mkey == 'C': ax.set_ylabel('Value', fontsize=8)
    if mkey == 'B60': ax.invert_yaxis()
fig.subplots_adjust(wspace=0.42)
save(fig, 'Figure4_parsimony_quartet')
pd.DataFrame(MET).T.to_csv(os.path.join(FIG, 'Figure4_data.csv'), encoding='utf-8-sig')

# ============================================================
# F5 is rendered by 53_figures_fix45.py (three-panel version, including panel (c) centre separability);
# it is not rendered here so that the older two-panel version cannot overwrite the correct one from 53

# F6 has moved to 52_figures_supp.py (three panels: KM + 36/60-month calibration curves, merged from the former S3)
print('\n[F4-F6 done]')
