import os
# -*- coding: utf-8 -*-
"""
52_figures_supp.py — supplementary figures S1–S6
  S1 univariable forest plot (81 spatial features, HR + 95% CI, ranked by P)
  S2 multi-seed stability box plots (20 seeds)
  S3 calibration plots (predicted vs. KM observed, 36/60 months)
  S4 decision curves (continuous threshold, 36 months)
  S5 EPV plot
  S6 redundancy by feature category
"""
import importlib.util, os, warnings, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
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
order = a['Case_No.']
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
SEEDS20 = [42] + list(range(19))

# ============================================================
# S1 — univariable forest plot
# ============================================================
u = pd.read_csv(os.path.join(RES, 'FINAL_univariate_lifelines_4center.csv'))
u = u.sort_values('P').head(24).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.9, 4.4))
for i, (_, r) in enumerate(u.iterrows()):
    sig = r.FDR < 0.05
    col = C_CLIN if sig else C_IMG
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.3, solid_capstyle='butt')
    ax.scatter(r.HR, i, s=26, color=col, edgecolors='#000000', linewidths=0.55, zorder=3)
ax.axvline(1, color='#000000', lw=1.0, ls='--')
ax.set_yticks(range(len(u)))
ax.set_yticklabels([s.replace('dwi_', 'DWI-').replace('dce_', 'DCE-') for s in u.feature], fontsize=7.2)
ax.set_xscale('log'); ax.set_xlim(0.30, 1.90)
ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.3, 1.6]); ax.set_xticklabels(['0.4', '0.6', '0.8', '1.0', '1.3', '1.6'], fontsize=7.5)
ax.set_xlabel('Hazard ratio per 1 SD (95% CI, log scale)', fontsize=8.5)
for i, (_, r) in enumerate(u.iterrows()):
    ax.text(1.92, i, f'{r.P:.3f}'.rstrip('0').rstrip('.') if r.P >= 0.001 else '<0.001',
            fontsize=6.6, va='center')
ax.text(1.92, len(u) - 0.3, 'P', fontsize=6.8, va='center')
ax.legend(handles=[Line2D([], [], color=C_CLIN, marker='o', ls='none', label='FDR < 0.05 (4 features)'),
                   Line2D([], [], color=C_IMG, marker='o', ls='none', label='FDR ≥ 0.05')],
          fontsize=6.8, loc='lower right', handlelength=0.9, borderpad=0.2)
save(fig, 'FigureS1_univariate_forest')

# ============================================================
# S2 — multi-seed stability box plots
# ============================================================
X_sets = {
    'Clinical 11': (C11, 'lasso'),
    'M4c': (pd.concat([C11, radH.add_prefix('DCEH_')], axis=1), 'lasso'),
    'M4c + spatial all': (pd.concat([C11, radH.add_prefix('DCEH_'), spw[P2]], axis=1), 'lasso'),
    'Spatial only': (pd.concat([spw[P2]], axis=1), 'lasso'),
    'Parsimonious 4': (None, 'ins'),
}
res2 = {}
for nm, (XX, kind) in X_sets.items():
    vals = []
    for sd in SEEDS20:
        if kind == 'lasso':
            o = m.nested_oof(XX.fillna(0).astype(float), y, seed=sd)
        else:
            o = np.full(len(a), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=sd).split(C3.values):
                Mtr, Mte = C3.iloc[tr].copy(), C3.iloc[te].copy(); cols = list(C3.columns)
                best = None; bv = 0.5; bs = 1
                for c in POOL.columns:
                    v = POOL.iloc[tr][c].values
                    if not np.isfinite(v).all() or np.std(v, ddof=1) == 0: continue
                    cc = concordance_index_censored(y[tr]['event'], y[tr]['time'], v)[0]
                    if abs(cc - 0.5) > bv - 0.5: best, bv, bs = c, cc, (1 if cc >= 0.5 else -1)
                Mtr[best] = bs * POOL.iloc[tr][best].values; Mte[best] = bs * POOL.iloc[te][best].values
                cols.append(best)
                s = StandardScaler()
                o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(s.fit_transform(Mtr[cols].values), y[tr]).predict(s.transform(Mte[cols].values))
        vals.append(concordance_index_censored(ev, tm, o)[0])
    res2[nm] = vals
    print(f'  S2 {nm}: {np.mean(vals):.4f}+-{np.std(vals,ddof=1):.4f}')

fig, ax = plt.subplots(figsize=(4.6, 2.8))
names = list(res2.keys())
cols5 = [C_CLIN, C_IMG, C_IMG, C_SPAT, C_PARS]
bp = ax.boxplot([res2[n] for n in names], positions=range(len(names)), widths=0.55,
                patch_artist=True, showfliers=False, medianprops=dict(color='#000000', lw=1.1),
                whiskerprops=dict(color='#000000', lw=0.9), capprops=dict(color='#000000', lw=0.9))
for patch, c in zip(bp['boxes'], cols5):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor('#000000'); patch.set_linewidth(0.8)
for i, n in enumerate(names):
    ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(res2[n])), res2[n],
               s=6, color='#000000', alpha=0.45, zorder=3, edgecolors='none')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(['Clinical\n11', 'M4c\n118', 'M4c +\nspatial', 'Spatial\nonly', 'Parsim\n4'], fontsize=7.2)
ax.set_ylabel('C-index (20 random partitions)', fontsize=8)
ax.tick_params(labelsize=7.5)
save(fig, 'FigureS2_multiseed_stability')
pd.DataFrame(res2).to_csv(os.path.join(FIG, 'FigureS2_data.csv'), index=False, encoding='utf-8-sig')

# ============================================================
# S3/S4 — calibration plots + decision curves
# ============================================================
def screen_k1(pool_tr, ytr):
    best, bv, bs = None, 0.5, 1
    for c in pool_tr.columns:
        v = pool_tr[c].values
        if not np.isfinite(v).all() or np.std(v, ddof=1) == 0: continue
        cc = concordance_index_censored(ytr['event'], ytr['time'], v)[0]
        if abs(cc - 0.5) > bv - 0.5: best, bv, bs = c, cc, (1 if cc >= 0.5 else -1)
    return best, bs

def run3(kind):
    lp = np.full(len(a), np.nan); s36 = np.full(len(a), np.nan); s60 = np.full(len(a), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(C11.values):
        if kind == 'M0': Mtr, Mte, cols = C11.iloc[tr].copy(), C11.iloc[te].copy(), CL
        elif kind == 'M4c':
            Mtr = pd.concat([C11.iloc[tr].reset_index(drop=True), radH.iloc[tr].reset_index(drop=True)], axis=1)
            Mte = pd.concat([C11.iloc[te].reset_index(drop=True), radH.iloc[te].reset_index(drop=True)], axis=1)
            cols = list(Mtr.columns)
        else:
            Mtr, Mte = C3.iloc[tr].copy(), C3.iloc[te].copy(); cols = list(C3.columns)
            c, bs = screen_k1(POOL.iloc[tr], y[tr])
            Mtr[c] = bs * POOL.iloc[tr][c].values; Mte[c] = bs * POOL.iloc[te][c].values; cols.append(c)
        s = StandardScaler(); A = s.fit_transform(Mtr[cols].values); B = s.transform(Mte[cols].values)
        if kind == 'M4c':
            _, ba, _ = m.run_lasso_cv(Mtr[cols], y[tr])
            cp = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=20000, fit_baseline_model=True).fit(A, y[tr])
            lp[te] = np.asarray(cp.predict(B)).ravel()
        else:
            cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr]); lp[te] = cp.predict(B)
        sf = cp.predict_survival_function(B, return_array=True); tt = cp.unique_times_
        for t, arr in [(36, s36), (60, s60)]:
            arr[te] = sf[:, max(int(np.searchsorted(tt, t, side='right') - 1), 0)]
    return lp, s36, s60

R3 = {k: run3(k) for k in ['M0', 'M4c', 'P4']}
LBL3 = {'M0': ('Clinical 11', C_CLIN), 'M4c': ('M4c 118', C_IMG), 'P4': ('Parsimonious 4', C_PARS)}

fig = plt.figure(figsize=(6.9, 2.6))
gs = fig.add_gridspec(1, 2, wspace=0.30)
for j, t in enumerate([36, 60]):
    ax = fig.add_subplot(gs[0, j])
    ax.plot([0, 1], [0, 1], color='#000000', ls='--', lw=0.9)
    for k, (nm, col) in LBL3.items():
        s_ = R4 = None
        s_ = R3[k][1] if t == 36 else R3[k][2]
        lp = R3[k][0]
        qs = pd.qcut(pd.Series(lp).rank(method='first'), 5, labels=False)
        px, ob, err = [], [], []
        for b in range(5):
            idx = np.where(qs == b)[0]
            if not len(idx): continue
            pred = 1 - np.nanmean(s_[idx])
            kmf = KaplanMeierFitter().fit(tm[idx], ev[idx])
            obs = 1 - float(kmf.predict(t))
            px.append(pred); ob.append(obs)
            err.append(float(kmf.predict(t)) * 0)
        ax.plot(px, ob, marker='o', ms=3.6, color=col, lw=1.3, label=nm)
    ax.set_xlabel(f'Predicted risk at {t} mo', fontsize=8)
    ax.set_ylabel(f'Observed risk at {t} mo (KM)', fontsize=8)
    ax.set_xlim(0, 0.55); ax.set_ylim(0, 0.45); ax.tick_params(labelsize=7.2)
    if j == 0: ax.legend(fontsize=6.8, loc='upper left', handlelength=1.0, borderpad=0.2)
    ax.set_title(f'{t} months', fontsize=8.5, pad=5)
# ★ Retired: the content was merged into the main figure, no separate output any more (was FigureS3_calibration)
# save(fig, 'FigureS3_calibration')

km_c = KaplanMeierFitter().fit(tm, 1 - ev.astype(int))
Gfun = lambda s: max(float(km_c.predict(s)), 1e-6)
def nb_curve(risk, t, pts):
    out = []
    for pt in pts:
        tp = fp = 0.0; n = len(risk); Gt = Gfun(t)
        for i in range(n):
            if tm[i] <= t and ev[i]:
                if risk[i] >= pt: tp += 1.0 / Gfun(tm[i])
            elif tm[i] > t:
                if risk[i] >= pt: fp += 1.0 / Gt
        out.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(out)

pix = np.arange(0.02, 0.42, 0.01)
km_all = KaplanMeierFitter().fit(tm, ev)
prev = 1 - float(km_all.predict(36))
fig, ax = plt.subplots(figsize=(3.35, 2.8))
for k, (nm, col) in LBL3.items():
    rk = 1 - R3[k][1]
    ax.plot(pix, nb_curve(np.where(np.isfinite(rk), rk, 0), 36, pix), color=col, lw=1.5, label=nm)
ta = prev - (1 - prev) * (pix / (1 - pix))
ax.plot(pix, ta, color='#000000', lw=1.0, ls='--', label='Treat all')
ax.axhline(0, color='#000000', lw=1.0, ls=':', label='Treat none')
ax.set_xlabel('Threshold probability', fontsize=8.5); ax.set_ylabel('Net benefit', fontsize=8.5)
ax.set_xlim(0.02, 0.41); ax.set_ylim(-0.22, 0.16); ax.tick_params(labelsize=7.5)
ax.legend(fontsize=6.8, loc='upper right', handlelength=1.2, borderpad=0.2)
# ★ Retired: the content was merged into the main figure, no separate output any more (was FigureS4_dca)
# save(fig, 'FigureS4_dca')

# ============================================================
# S5 — EPV
# ============================================================
epv = [('Clinical 3', 3), ('Parsimonious 4', 4), ('Clinical 11', 11), ('Clinical + DCE spatial', 35),
       ('M4c', 118), ('M4c + spatial all', 199)]
names5 = [e[0] for e in epv]; vals5 = [ev.sum() / e[1] for e in epv]
fig, ax = plt.subplots(figsize=(3.35, 2.6))
cols5 = [C_PARS, C_PARS, C_CLIN, C_IMG, C_IMG, C_IMG]
ax.bar(range(len(epv)), vals5, color=cols5, edgecolor='#000000', linewidth=0.7, width=0.62)
ax.axhline(10, color='#000000', ls='--', lw=1.0)
ax.text(len(epv) - 0.4, 10.6, 'EPV = 10', fontsize=7, ha='right')
ax.set_yscale('log'); ax.set_ylim(0.2, 60)
ax.set_xticks(range(len(epv)))
ax.set_xticklabels(['Clinical\n3', 'Parsim\n4', 'Clinical\n11', 'Clin+\nspatial', 'M4c\n118', 'M4c+\nspatial'], fontsize=6.8)
ax.set_ylabel('Events per variable (EPV)', fontsize=8)
for i, v_ in enumerate(vals5):
    ax.text(i, v_ * 1.15, f'{v_:.1f}', ha='center', fontsize=6.8)
ax.tick_params(labelsize=7.2)
# ★ Retired: the content was merged into the main figure, no separate output any more (was FigureS5_EPV)
# save(fig, 'FigureS5_EPV')

# ============================================================
# F4 — parsimony quintet (C-index / Brier@60 / AUC@60 / full decision curve / EPV)
#   ★ The former supplementary figures S4 (decision curve) and S5 (EPV) are merged into the main
#     figure; S4/S5 are then retired from the supplementary set
# ============================================================
# ★ Data source: the single authoritative definition produced by 134_figure4_authoritative.py
#   (C matches Table 4; AUC uses the training fold as the censoring reference; includes the
#   non-leaky 4-variable model). The old Figure4_data.csv used a different definition and is deprecated.
M4 = pd.read_csv(os.path.join(FIG, 'Figure4_authoritative.csv'), index_col=0)
_order4 = ['P4', 'M0', 'M4c']; _cols4 = [C_PARS, C_CLIN, C_IMG]
fig = plt.figure(figsize=(6.9, 3.78))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.10], hspace=0.84, wspace=0.42)
for _j, (_mk, _sk, _ttl) in enumerate([('C', 'Csd', '(a) C-index'),
                                       ('B60', 'B60sd', '(b) Brier score at 60 mo\n(lower is better)'),
                                       ('A60', 'A60sd', '(c) AUC at 60 mo')]):
    ax = fig.add_subplot(gs[0, _j])
    _v = [M4.loc[k_, _mk] for k_ in _order4]; _e = [M4.loc[k_, _sk] for k_ in _order4]
    # ★ Points with error bars rather than bars: the C/AUC/Brier of the three models differ
    #   only slightly, so the y axis has to be zoomed; a bar chart on a zoomed axis encodes the
    #   value in the bar length and therefore exaggerates the difference (a reviewer-facing flaw).
    #   A point plot encodes the value as position, which stays legitimate on a zoomed axis.
    for _i, (_u, _w, _c) in enumerate(zip(_v, _e, _cols4)):
        ax.errorbar([_i], [_u], yerr=[_w], fmt='o', ms=7.0, lw=1.4, capsize=3.0, capthick=1.1,
                    color=_c, markeredgecolor='#000000', markeredgewidth=0.7, zorder=3)
    _lo = min(u - w for u, w in zip(_v, _e)); _hi = max(u + w for u, w in zip(_v, _e)); _sp = _hi - _lo
    ax.set_ylim(_lo - _sp * 0.34, _hi + _sp * 0.44)
    for _i, (u, w) in enumerate(zip(_v, _e)):
        ax.text(_i, u + w + _sp * 0.06, f'{u:.3f}', ha='center', va='bottom', fontsize=6.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(['4', '11', '118'], fontsize=7.6)
    ax.set_xlim(-0.45, 2.45)
    ax.set_title(_ttl, fontsize=8.2, loc='left', pad=5); ax.tick_params(labelsize=7.0)
    ax.set_xlabel('Variables', fontsize=7.2, labelpad=1.2)
    if _j == 0: ax.set_ylabel('Value', fontsize=7.8)

# (d) full decision curve (36 months) — with the treat-all / treat-none reference lines
axd = fig.add_subplot(gs[1, :2])
for _k, (_nm, _col) in LBL3.items():
    _rk = 1 - R3[_k][1]
    axd.plot(pix, nb_curve(np.where(np.isfinite(_rk), _rk, 0), 36, pix), color=_col, lw=1.5,
             label=_nm, zorder=3)
axd.plot(pix, prev - (1 - prev) * (pix / (1 - pix)), color='#000000', lw=1.0, ls='--',
         label='Treat all', zorder=2)
axd.axhline(0, color='#000000', lw=1.0, ls=':', label='Treat none', zorder=2)
axd.set_xlim(0.02, 0.41); axd.set_ylim(-0.22, 0.17)
axd.set_xlabel('Threshold probability for 36-month DFS', fontsize=7.8)
axd.set_ylabel('Net benefit', fontsize=7.8); axd.tick_params(labelsize=7.2)
axd.legend(fontsize=6.4, loc='upper right', ncol=2, handlelength=1.3, borderpad=0.22,
           labelspacing=0.28, columnspacing=0.9)
axd.set_title('(d) Decision curve analysis', fontsize=8.2, loc='left', pad=5)

# (e) EPV — the 118-variable reference model has an EPV of only 0.47, so the incremental test
# must be moved to a reference model with ample EPV
axe = fig.add_subplot(gs[1, 2])
# ★ Points as in (a)(b)(c): on a log axis the bar length does not represent the value
#   (the bar height encodes the log value); position coding in a point plot is the correct form
#   on a log axis
for _i, (_vv, _cc) in enumerate(zip(vals5, cols5)):
    axe.plot([_i], [_vv], marker='o', ms=6.2, color=_cc,
             markeredgecolor='#000000', markeredgewidth=0.7, zorder=3)
axe.axhline(10, color='#000000', ls='--', lw=1.0, zorder=1)
axe.text(len(epv) - 0.35, 10.8, 'EPV = 10', fontsize=6.4, ha='right')
axe.set_yscale('log'); axe.set_ylim(0.2, 90)
axe.set_xticks(range(len(epv)))
axe.set_xticklabels(['Clinical 3', 'Parsim 4', 'Clinical 11', 'Clin+spat', 'M4c 118', 'M4c+spat'],
                    fontsize=5.9, rotation=90)      # ★ vertical labels: six horizontal categories would certainly overlap in a 1.3 in panel
axe.set_xlim(-0.55, len(epv) - 0.45)
axe.set_ylabel('Events per variable', fontsize=7.6)
for _i, _vv in enumerate(vals5):
    axe.text(_i, _vv * 1.18, f'{_vv:.1f}', ha='center', fontsize=6.3)
axe.tick_params(labelsize=6.8)
axe.set_title('(e) Events per variable', fontsize=8.2, loc='left', pad=5)
save(fig, 'Figure4_parsimony_quartet')
pd.DataFrame({'model': ['4 variables', '11 variables', '118 variables'],
              'C': [M4.loc[k_, 'C'] for k_ in _order4],
              'Brier60': [M4.loc[k_, 'B60'] for k_ in _order4],
              'AUC60': [M4.loc[k_, 'A60'] for k_ in _order4]}).to_csv(
    os.path.join(FIG, 'Figure4_fivefold_data.csv'), index=False, encoding='utf-8-sig')

# ============================================================
# S6 — redundancy by feature category
# ============================================================
def cat(c):
    b = re.sub(r'^(dwi|dce)_', '', c)
    if re.match(r'^[LMH]_[LMH]_', b): return 'Inter-habitat'
    base = b[2:] if re.match(r'^[LMH]_', b) else b
    if base in ('n_voxels', 'surface', 'sphericity', 'fractal_dim'): return 'Morphological'
    if base in ('n_components', 'largest_frac', 'euler'): return 'Topological'
    return 'Spatial distribution'
red['cat'] = red.feature.map(cat)
order6 = ['Morphological', 'Topological', 'Spatial distribution', 'Inter-habitat']
tab = pd.DataFrame({v: [(red[(red.cat == v)].max_abs_r >= 0.9).sum(),
                        ((red[(red.cat == v)].max_abs_r >= 0.8) & (red[(red.cat == v)].max_abs_r < 0.9)).sum(),
                        (red[(red.cat == v)].max_abs_r < 0.8).sum()] for v in order6},
                   index=['Redundant', 'Intermediate', 'Novel']).T
fig, ax = plt.subplots(figsize=(3.6, 2.6))
x6 = np.arange(len(order6)); w6 = 0.26
for j, (row, col) in enumerate(zip(['Novel', 'Intermediate', 'Redundant'], [C_SPAT, C_IMG, C_CLIN])):
    ax.bar(x6 + (j - 1) * w6, tab[row].values, w6, color=col, edgecolor='#000000', linewidth=0.7, label=row)
    for xi, v_ in zip(x6, tab[row].values):
        if v_ > 0: ax.text(xi + (j - 1) * w6, v_ + 0.6, f'{int(v_)}', ha='center', fontsize=6.6)
ax.set_xticks(x6)
ax.set_xticklabels(['Morpho-\nlogical', 'Topo-\nlogical', 'Spatial\ndistribution', 'Inter-\nhabitat'], fontsize=6.8)
ax.set_ylabel('Number of features', fontsize=8); ax.set_ylim(0, tab.values.max() * 1.28)
ax.legend(fontsize=6.8, loc='upper right', handlelength=0.9, borderpad=0.2)
ax.tick_params(labelsize=7.2)
# ★ Retired: the content was merged into the main figure, no separate output any more (was FigureS6_redundancy_by_category)
# save(fig, 'FigureS6_redundancy_by_category')
tab.to_csv(os.path.join(FIG, 'FigureS6_data.csv'), encoding='utf-8-sig')
print('\n[S1-S6 done]')


# ============================================================
# F6 — risk-stratified KM + calibration curves (merged from the former supplementary figure S3;
#   S3 is then retired from the supplementary set)
#   It lives in 52 because the OOF survival functions R3 needed for calibration are already computed here
# ============================================================
_lp4 = np.asarray(R3['P4'][0]).ravel()
_med = np.median(_lp4); _hi_g, _lo_g = _lp4 >= _med, _lp4 < _med
fig = plt.figure(figsize=(6.9, 2.92))
gs6 = fig.add_gridspec(1, 3, wspace=0.36)

ax = fig.add_subplot(gs6[0, 0])
for _m, _col, _lab in [(_lo_g, C_SPAT, 'Low risk'), (_hi_g, C_CLIN, 'High risk')]:
    _kmf = KaplanMeierFitter().fit(tm[_m], ev[_m])
    ax.plot(_kmf.survival_function_.index, _kmf.survival_function_.iloc[:, 0],
            color=_col, lw=1.6, label=_lab)
    _ci = _kmf.confidence_interval_
    ax.fill_between(_ci.index, _ci.iloc[:, 0], _ci.iloc[:, 1], color=_col, alpha=0.13, lw=0)
_lr = logrank_test(tm[_lo_g], tm[_hi_g], ev[_lo_g], ev[_hi_g])
_ps = _pf(_lr.p_value)
ax.text(0.04, 0.06, 'Log-rank $P$ ' + ('< 0.001' if _ps == '<0.001' else '= ' + _ps),
        transform=ax.transAxes, fontsize=7.6)
ax.text(0.04, 0.155, f'n = {int(_lo_g.sum())} / {int(_hi_g.sum())}', transform=ax.transAxes, fontsize=7.3)
ax.set_xlabel('Time (months)', fontsize=7.8); ax.set_ylabel('Disease-free survival', fontsize=7.8)
ax.set_ylim(0.60, 1.02); ax.set_xlim(0, 84); ax.tick_params(labelsize=7.2)
ax.legend(fontsize=6.9, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=False, handlelength=1.2)
ax.set_title('(a) Risk groups, 4-variable model', fontsize=8.2, loc='left', pad=26)

for _j, _t in enumerate([36, 60]):
    axc = fig.add_subplot(gs6[0, 1 + _j])
    axc.plot([0, 1], [0, 1], color='#000000', ls='--', lw=0.9, zorder=1)
    for _k, (_nm, _col) in LBL3.items():
        _sv = R3[_k][1] if _t == 36 else R3[_k][2]
        _lp = R3[_k][0]
        _qs = pd.qcut(pd.Series(_lp).rank(method='first'), 5, labels=False)
        _px, _ob = [], []
        for _b in range(5):
            _idx = np.where(_qs == _b)[0]
            if not len(_idx): continue
            _px.append(1 - np.nanmean(_sv[_idx]))
            _ob.append(1 - float(KaplanMeierFitter().fit(tm[_idx], ev[_idx]).predict(_t)))
        axc.plot(_px, _ob, marker='o', ms=3.4, color=_col, lw=1.3, label=_nm, zorder=3)
    axc.set_xlabel(f'Predicted risk at {_t} mo', fontsize=7.8)
    axc.set_ylabel(f'Observed risk at {_t} mo (KM)', fontsize=7.8)
    axc.set_xlim(0, 0.55); axc.set_ylim(0, 0.45); axc.tick_params(labelsize=7.2)
    if _j == 0:
        axc.legend(fontsize=6.4, loc='upper left', handlelength=1.0, borderpad=0.2, labelspacing=0.22)
    axc.set_title(f'({"bc"[_j]}) Calibration, {_t} months', fontsize=8.2, loc='left', pad=26)
save(fig, 'Figure6_km_risk_groups')
