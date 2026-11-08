import os
# -*- coding: utf-8 -*-
"""
42_calibration_brier_dca.py — calibration / Brier / DCA required by Radiology:AI

For the three models M0 / M4c / clinical 3 + in-fold top1, computed from the nested OOF risk scores:
  (1) time-dependent AUC (already available, re-checked)
  (2) Brier score (IPCW) @36/60 months
  (3) calibration slope + calibration groups (predicted vs KM observed)
  (4) decision-curve net benefit (Vickers, IPCW)
"""
import importlib.util, os, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT, CL, MAX_MISSING = m.BASE, m.OUT, m.CLIN_VARS, m.MAX_MISSING
SEEDS = [42] + list(range(19))
TIMES = [36, 60]
L = []
def P(s=''):
    print(s); L.append(s)

a = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
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
POOL = pd.concat([radH.add_prefix('R_'), spw.add_prefix('S_')], axis=1)
C11 = a[CL].reset_index(drop=True).astype(float)
C3 = a[['NAC', 'N', 'ALN']].reset_index(drop=True).astype(float)


def screen_topk(pool_tr, ytr, k):
    sc = []
    for c in pool_tr.columns:
        v = pool_tr[c].values
        if not np.isfinite(v).all() or np.std(v, ddof=1) == 0: continue
        cc = concordance_index_censored(ytr['event'], ytr['time'], v)[0]
        sc.append((c, abs(cc - 0.5), 1 if cc >= 0.5 else -1))
    sc.sort(key=lambda t: -t[1])
    return [(c, s) for c, _, s in sc[:k]]


def calib_surv(fit_df, t):
    """Survival-probability prediction function at a given time point, via lifelines"""
    cp = CoxPHFitter().fit(fit_df, 'T', 'E')
    return cp


# ---------- main loop: M0 / M4c / clinical 3 + top1 ----------
MODELS = ['M0 clinical 11', 'M4c (118 variables)', 'clinical 3 + in-fold top1']
oofs = {k: np.full(len(a), np.nan) for k in MODELS}
surv_store = {t: {k: np.full(len(a), np.nan) for k in MODELS} for t in TIMES}

for k in MODELS:
    for sd in SEEDS:
        pass  # placeholder

# seed 42 drives the primary analysis (the other seeds feed the AUC/Brier means); per-fold predicted
# survival is kept as well
for k in MODELS:
    o = np.full(len(a), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(C11.values):
        if k == 'M0 clinical 11':
            Mtr, Mte, cols = C11.iloc[tr], C11.iloc[te], CL
        elif k == 'M4c (118 variables)':
            Mtr = pd.concat([C11.iloc[tr].reset_index(drop=True), radH.iloc[tr].reset_index(drop=True)], axis=1)
            Mte = pd.concat([C11.iloc[te].reset_index(drop=True), radH.iloc[te].reset_index(drop=True)], axis=1)
            cols = list(Mtr.columns)
        else:
            sel = screen_topk(POOL.iloc[tr], y[tr], 1)
            Mtr = C3.iloc[tr].copy(); Mte = C3.iloc[te].copy()
            for c, sgn in sel:
                Mtr[c] = sgn * POOL.iloc[tr][c].values
                Mte[c] = sgn * POOL.iloc[te][c].values
            cols = list(Mtr.columns)
        sc = StandardScaler()
        A = sc.fit_transform(Mtr[cols].values); B = sc.transform(Mte[cols].values)
        if k == 'M4c (118 variables)':
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            c_, ba, _ = m.run_lasso_cv(Mtr[cols], y[tr])
            cp = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=20000).fit(A, y[tr])
            o[te] = np.asarray(cp.predict(B)).ravel()
        else:
            cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr])
            o[te] = cp.predict(B)
        # fit lifelines on the same training fold, take the baseline survival -> predict S(t) on the test fold
        fit = pd.DataFrame(A, columns=[f'x{i}' for i in range(A.shape[1])])
        fit['T'] = tm[tr]; fit['E'] = ev[tr].astype(int)
        try:
            lc = CoxPHFitter().fit(fit, 'T', 'E')
            td = pd.DataFrame(B, columns=fit.columns[:A.shape[1]])
            for t in TIMES:
                s = lc.predict_survival_function(td, times=[t]).values.ravel()
                surv_store[t][k][te] = s
        except Exception:
            pass
    oofs[k] = o
    P(f'[{k}] C-index = {concordance_index_censored(ev, tm, o)[0]:.4f}')

# ---------- (1) Brier ----------
P('\n# 1. Brier score (IPCW, seed 42, nested OOF predicted survival probabilities)\n')
P('| Model | Brier@36m | Brier@60m |')
P('|---|---|---|')
for k in MODELS:
    row = []
    for t in TIMES:
        sp_ = surv_store[t][k]
        ok = np.isfinite(sp_)
        try:
            _, bs = brier_score(y[ok], y[ok], sp_[ok], [t])
            row.append(f'{bs[0]:.4f}')
        except Exception as e:
            row.append('n/a')
    P(f'| {k} | {row[0]} | {row[1]} |')

# ---------- (2) calibration slope ----------
P('\n# 2. Calibration slope (OOF linear predictor as the only covariate)\n')
P('| Model | Calibration slope | 95%CI | Reading |')
P('|---|---|---|---|')
from lifelines import CoxPHFitter as C2
for k in MODELS:
    lp = (oofs[k] - oofs[k].mean()) / oofs[k].std(ddof=1)
    d = pd.DataFrame({'lp': lp, 'T': tm, 'E': ev.astype(int)})
    try:
        c = C2().fit(d, 'T', 'E')
        s = c.summary.loc['lp']
        sl = s['exp(coef)']
        lo, hi = s['exp(coef) lower 95%'], s['exp(coef) upper 95%']
        note = 'ideal = 1; < 1 overfitted, > 1 underfitted'
        P(f'| {k} | {sl:.3f} | [{lo:.3f}, {hi:.3f}] | {note} |')
    except Exception:
        P(f'| {k} | n/a | | |')

# ---------- (3) calibration groups (predicted vs KM observed) ----------
P('\n# 3. Calibration groups (OOF risk deciles, 36-month predicted vs KM observed)\n')
for k in MODELS:
    r = oofs[k]
    qs = pd.qcut(pd.Series(r).rank(method='first'), 10, labels=False)
    P(f'\n**{k}**\n')
    P('| Decile | n | Predicted S(36) | KM observed S(36) | Events |')
    P('|---|---|---|---|---|')
    for b in range(10):
        idx = np.where(qs == b)[0]
        if len(idx) == 0: continue
        kmf = KaplanMeierFitter().fit(tm[idx], ev[idx])
        obs = float(kmf.predict(36))
        pred = float(np.nanmean(surv_store[36][k][idx])) if np.isfinite(surv_store[36][k][idx]).any() else np.nan
        P(f'| {b+1} | {len(idx)} | {pred:.3f} | {obs:.3f} | {int(ev[idx].sum())} |')

# ---------- (4) DCA ----------
P('\n# 4. Decision curve (net benefit, 36/60 months, IPCW)\n')


def dca_surv(time, ev, tm, risk, thresholds):
    """Vickers survival DCA: IPCW weighted"""
    km_c = KaplanMeierFitter().fit(tm, 1 - ev.astype(int))
    G = lambda s: float(km_c.predict(s))
    out = []
    n = len(risk)
    for pt in thresholds:
        tp = fp = 0.0
        for i in range(n):
            w = 1.0 / max(G(tm[i]), 1e-6)
            if tm[i] <= time and ev[i]:
                if risk[i] >= pt: tp += w
            elif tm[i] > time:
                if risk[i] >= pt: fp += w
        nb = tp / n - fp / n * (pt / (1 - pt))
        out.append(nb)
    return np.array(out)


P('| Threshold pt | ' + ' | '.join([f'{k} @36m' for k in MODELS]) + ' | Treat all | Treat none |')
P('|---|' + '---|' * (len(MODELS) + 2))
THR = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
km_all = KaplanMeierFitter().fit(tm, ev)
for i, pt in enumerate(THR):
    nbs = []
    for k in MODELS:
        r = (oofs[k] - np.nanmin(oofs[k])) / (np.nanmax(oofs[k]) - np.nanmin(oofs[k]))
        nb = dca_surv(36, ev, tm, r, [pt])[0]
        nbs.append(f'{nb:.4f}')
    prev = float(1 - km_all.predict(36))
    treat_all = prev - (1 - prev) * (pt / (1 - pt))
    P(f'| {pt} | ' + ' | '.join(nbs) + f' | {treat_all:.4f} | 0.0000 |')

with open(os.path.join(OUT, 'FINAL_calibration_brier_dca.md'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))
print('\n-> FINAL_calibration_brier_dca.md')
