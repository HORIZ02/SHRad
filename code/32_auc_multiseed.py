import os
# -*- coding: utf-8 -*-
"""32_auc_multiseed.py — time-dependent AUC of each model in Center1-3, multi-seed robust version

Output: AUC@36m / AUC@60m, averaged over 20 seeds x 5 folds with ± SD and 95% CI
Cohort: Center1-3 (n=345, 51 events); Center4 has been excluded by the established decision
"""
import importlib.util, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT = m.BASE, m.OUT
CL, MAX_MISSING = m.CLIN_VARS, m.MAX_MISSING
SEEDS = list(range(20))
TIMES = [36, 60]

a = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)


def build(df):
    y = m.make_surv(df); order = df['Case_No.']
    _rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
    _rad['Case_No.'] = _rad['Case_No.'].astype(str)
    rf = [c for c in _rad.columns if c not in ('Case_No.', 'Habitat')]
    radH = _rad[_rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.').loc[order, rf].reset_index(drop=True)
    sp = pd.read_csv(os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv'))
    sp['Case_No.'] = sp['Case_No.'].astype(str)
    dw = sp[sp.modality == 'DWI'].set_index('Case_No.')
    dc = sp[sp.modality == 'DCE'].set_index('Case_No.')
    spw = pd.concat([dw[[c for c in sp.columns if c.startswith('dwi_')]].loc[order].reset_index(drop=True),
                     dc[[c for c in sp.columns if c.startswith('dce_')]].loc[order].reset_index(drop=True)], axis=1)
    ms = spw.isna().mean(); spw = spw[ms[ms <= MAX_MISSING].index]
    C11 = df[CL].reset_index(drop=True)
    C3 = df[['NAC', 'N', 'ALN']].reset_index(drop=True)
    SPH = dc.loc[order, 'dce_H_sphericity'].reset_index(drop=True).astype(float)
    sets = {
        'M0 临床11': pd.concat([C11], axis=1),
        'M4c 原论文(118)': pd.concat([C11, radH.add_prefix('DCEH_')], axis=1),
        'M4c+P2': pd.concat([C11, radH.add_prefix('DCEH_'), spw], axis=1),
        'P2_only': pd.concat([spw], axis=1),
        # the "clinical 3 + sphericity" arm was removed: that feature was selected on the full cohort,
        # which is selection leakage (flagged in the project record, do not cite)
    }
    sets['临床3+折内top1'] = pd.concat([C3, spw], axis=1)   # top-1 spatial descriptor picked inside the fold
    return y, sets, spw


# S2 needs "one point = one random partition", so per-seed values are stored alongside the summary
PERSEED = []


def auc_ms(y, sets, tag, spw=None):
    print(f'\n=== {tag} ===')
    print(f'{"Model":18s} {"AUC@36m":>16} {"AUC@60m":>16}')
    out = []
    idx = np.arange(len(y))          # the fixed_idx argument was removed (never used)
    spw_arr = spw.fillna(0).astype(float).values if spw is not None else None
    for name, X in sets.items():
        X = X.fillna(0).astype(float).values
        res = {t: [] for t in TIMES}
        for sd in SEEDS:
            per = {t: [] for t in TIMES}
            for tr, te in KFold(5, shuffle=True, random_state=sd).split(X):
                s = StandardScaler(); A = s.fit_transform(X[tr]); B = s.transform(X[te])
                try:
                    if name == '临床3+折内top1':
                        # in-fold screening: pick the top-1 spatial descriptor by training-fold univariable C-index
                        sp = spw_arr[tr]                       # spatial feature subset of the training fold
                        best, bestc = None, -1.0
                        for j in range(sp.shape[1]):
                            v = sp[:, j]
                            if np.nanstd(v) == 0:
                                continue
                            cc = concordance_index_censored(y[tr]['event'], y[tr]['time'], v)[0]
                            if cc > bestc:
                                bestc, best = cc, j
                        if best is None:
                            continue
                        keep = np.r_[np.arange(3), 3 + best]   # the first 3 columns are the clinical 3
                        cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A[:, keep], y[tr])
                        pr = cp.predict(B[:, keep])
                    else:
                        c_, ba, _ = m.run_lasso_cv(pd.DataFrame(X[tr]), y[tr])
                        cp = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=20000).fit(A, y[tr])
                        pr = np.asarray(cp.predict(B)).ravel()
                    au, _ = cumulative_dynamic_auc(y[tr], y[te], pr, TIMES)
                    for j, t in enumerate(TIMES):
                        per[t].append(au[j])
                except Exception:
                    pass
            for t in TIMES:
                if per[t]:
                    res[t].append(np.mean(per[t]))
        for t in TIMES:
            for _sd, _v in zip(SEEDS, res[t]):
                PERSEED.append(dict(cohort=tag, model=name, time=t, seed=_sd, auc=float(_v)))
        line = {}
        for t in TIMES:
            v = np.array(res[t])
            v = v[np.isfinite(v)]
            if len(v):
                lo, hi = np.percentile(v, [2.5, 97.5])
                line[t] = (v.mean(), v.std(ddof=1), lo, hi)
            else:
                line[t] = (np.nan,) * 4
        print(f'{name:18s} {line[36][0]:.4f}±{line[36][1]:.4f}  {line[60][0]:.4f}±{line[60][1]:.4f}')
        out.append(dict(队列=tag, Model=name,
                        AUC36=round(line[36][0], 4), AUC36_SD=round(line[36][1], 4),
                        AUC36_CI=f'[{line[36][2]:.3f},{line[36][3]:.3f}]',
                        AUC60=round(line[60][0], 4), AUC60_SD=round(line[60][1], 4),
                        AUC60_CI=f'[{line[60][2]:.3f},{line[60][3]:.3f}]'))
    return out


# Single cohort only: `a` is already Center1-3 (345 cases). The previous version had a separate
#   'all 4 centres (n=401, 55 events)' branch here; its per-seed values were verified to be identical
#   to this branch point by point (largest absolute difference 0.000e+00) -- Center4 had long been
#   dropped from the data source and that branch was only a stale label, so it was removed.
assert 'Center4' not in set(a.Center.astype(str)), 'data source still contains Center4, clean it first'
rows = []
y3, s3, SPW3 = build(a.reset_index(drop=True))
rows += auc_ms(y3, s3, 'Center1-3 (n=345, 51 events)', spw=SPW3)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'auc_multiseed_k3.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(PERSEED).to_csv(os.path.join(OUT, 'auc_multiseed_k3_perseed.csv'),
                                    index=False, encoding='utf-8-sig')

print(f'-> {os.path.join(OUT, "auc_multiseed_k3.csv")}')
