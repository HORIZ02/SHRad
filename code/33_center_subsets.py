import os
# -*- coding: utf-8 -*-
"""33_center_subsets.py — performance comparison across centre subsets (answering "does dropping a centre help")

# ────────────────────────────────────────────────────────────────────────────


Why did removing Center4 not improve anything? Measure C-index / AUC on each centre subset.
"""

# WARNING: historical diagnostic from the 401-era analysis (with vs without Center4). Center4 has been
#    excluded by the established decision and the working cohort is Center1-3 (n=345). The data source
#    no longer contains Center4, so each subset would be silently mislabelled (e.g. C1+C2+C4 is in fact
#    C1+C2); execution is therefore blocked explicitly. To reproduce the historical comparison, first
#    restore the original tables from the archived four-centre dataset (legacy archive directory of the
#    data tree).
raise SystemExit('Disabled: 33_center_subsets.py is a historical diagnostic from the 401-era analysis '
                 'and the data no longer contain Center4.')

import importlib.util, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT, CL = m.BASE, m.OUT, m.CLIN_VARS
SEEDS = list(range(5))

a = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
_rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
_rad['Case_No.'] = _rad['Case_No.'].astype(str)
RF = [c for c in _rad.columns if c not in ('Case_No.', 'Habitat')]

C1, C2, C3, C4 = 'Center1-193', 'Center2-78', 'Center3-74', 'Center4-56'
SUBSETS = {
    '全部4中心': [C1, C2, C3, C4],
    'C1+C2+C3': [C1, C2, C3],
    'C1+C2+C4': [C1, C2, C4],
    'C1+C3+C4': [C1, C3, C4],
    'C2+C3+C4': [C2, C3, C4],
    'C1+C2': [C1, C2],
    'C1+C3': [C1, C3],
    'C2+C3': [C2, C3],
    'C1+C4': [C1, C4],
    '仅C1': [C1],
}


def run(sub, label):
    df = a[a.Center.isin(sub)].reset_index(drop=True)
    y = m.make_surv(df); ev, tm = y['event'].astype(bool), y['time'].astype(float)
    order = df['Case_No.']
    radH = _rad[_rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.').loc[order, RF].reset_index(drop=True)
    C = df[CL].fillna(df[CL].median()).astype(float).values
    R = radH.values
    out = dict(子集=label, n=len(df), 事件=int(ev.sum()), 事件率=f'{ev.sum()/len(df)*100:.1f}%')
    for nm, X in [('M0', C), ('M4c', np.column_stack([C, R]))]:
        cs, a36, a60 = [], [], []
        for sd in SEEDS:
            oof = np.full(len(df), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=sd).split(X):
                s = StandardScaler(); A = s.fit_transform(X[tr]); B = s.transform(X[te])
                if nm == 'M0':
                    pr = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr]).predict(B)
                else:
                    try:
                        _, ba, _ = m.run_lasso_cv(pd.DataFrame(X[tr]), y[tr])
                        pr = np.asarray(CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba],
                                          max_iter=20000).fit(A, y[tr]).predict(B)).ravel()
                    except Exception:
                        pr = np.zeros(len(te))
                oof[te] = pr
            cs.append(concordance_index_censored(ev, tm, oof)[0])
            try:
                au, _ = cumulative_dynamic_auc(y, y, oof, [36, 60])
                a36.append(au[0]); a60.append(au[1])
            except Exception:
                pass
        out[f'{nm}_C'] = round(float(np.mean(cs)), 4)
        out[f'{nm}_C_SD'] = round(float(np.std(cs, ddof=1)), 4)
        out[f'{nm}_AUC36'] = round(float(np.mean(a36)), 4) if a36 else np.nan
        out[f'{nm}_AUC60'] = round(float(np.mean(a60)), 4) if a60 else np.nan
    return out


print('=== Centre-subset comparison (5 seeds x 5 folds) ===\n')
rows = [run(v, k) for k, v in SUBSETS.items()]
d = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(d[['子集', 'n', '事件', '事件率', 'M0_C', 'M4c_C', 'M4c_C_SD', 'M4c_AUC36', 'M4c_AUC60']].to_string(index=False))
d.to_csv(os.path.join(OUT, 'center_subsets_k3.csv'), index=False, encoding='utf-8-sig')

print('\n=== Key comparison: with and without Center4 ===')
f = d[d.子集 == '全部4中心'].iloc[0]
t = d[d.子集 == 'C1+C2+C3'].iloc[0]
for k in ['M0_C', 'M4c_C', 'M4c_AUC36', 'M4c_AUC60']:
    print(f'  {k:12s} 4 centres={f[k]:.4f}  C1-3={t[k]:.4f}  diff={t[k]-f[k]:+.4f}')
print('\n=== Which centre drags performance down (within-centre C-index) ===\n')
for c in [C1, C2, C3, C4]:
    df = a[a.Center == c].reset_index(drop=True)
    if len(df) == 0: continue
    y = m.make_surv(df); ev, tm = y['event'].astype(bool), y['time'].astype(float)
    if ev.sum() < 5:
        print(f'  {c:14s} n={len(df):3d} events={ev.sum():2d}  too few events, skipped')
        continue
    X = df[CL].fillna(df[CL].median()).astype(float).values
    o = np.full(len(X), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
        s = StandardScaler()
        o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(s.fit_transform(X[tr]), y[tr]).predict(s.transform(X[te]))
    print(f'  {c:14s} n={len(df):3d} events={ev.sum():2d}  within-centre C={concordance_index_censored(ev, tm, o)[0]:.4f}')
