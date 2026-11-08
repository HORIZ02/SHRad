import os
# -*- coding: utf-8 -*-
"""
26_parsimonious_models.py — parsimonious model check (EPV-compliant route)

Motivation: 55 events / EPV = 10 -> at most ~5 variables are supported.
      Test: can a minimal model such as "3 clinical variables + 1 genuinely novel
      spatial feature (dwi_M_n_components)" match the performance of M4c (118 variables)?
      If so, the selling point is "same performance, 30-fold less complexity".

Fixed variable sets + 5-fold CV (no inner selection, to avoid selection bias)
+ bootstrap 1000x 95% CI
Reference: M4c nested OOF C = 0.6859
"""
import importlib.util, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT, SEED = m.BASE, m.OUT, m.SEED

clin = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
clin['Case_No.'] = clin['Case_No.'].astype(str)
y = m.make_surv(clin); ev, tm = y['event'].astype(bool), y['time'].astype(float)

sp = pd.read_csv(os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv'))
sp['Case_No.'] = sp['Case_No.'].astype(str)
SPAT = {c: sp[c].values for c in sp.columns if c in ('dwi_M_n_components', 'dce_H_n_components',
        'dce_H_euler', 'dce_L_centroid_dist', 'dwi_H_euler', 'dwi_M_H_centroid_dist')}
# taken from the raw long table (DWI rows)
dw = sp[sp.modality == 'DWI'].set_index('Case_No.')
dc = sp[sp.modality == 'DCE'].set_index('Case_No.')
order = clin['Case_No.']
pick = lambda df, c: df.loc[order, c].reset_index(drop=True).astype(float)

S3  = ['NAC', 'N', 'ALN']
S6  = ['NAC', 'N', 'Age', 'Grade', 'ALN', 'proRT']
S11 = m.CLIN_VARS
NC  = ('dwi_M_n_components', pick(dw, 'dwi_M_n_components'), 'only genuinely novel + FDR-significant')
SPH = ('dce_H_sphericity',    pick(dc, 'dce_H_sphericity'),    'sphericity (ρ = 0.999 with an existing M4c feature)')

sets = [
    ('Clinical 3 (NAC, N, ALN)',        S3, []),
    ('Clinical 3 + n_components',      S3, [NC]),
    ('Clinical 3 + sphericity',        S3, [SPH]),
    ('Clinical 3 + n_comp + sphericity', S3, [NC, SPH]),
    ('Clinical 6 (original sel_vars)', S6, []),
    ('Clinical 6 + n_components',      S6, [NC]),
    ('Clinical 6 + n_comp + sphericity', S6, [NC, SPH]),
    ('Clinical 11 (M0)',               S11, []),
    ('Clinical 11 + n_components',     S11, [NC]),
    ('Clinical 11 + n_comp + sphericity', S11, [NC, SPH]),
]

Cbase = clin[S11].reset_index(drop=True)
rows, oofs = [], {}
print('=' * 96)
print('Reference: M4c (118 variables, nested OOF) C = 0.6859 ; M0 clinical 11 = 0.6860\n')
for name, cvars, extra in sets:
    X = Cbase[cvars].copy()
    for e in extra:
        X[e[0]] = e[1]
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(X), np.nan)
    for tr, te in kf.split(X.values):
        s = StandardScaler(); Xtr = s.fit_transform(X.values[tr]); Xte = s.transform(X.values[te])
        cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(Xtr, y[tr])
        oof[te] = cp.predict(Xte)
    c = m.concordance_index_censored(ev, tm, oof)[0]
    lo, hi = m.boot_ci(ev, tm, oof)
    oofs[name] = oof
    rows.append(dict(Model=name, n_vars=X.shape[1], C_OOF=round(float(c), 4), CI=f'[{lo:.3f},{hi:.3f}]'))
    print(f'  {name:26s} vars={X.shape[1]:2d}  C_OOF={c:.4f}  [{lo:.3f},{hi:.3f}]')

# comparison with M4c
_rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
_rad['Case_No.'] = _rad['Case_No.'].astype(str)
_rf = [c for c in _rad.columns if c not in ('Case_No.', 'Habitat')]
_radH = _rad[_rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.').loc[order, _rf].reset_index(drop=True)
X_m4c = pd.concat([Cbase, _radH.add_prefix('DCEH_')], axis=1).fillna(0).astype(float)
m4c_oof = m.nested_oof(X_m4c, y)
print('\n=== Paired bootstrap comparison against M4c (0.6859) ===')
for i, r0 in enumerate(rows):
    dlo, dhi, d, p = m.paired_boot(ev, tm, oofs[r0['Model']], m4c_oof)
    rows[i].update(dict(dC_vs_M4c=round(d, 4), dC_CI=f'[{dlo:+.3f},{dhi:+.3f}]', p=round(p, 4)))
    flag = '  ✅ not inferior' if p > 0.05 else ''
    print(f"  {r0['Model']:26s} dC={d:+.4f} [{dlo:+.3f},{dhi:+.3f}]  p={p:.4f}{flag}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'model_comparison_parsimonious_k3.csv'), index=False, encoding='utf-8-sig')
print('\n-> model_comparison_parsimonious_k3.csv')
