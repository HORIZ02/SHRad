import os
# -*- coding: utf-8 -*-
"""
27_parsimonious_stability.py — stability check of the parsimonious model (multiple seeds)

Problem: in a single 5-fold CV, "clinical 3 + sphericity" (4 variables) reached C = 0.7248,
      significantly above M4c at 0.6859 (p = 0.004).
      Partition luck has to be excluded, and the feature responsible has to be identified.

Approach: 10 random seeds x 5-fold CV, reporting mean +/- SD; plus a feature-swap decomposition.
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
BASE, OUT = m.BASE, m.OUT
SEEDS = [42, 7, 123, 2024, 888, 99, 314, 555, 1234, 66]

clin = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
clin['Case_No.'] = clin['Case_No.'].astype(str)
y = m.make_surv(clin); ev, tm = y['event'].astype(bool), y['time'].astype(float)

sp = pd.read_csv(os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv'))
sp['Case_No.'] = sp['Case_No.'].astype(str)
dw = sp[sp.modality == 'DWI'].set_index('Case_No.')
dc = sp[sp.modality == 'DCE'].set_index('Case_No.')

_rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
_rad['Case_No.'] = _rad['Case_No.'].astype(str)
_rf = [c for c in _rad.columns if c not in ('Case_No.', 'Habitat')]
_radH = _rad[_rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.')

order = clin['Case_No.']
g = lambda df, c: df.loc[order, c].reset_index(drop=True).astype(float)

BASE3 = clin[['NAC', 'N', 'ALN']].reset_index(drop=True)
C11 = clin[m.CLIN_VARS].reset_index(drop=True)

cands = {
    'Clinical 3':                      pd.concat([BASE3], axis=1),
    'Clinical 3 + sphericity (spatial)': pd.concat([BASE3, g(dc, 'dce_H_sphericity').rename('sph')], axis=1),
    'Clinical 3 + sphericity (PyRad)': pd.concat([BASE3, g(_radH, 'original_shape_Sphericity').rename('sph_p')], axis=1),
    'Clinical 3 + Max3DDiam':          pd.concat([BASE3, g(_radH, 'original_shape_Maximum3DDiameter').rename('m3d')], axis=1),
    'Clinical 3 + Grade':              pd.concat([BASE3, clin['Grade'].reset_index(drop=True).rename('grade')], axis=1),
    'Clinical 3 + sphericity + Max3D': pd.concat([BASE3, g(dc, 'dce_H_sphericity').rename('sph'),
                                        g(_radH, 'original_shape_Maximum3DDiameter').rename('m3d')], axis=1),
    'Clinical 3 + sphericity + Grade': pd.concat([BASE3, g(dc, 'dce_H_sphericity').rename('sph'),
                                        clin['Grade'].reset_index(drop=True).rename('grade')], axis=1),
    'Clinical 6':                      clin[['NAC', 'N', 'Age', 'Grade', 'ALN', 'proRT']].reset_index(drop=True),
    'Clinical 11':                     C11,
    'Sphericity only (spatial)':       pd.concat([g(dc, 'dce_H_sphericity').rename('sph')], axis=1),
    'Sphericity only (PyRad)':         pd.concat([g(_radH, 'original_shape_Sphericity').rename('sph_p')], axis=1),
}

print(f'{"Model":22s} {"vars":>4} {"C_OOF mean":>10} {"SD":>7} {"min":>7} {"max":>7}')
print('-' * 62)
res = []
for name, X in cands.items():
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    cs = []
    for sd in SEEDS:
        kf = KFold(5, shuffle=True, random_state=sd)
        oof = np.full(len(X), np.nan)
        for tr, te in kf.split(X.values):
            s = StandardScaler(); Xtr = s.fit_transform(X.values[tr]); Xte = s.transform(X.values[te])
            cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(Xtr, y[tr])
            oof[te] = cp.predict(Xte)
        cs.append(m.concordance_index_censored(ev, tm, oof)[0])
    res.append(dict(Model=name, n=X.shape[1], mean=float(np.mean(cs)), sd=float(np.std(cs, ddof=1)),
                    min=float(np.min(cs)), max=float(np.max(cs))))
    print(f'{name:22s} {X.shape[1]:4d} {np.mean(cs):10.4f} {np.std(cs, ddof=1):7.4f} '
          f'{np.min(cs):7.4f} {np.max(cs):7.4f}')

# M4c across the same seeds (nested LASSO)
X_m4c = pd.concat([C11, _radH.loc[order, _rf].reset_index(drop=True).add_prefix('DCEH_')], axis=1).fillna(0).astype(float)
print('-' * 62)
cs = []
for sd in SEEDS:
    oof = m.nested_oof(X_m4c, y, seed=sd)
    cs.append(m.concordance_index_censored(ev, tm, oof)[0])
res.append(dict(Model='M4c(118,LASSO)', n=X_m4c.shape[1], mean=float(np.mean(cs)),
                sd=float(np.std(cs, ddof=1)), min=float(np.min(cs)), max=float(np.max(cs))))
print(f'{"M4c(118,LASSO)":22s} {X_m4c.shape[1]:4d} {np.mean(cs):10.4f} {np.std(cs, ddof=1):7.4f} '
      f'{np.min(cs):7.4f} {np.max(cs):7.4f}')

df = pd.DataFrame(res)
df.to_csv(os.path.join(OUT, 'parsimonious_stability_k3.csv'), index=False, encoding='utf-8-sig')
print('\n=== Verdict (against the multi-seed M4c mean) ===')
m4c = df[df.Model == 'M4c(118,LASSO)'].iloc[0]
for _, r in df.iterrows():
    if r.Model == 'M4c(118,LASSO)': continue
    dm = r['mean'] - m4c['mean']
    z = dm / max(np.hypot(r.sd, m4c.sd), 1e-9)
    tag = '✅ consistently better' if z > 1.0 else ('≈ on par' if abs(z) <= 1.0 else '❌ worse')
    print(f"  {r.Model:26s} dMean={dm:+.4f}  (pooled SD={np.hypot(r.sd,m4c.sd):.4f})  z={z:+.2f}  {tag}")
print('\n-> parsimonious_stability_k3.csv')
