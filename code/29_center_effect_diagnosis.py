import os
# -*- coding: utf-8 -*-
"""
29_center_effect_diagnosis.py — multi-centre centre-effect diagnosis

Background: the four centres differ strongly in case mix (event rate 7.1%-29.7%,
      N-positive rate 29%-86.5%). If the model is evaluated on the pooled data,
      "centre" can act as a risk factor -> inflated C-index.

Tests:
  (1) Centre-only model: use only the event rate of that centre within the training fold as the risk score
      -> how much C-index does a pure centre effect buy
  (2) Clinical model vs clinical + centre
  (3) Leave-one-centre-out cross-validation (LOCO): train on 3 centres -> predict the held-out centre
      -> cross-centre generalisation
  (4) Within-centre C-index (5-fold inside each centre) -> true patient-level discrimination
  (5) Centre separability: can the clinical variables predict which centre a case belongs to
"""
import importlib.util, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE, OUT = m.BASE, m.OUT

c = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
c['Case_No.'] = c['Case_No.'].astype(str)
y = m.make_surv(c)
ev, tm = y['event'].astype(bool), y['time'].astype(float)
ctr = c['Center'].values
CL = m.CLIN_VARS
X = c[CL].fillna(c[CL].median()).astype(float).values

print('=' * 92)
print(f'Cohort {len(c)} cases / {ev.sum()} events; centres: ' +
      ', '.join(f'{k}(n={v},e={int(c[c.Center==k].DFS.sum())})' for k, v in c.Center.value_counts().items()))

# (1) Centre-only model ---------------------------------------------------------
print('\n[1] Pure centre effect: risk score = within-fold event rate of each centre')
kf = KFold(5, shuffle=True, random_state=42)
risk = np.zeros(len(c))
for tr, te in kf.split(c):
    rate = c.iloc[tr].groupby('Center')['DFS'].mean()
    risk[te] = c.iloc[te]['Center'].map(rate).fillna(rate.mean()).values
c_ctr = concordance_index_censored(ev, tm, risk)[0]
lo, hi = m.boot_ci(ev, tm, risk)
print(f'  C-index = {c_ctr:.4f}  [{lo:.3f},{hi:.3f}]   <- no imaging or clinical information at all!')


def oof_cox(Xm, y, seed=42):
    k = KFold(5, shuffle=True, random_state=seed)
    o = np.full(len(Xm), np.nan)
    for tr, te in k.split(Xm):
        s = StandardScaler(); a = s.fit_transform(Xm[tr]); b = s.transform(Xm[te])
        o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(a, y[tr]).predict(b)
    return o


# (2) Clinical vs clinical + centre ---------------------------------------------
print('\n[2] Clinical model with and without centre')
o_clin = oof_cox(X, y)
cc = concordance_index_censored(ev, tm, o_clin)[0]
print(f'  Clinical (11 variables)        C = {cc:.4f}  [{m.boot_ci(ev,tm,o_clin)[0]:.3f},{m.boot_ci(ev,tm,o_clin)[1]:.3f}]')
Xc = np.column_stack([X, pd.get_dummies(c.Center).values.astype(float)])
o_cc = oof_cox(Xc, y)
print(f'  Clinical + centre dummies      C = {concordance_index_censored(ev, tm, o_cc)[0]:.4f}  ({Xc.shape[1]} variables)')

# (3) Leave-one-centre-out (LOCO) -----------------------------------------------
print('\n[3] Leave-one-centre-out CV: train on 3 centres -> predict the held-out centre')
for k in sorted(c.Center.unique()):
    tr = ctr != k; te = ctr == k
    if ev[tr].sum() < 10 or ev[te].sum() < 2:
        print(f'  {k:14s} skipped (training events {ev[tr].sum()} / test events {ev[te].sum()} too few)')
        continue
    s = StandardScaler(); a = s.fit_transform(X[tr]); b = s.transform(X[te])
    cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(a, y[tr])
    pr = cp.predict(b)
    c_lo = concordance_index_censored(ev[te], tm[te], pr)[0]
    print(f'  {k:14s} n={te.sum():3d} events={ev[te].sum():2d}  cross-centre C = {c_lo:.4f}')

# (4) Within-centre C-index ------------------------------------------------------
print('\n[4] Within-centre C-index: 5-fold inside each centre (true patient-level discrimination)')
within = []
for k in sorted(c.Center.unique()):
    idx = np.where(ctr == k)[0]
    if ev[idx].sum() < 5:
        print(f'  {k:14s} skipped (only {ev[idx].sum()} events)')
        continue
    o = np.full(len(idx), np.nan)
    kk = KFold(5, shuffle=True, random_state=42)
    for tr, te in kk.split(idx):
        s = StandardScaler(); a = s.fit_transform(X[idx][tr]); b = s.transform(X[idx][te])
        o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(a, y[idx][tr]).predict(b)
    val = o[np.isfinite(o)]
    cwi = concordance_index_censored(ev[idx][np.isfinite(o)], tm[idx][np.isfinite(o)], val)[0]
    within.append(cwi)
    print(f'  {k:14s} n={len(idx):3d} events={ev[idx].sum():2d}  within-centre C = {cwi:.4f}')
print(f'  --> within-centre mean = {np.mean(within):.4f}   vs   pooled OOF clinical = {cc:.4f}' if within else '')

# (5) Centre separability --------------------------------------------------------
print('\n[5] Centre separability: can the clinical variables predict which centre a case belongs to')
Xz = StandardScaler().fit_transform(X)
for name, clf in [('LogReg', LogisticRegression(max_iter=2000)),
                  ('RandomForest', RandomForestClassifier(n_estimators=300, random_state=42))]:
    yc = pd.factorize(c.Center)[0]
    pred = cross_val_predict(clf, Xz, yc, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    acc = (pred == yc).mean()
    base = pd.Series(yc).value_counts(normalize=True).max()
    print(f'  {name:14s} accuracy = {acc:.4f}   (random baseline {base:.4f})  -> {"separable" if acc > base + 0.08 else "not separable"}')

df = pd.DataFrame([dict(diagnostic='pure centre effect', C=round(c_ctr, 4)),
                   dict(diagnostic='clinical 11', C=round(cc, 4)),
                   dict(diagnostic='within-centre mean', C=round(float(np.mean(within)), 4))])
df.to_csv(os.path.join(OUT, 'center_effect_diagnosis_k3.csv'), index=False, encoding='utf-8-sig')
print('\n-> center_effect_diagnosis_k3.csv')
