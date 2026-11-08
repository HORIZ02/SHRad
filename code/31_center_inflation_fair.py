import os
# -*- coding: utf-8 -*-
"""31_center_inflation_fair.py — magnitude of the centre effect on a common caliber (same modelling approach on both sides)"""

# WARNING: this script is a historical diagnostic from the 401-era analysis (with vs without Center4).
#    Center4 has been excluded by the established decision and the working cohort is Center1-3 (n=345).
#    Running it as-is would leave the Center4 subset empty and yield garbage values, so execution is
#    blocked explicitly here; to reproduce the historical comparison, first restore the 401 data from
#    the archived four-centre dataset (legacy archive directory of the data tree).
raise SystemExit('Disabled: 31_center_inflation_fair.py is a historical diagnostic from the 401-era analysis '
                 'and the data no longer contain Center4. To reproduce it, first restore the original tables '
                 'from the archived four-centre dataset (legacy archive directory of the data tree).')
# ────────────────────────────────────────────────────────────────────────────

import importlib.util, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
OUT = m.OUT
SEEDS = list(range(20))

a = pd.read_csv(os.path.join(BASE, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
CL = m.CLIN_VARS


def run(df, label, Xcol=None):
    y = m.make_surv(df); ev, tm = y['event'].astype(bool), y['time'].astype(float)
    X = df[CL].fillna(df[CL].median()).astype(float).values
    ctr = df.Center.values
    pooled, within_all, n_ev = [], [], int(ev.sum())
    for sd in SEEDS:
        o = np.full(len(X), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=sd).split(X):
            s = StandardScaler()
            o[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(s.fit_transform(X[tr]), y[tr]).predict(s.transform(X[te]))
        pooled.append(concordance_index_censored(ev, tm, o)[0])
        # within centre (independent 5-fold per centre, pooling the C of each centre)
        wc = []
        for k in np.unique(ctr):
            idx = np.where(ctr == k)[0]
            if ev[idx].sum() < 5: continue
            oo = np.full(len(idx), np.nan)
            for tr, te in KFold(5, shuffle=True, random_state=sd).split(idx):
                s = StandardScaler()
                oo[te] = CoxPHSurvivalAnalysis(alpha=0.1).fit(
                    s.fit_transform(X[idx][tr]), y[idx][tr]).predict(s.transform(X[idx][te]))
            wc.append((concordance_index_censored(ev[idx], tm[idx], oo)[0], len(idx), ev[idx].sum()))
        # event-weighted average
        if wc:
            wsum = sum(x[2] for x in wc)
            within_all.append(sum(x[0] * x[2] for x in wc) / wsum)
    print(f'{label:16s} n={len(df):3d} events={n_ev:2d} | pooled C={np.mean(pooled):.4f}±{np.std(pooled,ddof=1):.4f}'
          f' | within-centre (event-weighted) C={np.mean(within_all):.4f}±{np.std(within_all,ddof=1):.4f}'
          f' | **inflation = {np.mean(pooled)-np.mean(within_all):+.4f}**')
    for k in np.unique(ctr):
        idx = np.where(ctr == k)[0]
        print(f'      {k:14s} n={len(idx):3d} events={ev[idx].sum():2d} event rate={df.DFS.values[idx].mean()*100:5.1f}%')
    return np.mean(pooled), np.mean(within_all)


print('=== Same model (clinical 11 variables, CoxPH, 20 seeds): "pooled" vs "within centre" performance gap ===\n')
c4 = a[a.Center == 'Center4-56']
p4, w4 = run(a, 'all 4 centres')
print()
p3, w3 = run(a[a.Center != 'Center4-56'].reset_index(drop=True), 'Center1-3')
print()
print(f'Conclusion:')
print(f'  4 centres: pooled {p4:.4f} vs within centre {w4:.4f} -> inflation {p4-w4:+.4f}')
print(f'  3 centres: pooled {p3:.4f} vs within centre {w3:.4f} -> inflation {p3-w3:+.4f}')
print(f'\n  -> after removing Center4 the centre-effect inflation {"increases" if (p3-w3)>(p4-w4) else "decreases"} '
      f'({(p4-w4):+.4f} -> {(p3-w3):+.4f})')
