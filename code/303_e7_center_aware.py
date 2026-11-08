# -*- coding: utf-8 -*-
"""303_e7_center_aware.py — Tier B / E-7: centre-aware validation (leave-one-centre-out, LOCO)

Reviewer concern: the current draft pools the three centres for 5-fold CV, yet the centres differ widely
in event rate (9.3%-29.7%) and case mix. Pooled 5-fold CV lets cases from the same centre appear in both
the training and the test fold, so the reported C-index mixes in an optimistic bias from
"within-centre similarity".

This script promotes **leave-one-centre-out (LOCO)** to one of the primary analyses:
  for each arm, hold out Center1 / Center2 / Center3 in turn as the test set and train on the other two,
  report the 3 held-out C-indices with (a) mean±SD and (b) range,
  and compare against pooled 5-fold CV (the current draft caliber) to quantify the optimistic bias from
  "within-centre similarity".

Cohort A: Center1(193) / Center2(78) / Center3(74)
Cohort B: single centre -> LOCO not applicable (the script skips it automatically and notes it)

Input: <SHRAD_CLIN_DIR>/merged_data_corrected.csv + data/features/* (same source as 146)
Output: results/models/303_e7_loco.csv
"""
import os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
H2 = BASE
OUT = os.path.join(H2, 'results', 'models')
SEEDS = list(range(int(os.environ.get('NSEED', '20'))))
N_FEAT = 5


def log(*s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in s)), flush=True)


import importlib.util
spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

a = pd.read_csv(os.path.join(H2, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
CASES = a['Case_No.'].tolist()
y = m.make_surv(a)
ev = np.asarray(y['event']).astype(bool); tm = np.asarray(y['time']).astype(float)
cent = a['Center'].astype(str).values
log('n=%d events=%d | centres %s' % (len(a), int(ev.sum()), pd.Series(cent).value_counts().to_dict()))


def conv(path, mod):
    d = pd.read_csv(path); d['Case_No.'] = d['Case_No.'].astype(str)
    hc = [c for c in d.columns if c.lower() in ('habitat', 'region')][0]
    out = {}
    for r, g in d.groupby(hc):
        g = g.set_index('Case_No.').drop(columns=[hc]).reindex(CASES)
        g.columns = ['%s_%s_%s' % (mod, str(r), c) for c in g.columns]
        out[str(r)] = g
    return out


R = {**conv(os.path.join(H2, 'data', 'features', 'dce_pyradiomics_k3.csv'), 'DCE')}
sp = pd.read_csv(os.path.join(H2, 'data', 'features', 'habitat_spatial_features_k3.csv'))
sp['Case_No.'] = sp['Case_No.'].astype(str)
mc = [c for c in sp.columns if c.lower() == 'modality'][0]
d = sp[sp[mc].astype(str).str.lower() == 'dce'].set_index('Case_No.')
SPD = d[[c for c in d.columns if c.startswith('dce_') and d[c].notna().sum() >= 0.8 * len(CASES)]].reindex(CASES)

WT = [v for k, v in R.items() if 'whole' in k.lower()]
HB = [v for k, v in R.items() if 'whole' not in k.lower()]
ARMS = {
    '①conventional WT · DCE': (pd.concat(WT, axis=1), None),
    '②conventional habitat · DCE': (pd.concat(HB, axis=1), None),
    '③spatial habitat · DCE': (SPD, SPD),
}
C3 = a[['NAC', 'N', 'ALN']].reset_index(drop=True)
log('arms: %s' % list(ARMS))


def screen_topk(P, ytr, k=N_FEAT):
    """In-fold univariable C-index screening, top-k (same rule as 146/32)"""
    v = []
    for j in range(P.shape[1]):
        col = P[:, j]
        if np.nanstd(col) == 0:
            v.append(-1); continue
        try:
            v.append(concordance_index_censored(ytr['event'], ytr['time'], col)[0])
        except Exception:
            v.append(-1)
    v = np.array(v); v = np.where(np.isfinite(v), np.abs(v - 0.5), -1)
    return np.argsort(-v)[:k]


def fit_pred(A, ytr, B, cols=None):
    A = A if cols is None else A[:, cols]
    B = B if cols is None else B[:, cols]
    ss = StandardScaler(); A = ss.fit_transform(A); B = ss.transform(B)
    cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, ytr)
    return cp.predict(B)


def run_loco(X, y, cent, seeds=SEEDS):
    Xv = X.fillna(0).astype(float).values
    centers = sorted(set(cent))
    res = {}
    for hold in centers:
        te = np.where(cent == hold)[0]; tr = np.where(cent != hold)[0]
        Cs = []
        for sd in seeds:
            try:
                cols = screen_topk(Xv[tr], y[tr])
                pr = fit_pred(Xv[tr], y[tr], Xv[te], cols)
                Cs.append(concordance_index_censored(ev[te], tm[te], pr)[0])
            except Exception:
                pass
        res[hold] = (np.mean(Cs), np.std(Cs, ddof=1)) if Cs else (np.nan, np.nan)
    return res


def run_pooled(X, y, seeds=SEEDS):
    Xv = X.fillna(0).astype(float).values
    Cs = []
    for sd in seeds:
        per = []
        for tr, te in KFold(5, shuffle=True, random_state=sd).split(Xv):
            try:
                cols = screen_topk(Xv[tr], y[tr])
                pr = fit_pred(Xv[tr], y[tr], Xv[te], cols)
                per.append(concordance_index_censored(ev[te], tm[te], pr)[0])
            except Exception:
                pass
        if per:
            Cs.append(np.mean(per))
    return (np.mean(Cs), np.std(Cs, ddof=1)) if Cs else (np.nan, np.nan)


rows = []
for name, (X, _) in ARMS.items():
    lo = run_loco(X, y, cent)
    pooled = run_pooled(X, y)
    vals = [lo[c][0] for c in sorted(lo) if np.isfinite(lo[c][0])]
    rows.append(dict(arm=name,
                     C1=round(lo.get('Center1-193', (np.nan,))[0], 4),
                     C2=round(lo.get('Center2-78', (np.nan,))[0], 4),
                     C3=round(lo.get('Center3-74', (np.nan,))[0], 4),
                     LOCO_mean=round(float(np.mean(vals)), 4) if vals else np.nan,
                     LOCO_sd=round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else np.nan,
                     LOCO_range=round(float(max(vals) - min(vals)), 4) if vals else np.nan,
                     pooled_C=round(pooled[0], 4), pooled_sd=round(pooled[1], 4),
                     optimism=round(pooled[0] - (np.mean(vals) if vals else np.nan), 4)))
    log('  %-28s LOCO %s | mean %.4f (range %.4f) | pooled %.4f | optimism %+.4f'
        % (name, {c[:7]: round(lo[c][0], 3) for c in sorted(lo)}, rows[-1]['LOCO_mean'],
           rows[-1]['LOCO_range'], rows[-1]['pooled_C'], rows[-1]['optimism']))

L = pd.DataFrame(rows)
L['NOTE'] = 'cohort B is single-centre, LOCO not applicable'
L.to_csv(os.path.join(OUT, '303_e7_loco.csv'), index=False, encoding='utf-8-sig')
log('\n-> 303_e7_loco.csv')
log('\nOptimism = pooled - LOCO mean; a value > 0 means pooled 5-fold CV overestimates true external performance.')
