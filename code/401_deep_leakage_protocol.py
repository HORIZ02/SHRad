# -*- coding: utf-8 -*-
"""401_deep_leakage_protocol.py — run the same evaluation protocol on the deep features (the core of plan B)

Purpose: upgrade the paper's claim from "handcrafted radiomics descriptors add nothing" to
      "**the evaluation protocol determines the reported increment of any imaging feature** — handcrafted
      or learned".

Three things:
  (1) Leakage-protocol contrast: in-fold screening vs full-sample screening -> C-index swing
      reference benchmark = 0.0399 for the handcrafted arm (clinical 3 + sphericity 0.7498 -> clinical 3 + in-fold top1 0.7099)
  (2) Redundancy of deep features against conventional features: are there components in the deep
      embedding that are highly correlated with conventional features
  (3) EPV matching: also using only 5 variables (as in the three-arm comparison) to keep the arms comparable

Cohort A (345 cases / 51 events), three modality sets: DWI / DCE / DWI+DCE
Output: results/deep/401_deep_protocol.csv
      results/deep/401_deep_redundancy.csv
"""
import os, time, importlib.util
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

H2 = BASE
DEEP = os.path.join(H2, 'results', 'deep')
OUT = os.path.join(H2, 'results', 'models')
SEEDS = list(range(20))
K_VARS = 5                     # matches the EPV-matched three-arm comparison


def log(*s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in s)), flush=True)


spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

a = pd.read_csv(os.path.join(H2, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
CASES = a['Case_No.'].tolist()
y = m.make_surv(a)
ev = np.asarray(y['event']).astype(bool); tm = np.asarray(y['time']).astype(float)
log('cohort A n=%d events=%d' % (len(a), int(ev.sum())))

D = {}
for mod in ['dwi', 'dce']:
    d = pd.read_csv(os.path.join(DEEP, 'deep_emb_%s.csv' % mod), encoding='utf-8-sig')
    cn = d.columns[0]
    d[cn] = d[cn].astype(str)
    cols = [c for c in d.columns if c.startswith('deep_')]
    D[mod] = d.set_index(cn)[cols].reindex(CASES)
    log('  %s embedding %s' % (mod, D[mod].shape))

# conventional features (used for the redundancy analysis)
cv = pd.read_csv(os.path.join(H2, 'data', 'features', 'dce_pyradiomics_k3.csv'))
cv['Case_No.'] = cv['Case_No.'].astype(str)
hc = [c for c in cv.columns if c.lower() in ('habitat', 'region')][0]
Cw = cv.pivot_table(index='Case_No.', columns=hc, aggfunc='first')
Cw.columns = ['%s_%s' % (x, y_) for x, y_ in Cw.columns]
Cv = Cw.reindex(CASES).values.astype(float)


def screen_topk(P, ytr, k=K_VARS):
    v = []
    for j in range(P.shape[1]):
        c = P[:, j]
        if np.nanstd(c) == 0:
            v.append(-1); continue
        try:
            cc = concordance_index_censored(ytr['event'], ytr['time'], c)[0]
        except Exception:
            cc = 0.5
        v.append(abs(cc - 0.5) if np.isfinite(cc) else -1)
    return np.argsort(-np.asarray(v))[:k]


def eval_arm(X, leaky, seeds=SEEDS):
    """leaky=True -> screen the top-k on the full sample (leakage); False -> screen within each training fold (correct)"""
    Xv = X.fillna(0).astype(float).values
    if leaky:
        cols = screen_topk(Xv, y)
    perf = []
    for sd in seeds:
        per = []
        for tr, te in KFold(5, shuffle=True, random_state=sd).split(Xv):
            c = cols if leaky else screen_topk(Xv[tr], y[tr])
            ss = StandardScaler(); A = ss.fit_transform(Xv[tr][:, c]); B = ss.transform(Xv[te][:, c])
            try:
                cp = CoxPHSurvivalAnalysis(alpha=0.1).fit(A, y[tr])
                pr = np.asarray(cp.predict(B)).ravel()
                per.append(concordance_index_censored(ev[te], tm[te], pr)[0])
            except Exception:
                pass
        if per:
            perf.append(np.mean(per))
    perf = np.array(perf)
    return perf.mean(), perf.std(ddof=1) if len(perf) > 1 else np.nan


ARMS = {'deep·DWI': D['dwi'], 'deep·DCE': D['dce'],
        'deep·DWI+DCE': pd.concat([D['dwi'], D['dce']], axis=1)}

rows = []
for name, X in ARMS.items():
    cin, sin = eval_arm(X, False)
    clk, slk = eval_arm(X, True)
    rows.append(dict(arm=name, n_dim=X.shape[1], K=K_VARS,
                     C_infold=round(cin, 4), C_leaky=round(clk, 4),
                     swing=round(clk - cin, 4), sd_infold=round(sin, 4)))
    log('  %-14s in-fold screening %.4f | full-sample screening %.4f | swing %+.4f' % (name, cin, clk, clk - cin))

T = pd.DataFrame(rows)
T['ref_swing_handcrafted'] = 0.0399
T.to_csv(os.path.join(OUT, '401_deep_protocol.csv'), index=False, encoding='utf-8-sig')
log('\n== Leakage-protocol contrast (reference: handcrafted-arm swing 0.0399) ==')
log(T.to_string(index=False))

# -- redundancy: deep embeddings vs conventional features --
rr = []
for mod in ['dwi', 'dce']:
    S = D[mod].values.astype(float)
    # PCA down to 50 dimensions (a correlation matrix on 512 dimensions is too slow and too noisy)
    from sklearn.decomposition import PCA
    Z = PCA(n_components=min(50, S.shape[0] - 1), random_state=42).fit_transform(
        StandardScaler().fit_transform(np.nan_to_num(S)))
    Rs = np.apply_along_axis(lambda v: pd.Series(v).rank().values, 0, Z)
    Rc = np.apply_along_axis(lambda v: pd.Series(np.nan_to_num(v)).rank().values, 0, Cv)
    Rs = (Rs - Rs.mean(0)) / (Rs.std(0) + 1e-12)
    Rc = (Rc - Rc.mean(0)) / (Rc.std(0) + 1e-12)
    RHO = np.abs(Rs.T @ Rc) / len(CASES)
    mx = np.nanmax(RHO, axis=1)
    rr.append(dict(modality=mod, n_pc=Z.shape[1],
                   n_pc_rho_ge_090=int((mx >= 0.90).sum()),
                   n_pc_rho_ge_080=int((mx >= 0.80).sum()),
                   max_rho=round(float(np.nanmax(mx)), 4),
                   median_rho=round(float(np.nanmedian(mx)), 4)))
    log('  %s PCA50: max|rho|=%.3f  median %.3f  | >=0.90: %d  | >=0.80: %d'
        % (mod, np.nanmax(mx), np.nanmedian(mx), (mx >= 0.90).sum(), (mx >= 0.80).sum()))

R = pd.DataFrame(rr)
R.to_csv(os.path.join(OUT, '401_deep_redundancy.csv'), index=False, encoding='utf-8-sig')
log('\n== Deep embeddings vs conventional features: redundancy ==')
log(R.to_string(index=False))
log('\n-> 401_deep_protocol.csv / 401_deep_redundancy.csv')
