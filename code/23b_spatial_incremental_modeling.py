# -*- coding: utf-8 -*-
"""
23b_spatial_incremental_modeling.py  —  Step 3b: incremental value of the spatial features

Feature pools (anything with a missing rate > 20% is excluded, i.e. the three
DWI fractal_dim features are dropped):
  P1 = "novel pool": new spatial features with max|rho| (vs 856 radiomics features) < 0.80
  P2 = all spatial features (comparison pool)

Models (LASSO-Cox throughout, l1_ratio=1):
  M0          : 11 clinical variables
  M4c         : DCE_H texture (107) + clinical (11)   <- reference model of the original paper
  M4c+P1      : + novel pool                          <- the core question
  M4c+P2      : + all spatial features
  Clinical+P1 : clinical + novel pool
  P1_only     : novel pool only

Two evaluation tracks (keep them strictly apart, never mix them):
  1) C_index_CV  -- reproduces the original project's run_lasso_cv: pick the best of
                    20 alphas by CV -> comparable with the original table value of 0.7092,
                    but it carries alpha-selection bias (optimistic)
  2) C_index_OOF -- nested CV (outer 5-fold / inner 5-fold to pick alpha) -> clean,
                    unbiased OOF estimate
                    + 95%CI from 1000 patient-level bootstrap resamples
                    + paired bootstrap Delta C-index versus M4c with a two-sided p

Input:  data/features/habitat_spatial_features_k3.csv
        data/features/dce_pyradiomics_k3.csv
        results/merged_data_corrected.csv
        results/models/spatial_redundancy_summary_k3.csv   (source of the P1 list)
Output: results/models/model_comparison_spatial_k3.csv
        results/models/model_comparison_spatial_k3.md
"""
import os, warnings, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

warnings.filterwarnings('ignore')

OUT = os.path.join(BASE, 'results', 'models')
os.makedirs(OUT, exist_ok=True)

CLIN_VARS = ['NAC', 'N', 'Age', 'Grade', 'T', 'ALN', 'proRT', 'ER', 'PR', 'Her2', 'Meno']
ALPHAS = np.logspace(-1.5, 0, 20)
MAX_MISSING = 0.20
N_BOOT = 1000
SEED = 42


def make_surv(df):
    return np.array([(bool(e), float(t)) for e, t in zip(df['DFS'], df['Time'])],
                    dtype=[('event', 'bool'), ('time', 'float')])


def run_lasso_cv(X, y, alphas=ALPHAS):
    """Reproduce the original project's protocol: best of 20 alphas by CV.
    Returns (best_c, best_alpha, selected_series)."""
    Xs = X.values
    kf = KFold(5, shuffle=True, random_state=SEED)
    cv = {a: [] for a in alphas}
    for tr, te in kf.split(Xs):
        s = StandardScaler()
        Xtr = s.fit_transform(Xs[tr]); Xte = s.transform(Xs[te])
        try:
            cph = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=alphas, max_iter=10000).fit(Xtr, y[tr])
            for a in alphas:
                try:
                    p = cph.predict(Xte, alpha=a)
                    cv[a].append(concordance_index_censored(y['event'][te], y['time'][te], p)[0])
                except Exception:
                    cv[a].append(0.5)
        except Exception:
            for a in alphas:
                cv[a].append(0.5)
    cm = {a: float(np.mean(v)) for a, v in cv.items()}
    ba = max(cm, key=cm.get)
    if all(v <= 0.5 for v in cm.values()):
        return 0.5, 0.1, pd.Series(dtype=float)
    s = StandardScaler(); Xf = s.fit_transform(Xs)
    cph = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=10000).fit(Xf, y)
    coef = cph.coef_.flatten()
    return cm[ba], ba, pd.Series(coef, index=X.columns)[coef != 0].sort_values(key=np.abs, ascending=False)


def nested_oof(X, y, alphas=ALPHAS, seed=SEED):
    """Nested CV: outer 5-fold hold-out, inner 5-fold to pick alpha -> clean OOF risk score."""
    Xs = X.values
    n = len(Xs)
    oof = np.full(n, np.nan)
    outer = KFold(5, shuffle=True, random_state=seed)
    for tr, te in outer.split(Xs):
        s = StandardScaler()
        Xtr = s.fit_transform(Xs[tr]); Xte = s.transform(Xs[te])
        inner = KFold(5, shuffle=True, random_state=seed + 1)
        sc = {a: [] for a in alphas}
        for itr, ite in inner.split(Xtr):
            si = StandardScaler()
            Xi = si.fit_transform(Xtr[itr]); Xv = si.transform(Xtr[ite])
            try:
                c = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=alphas, max_iter=10000).fit(Xi, y[tr][itr])
                for a in alphas:
                    try:
                        p = c.predict(Xv, alpha=a)
                        sc[a].append(concordance_index_censored(y[tr]['event'][ite],
                                                               y[tr]['time'][ite], p)[0])
                    except Exception:
                        sc[a].append(0.5)
            except Exception:
                for a in alphas:
                    sc[a].append(0.5)
        ba = max({a: float(np.mean(v)) for a, v in sc.items()}, key=lambda a: np.mean(sc[a]))
        try:
            c = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=10000).fit(Xtr, y[tr])
            oof[te] = c.predict(Xte).flatten()
        except Exception:
            oof[te] = 0.0
    return oof


def boot_ci(ev, tm, risk, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(risk)
    cs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if ev[idx].sum() < 3:
            continue
        try:
            cs.append(concordance_index_censored(ev[idx], tm[idx], risk[idx])[0])
        except Exception:
            pass
    cs = np.array(cs)
    return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))) if len(cs) else (np.nan, np.nan)


def paired_boot(ev, tm, r1, r2, n_boot=N_BOOT, seed=SEED):
    """Delta C = C(r1) - C(r2), with patient-paired resampling."""
    rng = np.random.default_rng(seed + 7)
    n = len(r1)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if ev[idx].sum() < 3:
            continue
        try:
            a = concordance_index_censored(ev[idx], tm[idx], r1[idx])[0]
            b = concordance_index_censored(ev[idx], tm[idx], r2[idx])[0]
            ds.append(a - b)
        except Exception:
            pass
    ds = np.array(ds)
    if not len(ds):
        return np.nan, np.nan, np.nan, np.nan
    p = 2 * min((ds > 0).mean(), (ds < 0).mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(ds.mean()), float(min(p, 1.0))


def main():
    t0 = time.time()
    # ---------- Data ----------
    clin = pd.read_csv(os.path.join(BASE, '分析结果/merged_data_corrected.csv'), encoding='utf-8-sig')
    clin['Case_No.'] = clin['Case_No.'].astype(str)
    y = make_surv(clin)
    ev, tm = y['event'].astype(bool), y['time'].astype(float)

    # DCE_H texture (the M4c backbone)
    rad = pd.read_csv(os.path.join(BASE, 'data', 'features', 'dce_pyradiomics_k3.csv'))
    rad['Case_No.'] = rad['Case_No.'].astype(str)
    rf = [c for c in rad.columns if c not in ('Case_No.', 'Habitat')]
    radH = rad[rad.Habitat == 'habitat_H'].drop_duplicates('Case_No.').set_index('Case_No.')[rf]

    # Spatial features -> wide table
    sp = pd.read_csv(os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv'))
    sp['Case_No.'] = sp['Case_No.'].astype(str)
    a = sp[sp.modality == 'DWI'].set_index('Case_No.')[[c for c in sp.columns if c.startswith('dwi_')]]
    b = sp[sp.modality == 'DCE'].set_index('Case_No.')[[c for c in sp.columns if c.startswith('dce_')]]
    spw = pd.concat([a, b], axis=1)

    # Exclude features with a missing rate > 20%
    miss = spw.isna().mean()
    dropped = sorted(miss[miss > MAX_MISSING].index)
    spw = spw[miss[miss <= MAX_MISSING].index]

    # P1 = novel pool
    red = pd.read_csv(os.path.join(OUT, 'spatial_redundancy_summary_k3.csv'))
    novel = set(red[red.max_abs_r < 0.80].feature)
    P1 = [c for c in spw.columns if c in novel]
    P2 = list(spw.columns)
    print(f'Excluded {len(dropped)} features with >20% missing: {dropped}')
    print(f'P1 novel pool {len(P1)}  |  P2 comparison pool {len(P2)}')

    order = clin['Case_No.']
    S = spw.loc[order].reset_index(drop=True)
    R = radH.loc[order].reset_index(drop=True)
    C = clin[CLIN_VARS].reset_index(drop=True)

    models = [
        ('M0_Clinical',   '11 clinical variables',                 pd.concat([C], axis=1)),
        ('M4c_Original',  'DCE_H texture + clinical',             pd.concat([C, R.add_prefix('DCEH_')], axis=1)),
        ('M4c+P1',        'DCE_H texture + clinical + novel spatial pool', pd.concat([C, R.add_prefix('DCEH_'), S[P1]], axis=1)),
        ('M4c+P2',        'DCE_H texture + clinical + all spatial features', pd.concat([C, R.add_prefix('DCEH_'), S[P2]], axis=1)),
        ('Clinical+P1',   'clinical + novel spatial pool',        pd.concat([C, S[P1]], axis=1)),
        ('P1_only',       'novel spatial pool only',              pd.concat([S[P1]], axis=1)),
    ]

    rows, oofs = [], {}
    for name, desc, X in models:
        X = X.fillna(0).astype(float)
        t1 = time.time()
        c_cv, alpha, sel = run_lasso_cv(X, y)
        oof = nested_oof(X, y)
        c_oof = concordance_index_censored(ev, tm, oof)[0]
        lo, hi = boot_ci(ev, tm, oof)
        try:
            auc3 = float(cumulative_dynamic_auc(y, y, oof, times=[36])[0][0])
            auc5 = float(cumulative_dynamic_auc(y, y, oof, times=[60])[0][0])
        except Exception:
            auc3 = auc5 = np.nan
        oofs[name] = oof
        rows.append(dict(Model=name, Desc=desc, Features_in=X.shape[1],
                         Features_selected=len(sel),
                         C_index_CV=round(c_cv, 4), alpha=round(float(alpha), 4),
                         C_index_OOF=round(c_oof, 4), OOF_CI_low=round(lo, 4), OOF_CI_high=round(hi, 4),
                         AUC_3yr_OOF=round(auc3, 4) if auc3 == auc3 else np.nan,
                         AUC_5yr_OOF=round(auc5, 4) if auc5 == auc5 else np.nan,
                         selected_top=','.join(sel.index[:6].tolist())))
        print(f'  [{name}] in={X.shape[1]} sel={len(sel)}  C_CV={c_cv:.4f}  '
              f'C_OOF={c_oof:.4f} [{lo:.3f},{hi:.3f}]  ({time.time()-t1:.0f}s)', flush=True)

    df = pd.DataFrame(rows)

    # Paired bootstrap against M4c
    ref = 'M4c_Original'
    for c1, c2 in [('OOF_CI_low', 'OOF_CI_high')]:
        pass
    dl, dh, dm, dp = [], [], [], []
    for name in df.Model:
        if name == ref:
            dl.append(np.nan); dh.append(np.nan); dm.append(np.nan); dp.append(np.nan); continue
        lo, hi, m, p = paired_boot(ev, tm, oofs[name], oofs[ref])
        dl.append(round(lo, 4)); dh.append(round(hi, 4)); dm.append(round(m, 4)); dp.append(round(p, 4))
    df['dC_vs_M4c'] = dm
    df['dC_CI_low'] = dl
    df['dC_CI_high'] = dh
    df['dC_p'] = dp

    df.to_csv(os.path.join(OUT, 'model_comparison_spatial_k3.csv'), index=False)

    # ---------- Report ----------
    L = ['# Incremental value of the spatial features (3b)\n']
    L.append(f'> Cohort {len(y)} cases / events {int(ev.sum())} ({ev.mean()*100:.1f}%) | LASSO-Cox (l1_ratio=1) | {len(dropped)} features excluded for >20% missing')
    L.append(f'> P1 novel pool = max|rho|<0.80 ({len(P1)} features) | P2 comparison pool = all ({len(P2)} features)\n')
    L.append('## Main table\n')
    L.append('| Model | Input features | LASSO selected | C-index (CV, original protocol) | **C-index (OOF)** | **OOF 95%CI** | AUC3yr | AUC5yr |')
    L.append('|---|---|---|---|---|---|---|---|')
    for _, r in df.iterrows():
        L.append(f"| {r.Model} | {r.Features_in} | {r.Features_selected} | {r.C_index_CV:.4f} | "
                 f"**{r.C_index_OOF:.4f}** | {r.OOF_CI_low:.3f}–{r.OOF_CI_high:.3f} | "
                 f"{r.AUC_3yr_OOF if pd.notna(r.AUC_3yr_OOF) else 'NA'} | "
                 f"{r.AUC_5yr_OOF if pd.notna(r.AUC_5yr_OOF) else 'NA'} |")
    L.append('')
    L.append('## Paired bootstrap Delta C-index versus M4c\n')
    L.append('| Model | ΔC vs M4c | 95%CI | two-sided p | Conclusion |')
    L.append('|---|---|---|---|---|')
    for _, r in df.iterrows():
        if r.Model == ref:
            L.append(f"| {r.Model} | — | — | — | reference |"); continue
        concl = ('significant improvement' if (pd.notna(r.dC_p) and r.dC_p < 0.05 and r.dC_vs_M4c > 0) else
                 'significant decrease' if (pd.notna(r.dC_p) and r.dC_p < 0.05 and r.dC_vs_M4c < 0) else
                 'no significant difference')
        L.append(f"| {r.Model} | {r.dC_vs_M4c:+.4f} | {r.dC_CI_low:+.3f}–{r.dC_CI_high:+.3f} | "
                 f"{r.dC_p:.4f} | {concl} |")
    L.append('')
    L.append('## Top LASSO-selected features (full-sample fit, for reference only)\n')
    for _, r in df.iterrows():
        L.append(f"- **{r.Model}** ({r.Features_selected} features): {r.selected_top}")
    L.append('')
    L.append('## Notes on the evaluation protocol\n')
    L.append('- `C_index_CV`: reproduces the original project protocol (best of 20 alphas by CV), '
             '**carries alpha-selection bias and is optimistic**;')
    L.append('  it is used to benchmark against the original `model_comparison_all.csv` (M4c=0.7092).')
    L.append('- `C_index_OOF`: pooled OOF estimate from nested CV (outer 5-fold hold-out / inner 5-fold '
             'alpha selection), **unbiased**, and the basis for the main conclusions.')
    L.append('- Bootstrap 95%CI: 1000 patient-level resamples (OOF risk scores).')
    L.append('- Delta C: paired bootstrap (difference in C-index between the two models on the same '
             'resample), two-sided p.')
    with open(os.path.join(OUT, 'model_comparison_spatial_k3.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))

    print('\n=== Results ===')
    print(df[['Model', 'Features_in', 'Features_selected', 'C_index_CV',
              'C_index_OOF', 'OOF_CI_low', 'OOF_CI_high', 'dC_vs_M4c', 'dC_p']].to_string(index=False))
    print(f'\n-> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
