# -*- coding: utf-8 -*-
"""
22_univariate_spatial_cox_k3.py  —  Step 2: univariate survival screening of spatial-structure features (K=3)

A univariate Cox model is fitted for each of the 84 spatial features
(DWI 42 + DCE 42):
  - HR expressed per 1 SD (z-score) so that features are comparable
  - Wald P, BH-FDR correction (84 tests)
  - per-feature C-index, median-split log-rank P
  - NaN handling: available-case (listwise) analysis per feature, reporting the actual n
    (DWI fractal_dim is missing in roughly 70% of cases, so its n is markedly
     smaller; keep this in mind when interpreting the results)

Input:  data/features/habitat_spatial_features_k3.csv
        results/merged_data_corrected.csv  (Case_No., Time, DFS)
Output: results/models/univariate_spatial_cox_k3.csv
        results/models/univariate_spatial_summary_k3.md
"""
import os, warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

warnings.filterwarnings('ignore')

FEAT_CSV = os.path.join(BASE, 'data', 'features', 'habitat_spatial_features_k3.csv')
CLIN_CSV = os.path.join(BASE, '分析结果', 'merged_data_corrected.csv')
OUT_DIR = os.path.join(BASE, 'results', 'models')
OUT_CSV = os.path.join(OUT_DIR, 'univariate_spatial_cox_k3.csv')
OUT_MD = os.path.join(OUT_DIR, 'univariate_spatial_summary_k3.md')

os.makedirs(OUT_DIR, exist_ok=True)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main():
    df = pd.read_csv(FEAT_CSV)
    clin = pd.read_csv(CLIN_CSV, encoding='utf-8-sig')
    clin = clin[['Case_No.', 'Time', 'DFS']].copy()
    clin['Case_No.'] = clin['Case_No.'].astype(str)

    # Long -> wide table (Case_No. x 84 features)
    # Note: feature names already carry the modality prefix (dwi_/dce_), and unstack
    # would add another layer -> columns must be selected per modality and concatenated
    f_dwi = [c for c in df.columns if c.startswith('dwi_')]
    f_dce = [c for c in df.columns if c.startswith('dce_')]
    a = df[df.modality == 'DWI'].set_index('Case_No.')[f_dwi]
    b = df[df.modality == 'DCE'].set_index('Case_No.')[f_dce]
    wide = pd.concat([a, b], axis=1).reset_index()
    wide['Case_No.'] = wide['Case_No.'].astype(str)

    dat = wide.merge(clin, on='Case_No.', how='inner')
    feats = [c for c in wide.columns if c != 'Case_No.']

    print(f'Merged: {len(dat)} cases (mask {len(wide)} / clinical {len(clin)})')
    print(f'Events: {int(dat.DFS.sum())}/{len(dat)} ({dat.DFS.mean()*100:.1f}%)')
    print(f'Follow-up: median {dat.Time.median():.0f} months, IQR [{dat.Time.quantile(.25):.0f},{dat.Time.quantile(.75):.0f}]')
    print(f'Features: {len(feats)}')

    rows = []
    for f in feats:
        sub = dat[[f, 'Time', 'DFS']].dropna()
        n, n_ev = len(sub), int(sub.DFS.sum())
        if n < 30 or n_ev < 10:
            rows.append(dict(feature=f, n=n, n_events=n_ev, note='insufficient sample/events'))
            continue
        x = sub[f].astype(float)
        mu, sd = x.mean(), x.std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            rows.append(dict(feature=f, n=n, n_events=n_ev, note='zero variance'))
            continue
        tmp = pd.DataFrame({'x': (x - mu) / sd, 'T': sub.Time, 'E': sub.DFS})
        try:
            cph = CoxPHFitter()
            cph.fit(tmp, duration_col='T', event_col='E')
            s = cph.summary.loc['x']
            hr, lo, hi, p = s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], s['p']
            cidx = float(cph.concordance_index_)
        except Exception as e:
            rows.append(dict(feature=f, n=n, n_events=n_ev, note=f'Cox failed:{type(e).__name__}'))
            continue
        # Median-split log-rank
        med = x.median()
        g = x > med
        try:
            lr = logrank_test(sub.Time[g], sub.Time[~g], sub.DFS[g], sub.DFS[~g])
            lrp = float(lr.p_value)
        except Exception:
            lrp = np.nan
        rows.append(dict(feature=f, n=n, n_events=n_ev,
                         HR_perSD=round(hr, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                         P_wald=p, Cindex=round(cidx, 4), P_logrank=lrp,
                         mean=round(mu, 4), sd=round(sd, 4), note=''))

    res = pd.DataFrame(rows)
    ok = res['P_wald'].notna()
    res.loc[ok, 'FDR'] = bh_fdr(res.loc[ok, 'P_wald'].values)
    res = res.sort_values('P_wald', na_position='last').reset_index(drop=True)
    res.to_csv(OUT_CSV, index=False)
    print(f'\n-> {OUT_CSV}  ({len(res)} rows)')

    # ---------- Summary ----------
    L = []
    L.append('# Univariate Cox screening of spatial-structure features (K=3, DWI+DCE)\n')
    L.append(f'> Cohort {len(dat)} cases / events {int(dat.DFS.sum())} ({dat.DFS.mean()*100:.1f}%) / '
             f'median follow-up {dat.Time.median():.0f} months')
    L.append(f'> {len(feats)} features (DWI 42 + DCE 42), HR per 1 SD, FDR = BH correction\n')

    for tag, name in [('dwi', 'DWI'), ('dce', 'DCE'), ('', 'All')]:
        sub = res[res.feature.str.startswith(tag)] if tag else res
        p05 = sub[sub.P_wald < 0.05]
        f05 = sub[sub.FDR < 0.05]
        L.append(f'## {name}: {len(p05)}/{len(sub)} with P<0.05, {len(f05)} with FDR<0.05\n')

    L.append('## Features with univariate P<0.05 (ascending P)\n')
    L.append('| Feature | n | Events | HR(1SD) | 95%CI | P | FDR | C-index | log-rank P |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for _, r in res[res.P_wald < 0.05].iterrows():
        L.append(f"| {r.feature} | {int(r.n)} | {int(r.n_events)} | {r.HR_perSD:.3f} | "
                 f"{r.CI_low:.3f}–{r.CI_high:.3f} | {r.P_wald:.4f} | {r.FDR:.4f} | "
                 f"{r.Cindex:.3f} | {r.P_logrank:.4f} |")
    L.append('')

    L.append('## Features with FDR<0.05 (robust)\n')
    f05 = res[res.FDR < 0.05]
    if len(f05) == 0:
        L.append('**None** — no feature survives multiple-testing correction, i.e. no single '
                 'spatial feature carries a robust independent prognostic signal.\n')
    else:
        L.append('| Feature | n | HR(1SD) | 95%CI | P | FDR | C-index |')
        L.append('|---|---|---|---|---|---|---|')
        for _, r in f05.iterrows():
            L.append(f"| {r.feature} | {int(r.n)} | {r.HR_perSD:.3f} | {r.CI_low:.3f}–{r.CI_high:.3f} "
                     f"| {r.P_wald:.4f} | {r.FDR:.4f} | {r.Cindex:.3f} |")
        L.append('')

    L.append('## Degraded features (smaller sample size, interpret with caution)\n')
    deg = res[res.n < 350]
    if len(deg):
        L.append('| Feature | Available n | Missing rate | P |')
        L.append('|---|---|---|---|')
        for _, r in deg.iterrows():
            L.append(f"| {r.feature} | {int(r.n)} | {(1-r.n/len(dat))*100:.0f}% | "
                     f"{r.P_wald if pd.notna(r.P_wald) else 'NA'} |")
    else:
        L.append('None (every feature has an available n ≥ 350)')
    L.append('')

    with open(OUT_MD, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'-> {OUT_MD}')

    # Console overview
    print('\n=== Features with P<0.05 ===')
    v = res[res.P_wald < 0.05][['feature', 'n', 'HR_perSD', 'CI_low', 'CI_high', 'P_wald', 'FDR', 'Cindex']]
    print(v.to_string(index=False) if len(v) else 'None')
    print(f'\nFDR<0.05: {len(res[res.FDR<0.05])} features')


if __name__ == '__main__':
    main()
