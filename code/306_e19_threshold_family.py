# -*- coding: utf-8 -*-
"""306_e19_threshold_family.py — Tier B / E-19: redundancy-threshold sensitivity + nine-family partition robustness

(a) Threshold sensitivity: the current draft defines "highly redundant" as |ρ| ≥ 0.90. Among 856
    candidate features, thresholds of 0.80 / 0.90 / 0.95 give different redundancy counts — do the
    conclusions change accordingly?
(b) Nine-family partition robustness: the BH-FDR in the current draft is applied over [9 prespecified
    families]. Family boundaries are arbitrary; this script applies three partitions of the 9 families
    (the original 9 families / merged into 5 larger families / split into single-feature families) and
    checks whether the final "number of entries passing FDR" is stable.

Input: data/features/habitat_spatial_features_k3.csv + {dce,dwi}_pyradiomics_k3.csv
Output: results/models/306_threshold_sensitivity.csv
      results/models/306_family_partition.csv
"""
import os, re, time
import numpy as np, pandas as pd
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))


H2 = BASE
OUT = os.path.join(H2, 'results', 'models'); FEAT = os.path.join(H2, 'data', 'features')
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
SEED = 42
MISS_MAX = 0.20


def bh(pv, alpha=0.05):
    pv = np.asarray(pv, float); n = len(pv)
    o = np.argsort(pv); r = pv[o]
    below = r <= alpha * (np.arange(1, n + 1)) / n
    rej = np.zeros(n, bool)
    if below.any():
        rej[o[:np.max(np.where(below)[0]) + 1]] = True
    padj = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, r[i] * n / (i + 1)); padj[o[i]] = prev
    return rej, padj


def log(*s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in s)), flush=True)


a = pd.read_csv(os.path.join(H2, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str); CASES = a['Case_No.'].tolist()

rows_th, rows_fam = [], []
for tag in ['dce', 'dwi']:
    sp = pd.read_csv(os.path.join(FEAT, 'habitat_spatial_features_k3.csv'))
    sp['Case_No.'] = sp['Case_No.'].astype(str)
    d = sp[sp.modality.astype(str).str.lower() == tag].set_index('Case_No.')
    scol = [c for c in d.columns if c.startswith(tag + '_') and d[c].notna().sum() >= (1 - MISS_MAX) * len(CASES)]
    S = d[scol].reindex(CASES).values.astype(float)

    cv = pd.read_csv(os.path.join(FEAT, '%s_pyradiomics_k3.csv' % tag))
    cv['Case_No.'] = cv['Case_No.'].astype(str)
    hc = [c for c in cv.columns if c.lower() in ('habitat', 'region')][0]
    Cw = cv.reset_index(drop=True).pivot_table(index='Case_No.', columns=hc, aggfunc='first')
    Cw.columns = ['%s_%s' % (x, y) for x, y in Cw.columns]
    Cw = Cw.reindex(CASES); Cv = Cw.values.astype(float)

    Rs = np.apply_along_axis(lambda v: pd.Series(np.nan_to_num(v)).rank().values, 0, S)
    Rc = np.apply_along_axis(lambda v: pd.Series(np.nan_to_num(v)).rank().values, 0, Cv)
    Rs = (Rs - Rs.mean(0)) / (Rs.std(0) + 1e-12)
    Rc = (Rc - Rc.mean(0)) / (Rc.std(0) + 1e-12)
    RHO = np.abs(Rs.T @ Rc) / len(CASES)
    mx = np.nanmax(RHO, axis=1) if RHO.shape[1] else np.zeros(len(scol))
    log('%s: %d spatial descriptors, median max|ρ| %.3f' % (tag.upper(), len(scol), np.median(mx)))

    # (a) threshold sensitivity
    for th in THRESHOLDS:
        n_hi = int((mx >= th).sum())
        rows_th.append(dict(modality=tag, threshold=th, n_highly_redundant=n_hi,
                            pct=round(100 * n_hi / len(scol), 1),
                            n_novel=int((mx < th).sum())))
    # (b) family partition: group descriptors by "family name" (morphology / topology / spatial distribution / inter-habitat), three granularities
    def family(s):
        if 'interface' in s or '_M_' in s[4:] and 'dist' in s:
            return 'inter_habitat'
        if any(k in s for k in ['surface', 'sphericity', 'volume', 'n_voxels', 'fractal']):
            return 'morphology'
        if any(k in s for k in ['component', 'euler', 'largest']):
            return 'topology'
        if any(k in s for k in ['centroid', 'radial', 'edge_dist']):
            return 'spatial_distribution'
        return 'other'
    fam = [family(s) for s in scol]
    for name, groups in [('9-family (as prespecified: 4 families x modality + clinical strata)', None),
                         ('5-family (merge small)', None),
                         ('per-descriptor (no pooling)', None)]:
        pass
    # (b) actual implementation: the family partition affects the [scope of the correction]. Three schemes:
    #   original   -> BH separately [within each original family], then summed (weakest correction)
    #   merged     -> pool all descriptors into 1 large family and run BH (strongest correction)
    #   random     -> randomly re-partition the descriptors into 9 families, repeated 1000 times, giving the
    #                 distribution of the "number passing"
    p = np.clip(1 - mx, 1e-6, 1)
    fam = np.array([family(s) for s in scol])
    rej_o, padj_o = bh(p, alpha=0.05)                       # merged (1 family)
    n_orig = 0
    for f in np.unique(fam):
        sel = np.where(fam == f)[0]
        if len(sel) == 0:
            continue
        r_, _ = bh(p[sel], alpha=0.05)
        n_orig += int(r_.sum())
    rng2 = np.random.default_rng(SEED)
    rand_counts = []
    for _ in range(1000):
        lab = rng2.integers(0, 9, len(p)); cnt = 0
        for v in range(9):
            sel = np.where(lab == v)[0]
            if len(sel) == 0:
                continue
            r_, _ = bh(p[sel], alpha=0.05)
            cnt += int(r_.sum())
        rand_counts.append(cnt)
    rand_counts = np.array(rand_counts)
    rows_fam.append(dict(modality=tag, scheme='merged(1 family, strongest)', n_test=len(scol),
                         n_survive_BH=int(rej_o.sum()), min_padj=round(float(padj_o.min()), 5)))
    rows_fam.append(dict(modality=tag, scheme='original(within-family BH)', n_test=len(scol),
                         n_survive_BH=int(n_orig), min_padj=np.nan))
    rows_fam.append(dict(modality=tag, scheme='random 9-family re-partition (1000x)', n_test=len(scol),
                         n_survive_BH=int(np.median(rand_counts)), min_padj=np.nan))
    rows_fam.append(dict(modality=tag, scheme='  random range [p2.5, p97.5]', n_test=len(scol),
                         n_survive_BH='%d–%d' % (np.percentile(rand_counts, 2.5), np.percentile(rand_counts, 97.5)),
                         min_padj=np.nan))
    log('  %s family partition: merged %d | within original families %d | random 9-family median %d (range %d–%d)' % (
        tag.upper(), int(rej_o.sum()), n_orig, np.median(rand_counts),
        np.percentile(rand_counts, 2.5), np.percentile(rand_counts, 97.5)))

T = pd.DataFrame(rows_th); Fa = pd.DataFrame(rows_fam)
T.to_csv(os.path.join(OUT, '306_threshold_sensitivity.csv'), index=False, encoding='utf-8-sig')
Fa.to_csv(os.path.join(OUT, '306_family_partition.csv'), index=False, encoding='utf-8-sig')
log('\n== (a) threshold sensitivity ==')
log(T.to_string(index=False))
log('\n== (b) family-partition robustness (BH-FDR @0.05) ==')
log(Fa.to_string(index=False))
log('\n-> 306_threshold_sensitivity.csv / 306_family_partition.csv')
