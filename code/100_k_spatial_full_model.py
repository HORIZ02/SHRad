import tempfile
# -*- coding: utf-8 -*-
"""
100_k_spatial_full_model.py — rerun the incremental modelling on the [complete spatial habitat pool] at every K

Difference from 97 (which uses only 5 descriptor proxies): this script uses the full feature set produced by 99
  K=2: 25 / K=3: 42 / K=4: 62 / K=5: 85 / K=6: 111 features per modality
including: 11 single-habitat descriptors x K + 3 pairwise habitat-interface descriptors x C(K,2)

Design:
  base model = clinical 11 + the 107 PyRadiomics features of the highest-signal habitat at that K (same structure as paper model M4c)
  incremental model = base + spatial pool, with two pools:
      novel : max|Spearman| against the radiomics features of the same K < 0.80 (novel, the paper's P1 caliber)
      all   : all spatial features (the paper's P2 caliber)
  modality sets: DWI / DCE / DWI+DCE
  beyond the unsupervised criterion, per-seed paired dC + bootstrap CI

WARNING data quality: in 57/401 cases (the Center 2 subgroup of Center1) the DWI images were binarised to {0,1},
   so their DWI habitats cannot be ranked by signal and the texture features degenerate -> these cases are
   excluded from DWI-related analyses by default (--keepbad retains them).

Stages: model | fig | all
"""
import os, sys, warnings, importlib.util, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon
warnings.filterwarnings('ignore')

BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

FEAT = os.path.join(BASE, 'data', 'features')
RES = os.path.join(BASE, 'results', 'models')
OUT = os.path.join(BASE, 'results', 'habitat_spatial_viz')
FIGJ = os.path.join(BASE, 'results', 'figures_journal')
MIN_FEAT = os.path.join(BASE, 'data', 'file_mapping.csv')
KS = [2, 3, 4, 5, 6]
MODSETS = [('DWI', ['dwi']), ('DCE', ['dce']), ('DWI+DCE', ['dwi', 'dce'])]
COL = {'DWI': '#3498DB', 'DCE': '#E74C3C', 'DWI+DCE': '#8E44AD'}
SEEDS = [42] + list(range(19))
RHO_CUT = 0.80
SUB_SINGLE = ['n_voxels', 'n_components', 'largest_frac', 'euler', 'surface', 'sphericity',
              'fractal_dim', 'centroid_dist', 'radial_mean', 'radial_sd', 'edge_dist_mean']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'; rcParams['font.serif'] = ['Times New Roman']
rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 7.2,
                 'axes.linewidth': 0.9, 'legend.frameon': False, 'savefig.facecolor': 'white',
                 'figure.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

# font compliance: $...$ mathtext must use the same roman family as the body text (Times New Roman)
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.sf'] = 'Times New Roman'
rcParams['mathtext.tt'] = 'Times New Roman'
rcParams['mathtext.cal'] = 'Times New Roman:italic'
rcParams['mathtext.default'] = 'it'


def save(fig, name, dpi=600):
    """Journal specification: output width exactly = 3.35 in single column or 6.90 in double column; height <= 8.60 in; includes a glyph audit"""
    from matplotlib.transforms import Bbox
    # ---- glyph audit: warn about any visible text whose glyphs are missing from Times New Roman ----
    from matplotlib import font_manager as _fm
    from matplotlib.ft2font import FT2Font as _FT
    try:
        _cm = _FT(_fm.findfont(_fm.FontProperties(family='Times New Roman'))).get_charmap()
        _bad = {}
        for _t in fig.findobj(match=lambda o: hasattr(o, 'get_text')):
            try: _s = _t.get_text()
            except Exception: continue
            if not _s: continue
            for _c in _s:
                if ord(_c) >= 128 and ord(_c) not in _cm:
                    _bad.setdefault(_c, 0)
                    _bad[_c] += 1
        if _bad:
            print(f'  !! missing glyphs {name}: ' + ', '.join(f'{c!r}(U+{ord(c):04X})x{n}' for c, n in _bad.items()), flush=True)
    except Exception:
        pass
    def _tight():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())   # note: matplotlib already returns inches
        return bb, bb.width, bb.height
    bb, w, h = _tight()
    target = 3.35 if w <= 4.81 else 6.90    # cut-off = sqrt(3.35x6.90); the old value 3.36 padded 3.4in content out to 6.90 and left a wide white band    # snap to the nearest: <=3.36 -> single column, otherwise -> double column (intermediate widths are not allowed)
    for _ in range(5):                       # iterative narrowing (text size is fixed, so several passes may be needed)
        if w <= target + 1e-6: break
        fig.set_size_inches(*[x * (target / w) for x in fig.get_size_inches()])
        bb, w, h = _tight()
    if h > 8.60:
        s = 8.60 / h
        fig.set_size_inches(*[x * s for x in fig.get_size_inches()])
        bb, w, h = _tight()
        if w > target + 1e-6:
            s2 = target / w
            fig.set_size_inches(*[x * s2 for x in fig.get_size_inches()])
            bb, w, h = _tight()
    pad = max((target - w) / 2.0, 0.0)
    PADY = 0.05                                   # 0.05 in above and below so titles/tick labels never touch the edge
    nb = Bbox.from_extents(bb.x0 - pad, bb.y0 - PADY, bb.x1 + pad, bb.y1 + PADY)
    if not (1.0 < (nb.width) < 7.0 and 0.8 < nb.height < 9.0):
        raise RuntimeError(f'abnormal figure extent {name}: {nb.width:.2f} x {nb.height:.2f} in')
    dests = []
    for g in ('OUT', 'FIG', 'FIGJ'):
        v = globals().get(g)
        if v and v not in dests:
            dests.append(v)
    for di, d in enumerate(dests):
        os.makedirs(d, exist_ok=True)
        for ext, kw in [('png', dict(dpi=dpi)), ('tiff', dict(dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})),
                        ('pdf', {}), ('svg', {})]:
            if ext == 'svg' and di > 0:
                continue
            fig.savefig(os.path.join(d, f'{name}.{ext}'), bbox_inches=nb, **kw)
    plt.close(fig)
    print(f'  OK {name}', flush=True)


# ---------------- cohort ----------------
a = pd.read_csv(os.path.join(BASE, '分析结果/merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
y = m.make_surv(a)
ev, tm = y['event'].astype(bool), y['time'].astype(float)
CASES = a['Case_No.'].tolist()
C11 = a[m.CLIN_VARS].reset_index(drop=True)
SCAN = os.path.join(tempfile.gettempdir(), 'img_uniq_scan.csv')
BAD_DWI = set()
if os.path.exists(SCAN):
    _s = pd.read_csv(SCAN)
    BAD_DWI = set(_s.loc[_s.DWI_nuniq <= 3, 'case'].astype(str))
print('cohort %d cases / %d events; binarised DWI cases %d' % (len(a), int(ev.sum()), len(BAD_DWI)), flush=True)


# ---------------- feature loading ----------------
def _rename_k3(cols):
    """Normalise the L/M/H naming of the K=3 reference file to S0 (highest)/S1/S2"""
    mp = {'_L_': '_S2_', '_M_': '_S1_', '_H_': '_S0_'}
    out = []
    for c in cols:
        d = c
        for k_, v_ in mp.items():
            d = d.replace(k_, v_)
        out.append(d)
    return out


def load_spatial(K, mods, vcases):
    p = os.path.join(FEAT, f'habitat_spatial_features_k{K}.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, encoding='utf-8-sig')
    d['Case_No.'] = d['Case_No.'].astype(str)
    if K == 3:                                     # K=3 is the existing reference file (L/M/H naming)
        d = d.rename(columns=dict(zip(d.columns, _rename_k3(d.columns))))
    parts = []
    cols_all = []
    for mod in mods:
        sub = d[d.modality.astype(str).str.upper() == mod.upper()]
        if len(sub) == 0:
            return None
        cc = [c for c in sub.columns if c.startswith(f'{mod}_')]
        w = sub.set_index('Case_No.')[cc].reindex(vcases)      # <- coverage is judged within the valid case set only
        ref = next((c for c in cc if c.startswith(f'{mod}_S0_')), cc[0])
        cov = float(w[ref].notna().mean())
        if cov < 0.95:
            print(f'  !! K={K} {mod} spatial feature coverage {cov:.1%} ({ref}, n={len(vcases)}), skipped', flush=True)
            return None
        parts.append(w.reset_index(drop=True))
        cols_all += cc
    return pd.concat(parts, axis=1), cols_all


def load_rad(K, mods, vcases):
    parts = []
    for mod in mods:
        p = os.path.join(FEAT, f'k{K}_{mod}_pyradiomics.csv')
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, encoding='utf-8-sig')
        d['Case_No.'] = d['Case_No.'].astype(str)
        d = d[d['rank'] == 0]
        fc = [c for c in d.columns if c.startswith('original_')]
        w = d.set_index('Case_No.')[fc].add_prefix(f'{mod.upper()}_').reindex(vcases)
        if float(w.iloc[:, 0].notna().mean()) < 0.95:
            return None
        parts.append(w.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def novel_cols(SP, R):
    """Spatial features with max|rho| < RHO_CUT against the same-modality radiomics features (the novel pool)"""
    keep = []
    for c in SP.columns:
        mod = c.split('_')[0]
        rc = [x for x in R.columns if x.startswith(mod.upper() + '_')]
        if not rc:
            keep.append(c); continue
        mx = 0.0
        v = SP[c].values.astype(float)
        for r in rc:
            w = R[r].values.astype(float)
            ok = np.isfinite(v) & np.isfinite(w)
            if ok.sum() < 30:
                continue
            rho = abs(spearmanr(v[ok], w[ok]).correlation)
            if np.isfinite(rho) and rho > mx:
                mx = rho
        if mx < RHO_CUT:
            keep.append(c)
    return keep


# ---------------- modelling ----------------
def run_model(keep_bad=False, gapmin=0, tag=''):
    bad = set() if keep_bad else BAD_DWI
    if gapmin:
        _g = pd.read_csv(os.path.join(RES, 'gap_statistic_b40v3000.csv'), encoding='utf-8-sig')
        gap_ok = set(_g[(_g.modality == 'DWI') & (_g.gapK >= gapmin)]['Case_No.'].astype(str))
        print(f'  [gapmin={gapmin}] identifiable DWI habitat structure (gapK>={gapmin}) {len(gap_ok)}/{len(_g)//2} cases')
    rows, red_rows = [], []
    Xc = C11.fillna(0).astype(float)
    for sd in SEEDS:
        o = m.nested_oof(Xc, y, seed=sd)
        rows.append({'K': 0, 'modset': 'Clinical', 'pool': '-', 'seed': sd,
                     'C': float(m.concordance_index_censored(ev, tm, o)[0])})
    print('  clinical 11 done', flush=True)
    for ms, mods in MODSETS:
        valid = np.ones(len(a), dtype=bool)
        if 'dwi' in mods and len(bad):
            valid = ~a['Case_No.'].isin(bad).values
        if gapmin and 'dwi' in mods:
            valid = valid & a['Case_No.'].astype(str).isin(gap_ok).values
        vcases = [c for c, v in zip(CASES, valid) if v]
        ym = y[valid]
        evm, tmm = ym['event'].astype(bool), ym['time'].astype(float)
        Cv = C11.loc[valid].reset_index(drop=True).fillna(0).astype(float)
        for sd in SEEDS:                                   # clinical reference inside this subset
            o = m.nested_oof(Cv, ym, seed=sd)
            rows.append({'K': 0, 'modset': ms + ' [clin ref]', 'pool': '-', 'seed': sd,
                         'C': float(m.concordance_index_censored(evm, tmm, o)[0])})
        for K in KS:
            R = load_rad(K, mods, vcases)
            SPx = load_spatial(K, mods, vcases)
            if R is None or SPx is None:
                continue
            SP, spcols = SPx
            assert SP.shape[0] == len(vcases) and R.shape[0] == len(vcases), 'row counts do not match'
            nov = novel_cols(SP, R)
            _miss = SP[spcols].isna().mean()               # the paper's MAX_MISSING caliber
            spok = [c for c in spcols if _miss.get(c, 1) <= 0.20]
            nov = [c for c in nov if c in spok]
            red_rows.append({'modset': ms, 'K': K, 'n_cases': len(vcases),
                             'n_spatial_all': len(spok), 'n_spatial_raw': len(spcols),
                             'n_novel': len(nov), 'n_redundant': len(spok) - len(nov)})
            print(f'  {ms} K={K}: radiomics {R.shape[1]} spatial {len(spok)} (raw {len(spcols)}, novel {len(nov)}) cases {len(vcases)}',
                  flush=True)
            sets = {'-': None, 'novel': nov, 'all': spok}
            for pool, pc in sets.items():
                X = pd.concat([Cv, R], axis=1)
                if pc:
                    X = pd.concat([X, SP[pc]], axis=1)
                X = X.fillna(0).astype(float)
                assert X.shape[0] == len(vcases), f'{ms} K={K} {pool} row count mismatch {X.shape}'
                for sd in SEEDS:
                    o = m.nested_oof(X, ym, seed=sd)
                    rows.append({'K': K, 'modset': ms, 'pool': pool, 'seed': sd,
                                 'C': float(m.concordance_index_censored(evm, tmm, o)[0])})
    R_ = pd.DataFrame(rows)
    R_.to_csv(os.path.join(RES, 'k_spatial_full_perseed.csv'), index=False, encoding='utf-8-sig')
    agg = R_.groupby(['K', 'modset', 'pool'])['C'].agg(['mean', 'std', 'min', 'max']).reset_index()
    agg.to_csv(os.path.join(RES, f'k_spatial_full{tag}.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(red_rows).to_csv(os.path.join(RES, f'k_spatial_redundancy{tag}.csv'), index=False, encoding='utf-8-sig')
    print('\n' + agg.round(4).to_string(index=False))
    # per-seed paired dC
    out = []
    for ms, _ in MODSETS:
        for K in KS:
            base = R_[(R_.K == K) & (R_.modset == ms) & (R_.pool == '-')].set_index('seed')['C']
            if len(base) == 0:
                continue
            for pool in ['novel', 'all']:
                q = R_[(R_.K == K) & (R_.modset == ms) & (R_.pool == pool)].set_index('seed')['C']
                j = pd.concat([base.rename('b'), q.rename('p')], axis=1).dropna()
                if len(j) < 3:
                    continue
                d = (j.p - j.b).values
                rng = np.random.default_rng(42)
                bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(2000)])
                out.append({'modset': ms, 'K': K, 'pool': pool,
                            'C_base': round(float(j.b.mean()), 4), 'C_add': round(float(j.p.mean()), 4),
                            'dC': round(float(d.mean()), 5), 'dC_sd': round(float(d.std(ddof=1)), 5),
                            'CI_lo': round(float(np.percentile(bs, 2.5)), 5),
                            'CI_hi': round(float(np.percentile(bs, 97.5)), 5),
                            't_p': round(float(ttest_rel(j.p, j.b).pvalue), 4),
                            'wilcoxon_p': round(float(wilcoxon(j.p, j.b).pvalue), 4)})
    D = pd.DataFrame(out)
    D.to_csv(os.path.join(RES, f'k_spatial_full_delta{tag}.csv'), index=False, encoding='utf-8-sig')
    print('\n=== Increment of the complete spatial pool (per-seed paired, n=%d) ===' % len(SEEDS))
    print(D.to_string(index=False))


# ---------------- figures ----------------
def fig_k():
    Dp = os.path.join(RES, 'k_spatial_full_delta.csv')
    if not os.path.exists(Dp):
        print('no delta results'); return
    D = pd.read_csv(Dp)
    RED = pd.read_csv(os.path.join(RES, 'k_spatial_redundancy.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for i, pool in enumerate(['novel', 'all']):
        ax = axes[i]
        ax.axhline(0, color='k', lw=0.8)
        d = D[D.pool == pool]
        if len(d):
            alld = pd.concat([d.CI_lo, d.CI_hi, d.dC])
            _lo, _hi = float(alld.min()), float(alld.max()); _pad = max(0.25 * (_hi - _lo), 1e-4)
            ax.set_ylim(_lo - _pad, _hi + 2.1 * _pad)
            for ms, _ in MODSETS:
                dd = d[d.modset == ms].sort_values('K')
                if len(dd) == 0:
                    continue
                ax.errorbar(dd.K, dd.dC, yerr=[dd.dC - dd.CI_lo, dd.CI_hi - dd.dC],
                            marker='o', ms=3.5, lw=1.4, capsize=2.2, color=COL[ms],
                            ecolor='#000000', elinewidth=0.7, label=ms)
            ax.set_xticks(KS); ax.legend(fontsize=6.6, loc='upper center', ncol=3,
                                         columnspacing=0.8, handlelength=1.4, borderaxespad=0.2)
        ax.set_xlabel('K', fontsize=8.5)
        ax.set_ylabel('ΔC-index (spatial added)', fontsize=8.5)
        ax.set_title('(%s) %s spatial pool' % ('ab'[i], pool), fontsize=9, pad=5)
    fig.tight_layout()
    save(fig, 'K7_k_spatial_increment')

    # redundancy structure: total / novel / redundant spatial features at each K
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    for ms, _ in MODSETS:
        d = RED[RED.modset == ms].sort_values('K')
        if len(d) == 0:
            continue
        ax.plot(d.K, d.n_novel / d.n_spatial_all, marker='o', ms=3.5, lw=1.4, color=COL[ms], label=ms)
    ax.set_xlabel('K', fontsize=8.5); ax.set_xticks(KS)
    ax.set_ylabel('Fraction of novel spatial features', fontsize=8.5)
    ax.legend(fontsize=6.8); ax.set_ylim(0, 1)
    ax.set_title('Novelty vs conventional radiomics', fontsize=9, pad=5)
    fig.tight_layout()
    save(fig, 'K8_k_spatial_novelty')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='all', choices=['model', 'fig', 'all'])
    ap.add_argument('--keepbad', action='store_true', help='retain the binarised DWI cases')
    ap.add_argument('--gapmin', type=int, default=0,
                    help='for DWI-related modality sets, include only cases with gapK >= this value (identifiable habitat structure)')
    ap.add_argument('--tag', default='', help='suffix for output file names')
    ar = ap.parse_args()
    if ar.stage in ('model', 'all'):
        run_model(keep_bad=ar.keepbad, gapmin=ar.gapmin, tag=ar.tag)
    if ar.stage in ('fig', 'all'):
        fig_k()
    print('\n[100 done]')
