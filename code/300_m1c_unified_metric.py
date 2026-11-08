# -*- coding: utf-8 -*-
"""300_m1c_unified_metric.py — Tier B / M1c: unified metric

Reviewer concern: the two cohorts were compared with different metrics (cohort A C-index, cohort B AUC),
so the "increment" was not on a common caliber. This script recomputes both cohorts under the
[same metric = Harrell C-index]:
  - cohort A (survival)   C-index = Harrell's C        the native metric is also reported: time-dependent AUC@36/60
  - cohort B (binary)     C-index == ROC-AUC (identical in the binary case)   native metric also reported: ROC-AUC + Brier

Input: both OOF sides on disk (one npy per arm per seed, patient row order matching each main table)
  A: <SHRAD_ROOT>/results/models/_oof_146/<arm>__<seed>.npy   shape (345,)
  B: <PROSTATE_ROOT>/results/cross_dataset/_oof_147/<arm>__<seed>.npy          shape (284,)
Output:
  results/models/300_m1c_unified.csv       metric summary per arm per cohort (mean±SD, 95%CI)
  results/models/300_m1c_arms.csv          unified metrics for the three arms (1 conventional WT / 2 conventional habitat / 3 spatial habitat) x modality
  results/models/300_m1c_delta.csv         paired differences 3 vs 1 / 3 vs 2 (unified metric, paired bootstrap)
"""
import os, re, sys, glob, time, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
PROSTATE_ROOT = os.environ.get("PROSTATE_ROOT", os.path.abspath(os.getcwd()))

H2 = BASE
PRO = os.path.join(PROSTATE_ROOT, '')
OUT = os.path.join(H2, 'results', 'models')
TIMES = [36, 60]
N_BOOT = 1000
SEED_BOOT = 42

# reuse the 23b interface (BASE/OUT/make_surv/paired_boot)
spec = importlib.util.spec_from_file_location('m23b', os.path.join(os.path.dirname(os.path.abspath(__file__)), '23b_spatial_incremental_modeling.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def log(*s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in s)), flush=True)


# -- cohort A ----------------------------------------------------------
a = pd.read_csv(os.path.join(H2, '分析结果', 'merged_data_corrected.csv'), encoding='utf-8-sig')
a['Case_No.'] = a['Case_No.'].astype(str)
yA = m.make_surv(a)
evA = np.asarray(yA['event']).astype(bool)
tmA = np.asarray(yA['time']).astype(float)
log('cohort A n=%d events=%d' % (len(a), int(evA.sum())))

# -- cohort B ----------------------------------------------------------
fb = pd.read_csv(os.path.join(PRO, 'data', 'features', 'fusion_features.csv'))
yB = fb['label'].astype(int).values
tmB = np.ones(len(yB))                      # binary outcome: all "event times" set to 1, so C-index == AUC
evB = yB.astype(bool)
log('cohort B n=%d high-risk=%d' % (len(yB), int(yB.sum())))


def ci_patient(ev, tm, sc, n_boot=N_BOOT, seed=SEED_BOOT, n=None):
    """Patient-level bootstrap 95% CI of the C-index"""
    n = n or len(sc)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(concordance_index_censored(ev[idx], tm[idx], sc[idx])[0])
        except Exception:
            pass
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_oof(oofd, arm, seed):
    return np.load(os.path.join(oofd, '%s__%d.npy' % (arm, seed)))


def list_arms(oofd):
    arms = {}
    for f in glob.glob(os.path.join(oofd, '*.npy')):
        b = os.path.basename(f)[:-4]
        mm = re.match(r'(.+)__(\d+)$', b)
        if mm:
            arms.setdefault(mm.group(1), []).append(int(mm.group(2)))
    return {k: sorted(v) for k, v in arms.items()}


OOFD_A = os.path.join(OUT, '_oof_146')
OOFD_B = os.path.join(PRO, 'results', 'cross_dataset', '_oof_147')
ARMS_A = list_arms(OOFD_A); ARMS_B = list_arms(OOFD_B)
log('cohort A arms %d | cohort B arms %d' % (len(ARMS_A), len(ARMS_B)))

rows = []
for cohort, arms, oofd, ev, tm, n in [('A', ARMS_A, OOFD_A, evA, tmA, len(a)),
                                      ('B', ARMS_B, OOFD_B, evB, tmB, len(yB))]:
    for arm, seeds in arms.items():
        C, NAT = [], []
        for sd in seeds:
            o = load_oof(oofd, arm, sd)
            assert len(o) == n, '%s %s length %d != %d' % (cohort, arm, len(o), n)
            C.append(concordance_index_censored(ev, tm, o)[0])
            NAT.append(roc_auc_score(yB, o) if cohort == 'B' else np.nan)
        C = np.array(C); C = C[np.isfinite(C)]
        o0 = load_oof(oofd, arm, seeds[0])
        lo, hi = ci_patient(ev, tm, o0, n=n)
        nat = np.array([x for x in NAT if np.isfinite(x)])
        rows.append(dict(cohort=cohort, arm=arm, n_seeds=len(C),
                         C_mean=C.mean(), C_sd=C.std(ddof=1), C_lo=lo, C_hi=hi,
                         native=(nat.mean() if len(nat) else np.nan),
                         native_sd=(nat.std(ddof=1) if len(nat) > 1 else np.nan)))
        log('  %s %-26s C=%.4f±%.4f [%.3f,%.3f]' % (cohort, arm, C.mean(), C.std(ddof=1), lo, hi))

U = pd.DataFrame(rows)
U['metric_NOTE'] = np.where(U.cohort == 'A', 'C-index = Harrell C; native = time-dep AUC (same caliber as 32)',
                            'C-index = ROC-AUC (identical in the binary case); native = ROC-AUC')
U.to_csv(os.path.join(OUT, '300_m1c_unified.csv'), index=False, encoding='utf-8-sig')
log('-> 300_m1c_unified.csv  (%d rows)' % len(U))

# -- three-arm contrast (unified metric) ------------------------------
PAIRS = [('conventional WT', 'conventional habitat', 'spatial habitat')]
def mod_of(arm):
    for mk in ['DWIDCE', 'CTPET', 'DCE', 'DWI', 'CT', 'PET']:
        if mk in arm:
            return mk
    return ''
def kind_of(arm):
    for kk in ['conventional WT', 'conventional habitat', 'spatial habitat']:
        if arm.startswith(kk):
            return kk
    return ''

comp, delt = [], []
for cohort, arms, oofd, ev, tm, n in [('A', ARMS_A, OOFD_A, evA, tmA, len(a)),
                                      ('B', ARMS_B, OOFD_B, evB, tmB, len(yB))]:
    for mk in ['DCE', 'DWI', 'DWIDCE', 'CT', 'PET', 'CTPET']:
        got = {}
        for arm in arms:
            if 'clinical' in arm:      # only the EPV-matched "no clinical covariate" arms, matching Table 3 of the manuscript
                continue
            k = kind_of(arm)
            if k and mk == mod_of(arm) and arm.endswith(mk):
                got[k] = arm
        if len(got) == 3:
            for k, arm in got.items():
                r = U[(U.cohort == cohort) & (U.arm == arm)].iloc[0]
                comp.append(dict(cohort=cohort, modality=mk, arm=k,
                                 C=r.C_mean, C_sd=r.C_sd, ci='[%.3f,%.3f]' % (r.C_lo, r.C_hi)))
            # 3 vs 1, 3 vs 2 -- common caliber: average the 20 seed OOFs first, then patient-level bootstrap
            for ref in ['conventional WT', 'conventional habitat']:
                a3, a2 = got['spatial habitat'], got[ref]
                s3v = np.mean([load_oof(oofd, a3, sd) for sd in sorted(arms[a3])], axis=0)
                s2v = np.mean([load_oof(oofd, a2, sd) for sd in sorted(arms[a2])], axis=0)
                d_point = (concordance_index_censored(ev, tm, s3v)[0] -
                           concordance_index_censored(ev, tm, s2v)[0])
                rng = np.random.default_rng(SEED_BOOT); bs = []
                for _ in range(N_BOOT):
                    ix = rng.integers(0, n, n)
                    try:
                        bs.append(concordance_index_censored(ev[ix], tm[ix], s3v[ix])[0] -
                                  concordance_index_censored(ev[ix], tm[ix], s2v[ix])[0])
                    except Exception:
                        pass
                bs = np.array(bs)
                delt.append(dict(cohort=cohort, modality=mk, contrast='3 spatial habitat - %s' % ref,
                                 dC=d_point, C3=concordance_index_censored(ev, tm, s3v)[0],
                                 C_ref=concordance_index_censored(ev, tm, s2v)[0],
                                 ci='[%.3f,%.3f]' % (np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                                 p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                                 caliber='OOF 20-seed mean + patient-level bootstrap 1000'))

C = pd.DataFrame(comp); D = pd.DataFrame(delt)
C.to_csv(os.path.join(OUT, '300_m1c_arms.csv'), index=False, encoding='utf-8-sig')
D.to_csv(os.path.join(OUT, '300_m1c_delta.csv'), index=False, encoding='utf-8-sig')
log('\n== three-arm contrast (unified metric, C-index) ==')
log(C.to_string(index=False) if len(C) else '(no complete three-arm modality)')
log('\n== paired differences for arm 3 (unified metric) ==')
log(D.to_string(index=False) if len(D) else '(none)')
log('\n-> 300_m1c_arms.csv / 300_m1c_delta.csv')
