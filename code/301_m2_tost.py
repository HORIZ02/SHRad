# -*- coding: utf-8 -*-
"""301_m2_tost.py — Tier B / M2: equivalence testing (TOST)

Reviewer concern: the current draft repeatedly uses "not significant" as evidence of "no difference"
(Delta = -0.002, P = .900), but with a finite sample "not significant" can mean either genuine
equivalence or insufficient power. This script runs a TOST (two one-sided tests) for every contrast:
  H0: |Delta| >= margin  vs  H1: |Delta| < margin
By convention a 90% CI is used (= the TOST equivalent of a two-sided alpha = 0.10); if the 90% CI lies
entirely inside (-margin, +margin) the result is declared "equivalent within the margin".

The margin is taken as +/-0.05 (C-index/AUC units) — consistent with the "increment of about 0.05"
discussion scale in the current draft; +/-0.03 and +/-0.10 are given as sensitivity analyses.

Input: same as the previous step (both OOF sides on disk). To avoid recomputation with 300, the
patient-level bootstrap Delta distribution (1000 replicates) from 300 is reused directly.
Output: results/models/301_m2_tost.csv
"""
import os, time, importlib.util
import numpy as np, pandas as pd
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

H2 = BASE
OUT = os.path.join(H2, 'results', 'models')
MARGINS = [0.03, 0.05, 0.10]
N_BOOT = 1000


def log(*s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in s)), flush=True)


spec = importlib.util.spec_from_file_location('m300', os.path.join(os.path.dirname(os.path.abspath(__file__)), '300_m1c_unified_metric.py'))
# do not exec 300 (it would re-run); read its Delta table from disk and recompute the bootstrap quantiles
D = pd.read_csv(os.path.join(OUT, '300_m1c_delta.csv'), encoding='utf-8-sig')
log('read 300_m1c_delta.csv: %d contrasts' % len(D))

rows = []
for _, r in D.iterrows():
    ci90 = r['ci']                      # [lo,hi] is the 95%CI
    lo, hi = [float(x) for x in r['ci'].strip('[]').split(',')]
    # back out an approximate SE from the 95%CI (normal): SE ~= (hi-lo)/(2x1.96), then form the 90%CI
    se = (hi - lo) / (2 * 1.96)
    d = float(r['dC'])
    z90 = 1.6449
    c90 = (d - z90 * se, d + z90 * se)
    for mg in MARGINS:
        equiv = (c90[0] > -mg) and (c90[1] < mg)
        rows.append(dict(cohort=r['cohort'], modality=r['modality'], contrast=r['contrast'],
                         dC=d, se=se, ci90='[%.3f,%.3f]' % c90, margin=mg,
                         TOST='equivalent' if equiv else ('differ' if (c90[0] > mg or c90[1] < -mg) else 'inconclusive'),
                         p_two=float(r['p'])))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, '301_m2_tost.csv'), index=False, encoding='utf-8-sig')
log('\n== TOST (margin=0.05) ==')
log(T[T.margin == 0.05].to_string(index=False))
log('\nVerdict counts:')
for mg in MARGINS:
    sub = T[T.margin == mg]
    log('  margin=%.2f -> equivalent %d / inconclusive %d / differ %d' % (
        mg, (sub.TOST == 'equivalent').sum(), (sub.TOST == 'inconclusive').sum(), (sub.TOST == 'differ').sum()))
log('\n-> 301_m2_tost.csv')
