import os
#!/usr/bin/env python3
"""M0~M7 K3 habitat modeling - full LASSO-Cox evaluation"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

OUT = os.path.join(BASE, 'results/models')
os.makedirs(OUT, exist_ok=True)

# ===== Data loading =====
clin = pd.read_csv(os.path.join(BASE, '分析结果/merged_data_corrected.csv'))
clin_vars = ['NAC','N','Age','Grade','T','ALN','proRT','ER','PR','Her2','Meno']
sel_vars = ['NAC','N','Age','Grade','ALN','proRT']

dwi = pd.read_csv(os.path.join(BASE, 'data/features/dwi_pyradiomics_k3.csv'))
dce = pd.read_csv(os.path.join(BASE, 'data/features/dce_pyradiomics_k3.csv'))
dw_feats = [c for c in dwi.columns if c not in ['Case_No.','Habitat']]
dc_feats = [c for c in dce.columns if c not in ['Case_No.','Habitat']]

def get_hab_feats(df, hab, feats):
    sub = df[df['Habitat']==hab][['Case_No.']+feats].drop_duplicates('Case_No.')
    return sub.set_index('Case_No.')[feats]

# Pre-extract the features of each habitat
dwi_wt = get_hab_feats(dwi, 'Whole_tumor', dw_feats)
dwi_l  = get_hab_feats(dwi, 'habitat_L', dw_feats)
dwi_m  = get_hab_feats(dwi, 'habitat_M', dw_feats)
dwi_h  = get_hab_feats(dwi, 'habitat_H', dw_feats)
dce_wt = get_hab_feats(dce, 'Whole_tumor', dc_feats)
dce_l  = get_hab_feats(dce, 'habitat_L', dc_feats)
dce_m  = get_hab_feats(dce, 'habitat_M', dc_feats)
dce_h  = get_hab_feats(dce, 'habitat_H', dc_feats)

models = {
    'M0_all':  ('11 clinical variables', clin_vars),
    'M0_sel':  ('6 clinical variables', sel_vars),
    'M1':      ('DWI WT', dwi_wt),
    'M2a':     ('DWI L', dwi_l),
    'M2b':     ('DWI M', dwi_m),
    'M2c':     ('DWI H', dwi_h),
    'M3':      ('DCE WT', dce_wt),
    'M4a':     ('DCE L', dce_l),
    'M4b':     ('DCE M', dce_m),
    'M4c':     ('DCE H', dce_h),
    'M5':      ('DWI H+DCE H', pd.concat([dwi_h.add_prefix('DWI_H_'), dce_h.add_prefix('DCE_H_')], axis=1)),
    'M6':      ('DWI+DCE all', pd.concat([dwi_wt.add_prefix('DWI_WT_'), dwi_l.add_prefix('DWI_L_'),
                                         dwi_m.add_prefix('DWI_M_'), dwi_h.add_prefix('DWI_H_'),
                                         dce_wt.add_prefix('DCE_WT_'), dce_l.add_prefix('DCE_L_'),
                                         dce_m.add_prefix('DCE_M_'), dce_h.add_prefix('DCE_H_')], axis=1)),
    'M7':      ('DWI H+DCE L', pd.concat([dwi_h.add_prefix('DWI_H_'), dce_l.add_prefix('DCE_L_')], axis=1)),
}

def make_surv(y):
    return np.array([(bool(e),t) for e,t in zip(y['DFS'],y['Time'])], dtype=[('event','bool'),('time','float')])

def run_lasso_cv(X, y):
    alphas = np.logspace(-1.5, 0, 20)
    X_s = X.values
    kf = KFold(3, shuffle=True, random_state=42)
    cv_scores = {a:[] for a in alphas}
    for tr, te in kf.split(X_s):
        s = StandardScaler()
        X_tr = s.fit_transform(X_s[tr])
        X_te = s.transform(X_s[te])
        try:
            cph = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=alphas, max_iter=10000)
            cph.fit(X_tr, y[tr])
            for a in alphas:
                try:
                    p = cph.predict(X_te, alpha=a)
                    cv_scores[a].append(concordance_index_censored(y['event'][te], y['time'][te], p)[0])
                except:
                    cv_scores[a].append(0.5)
        except:
            for a in alphas:
                cv_scores[a].append(0.5)
    cm = {a: np.mean(v) for a,v in cv_scores.items()}
    ba = max(cm, key=cm.get)
    if all(v <= 0.5 for v in cm.values()):
        return 0.5, 0.1, pd.Series(dtype=float), StandardScaler(), None
    s = StandardScaler()
    X_f = s.fit_transform(X_s)
    cph = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[ba], max_iter=10000)
    cph.fit(X_f, y)
    coef = cph.coef_.flatten()
    sel = pd.Series(coef, index=X.columns)[coef != 0]
    return cm[ba], ba, sel, s, cph

results = []
y = clin[['DFS','Time']]
y_arr = make_surv(y)

for mn, (desc, feat_src) in models.items():
    print(f"\n{'='*50}")
    print(f"  {mn}: {desc}")
    
    if isinstance(feat_src, list):  # clinical baseline, no need to append clinical again
        X = clin[feat_src]
    else:  # imaging features, clinical variables must be appended
        X = pd.concat([clin[clin_vars], feat_src.loc[clin['Case_No.']].reset_index(drop=True)], axis=1)
    print(f"  N={len(X)}, Events={int(y['DFS'].sum())}, Features={X.shape[1]}")
    
    best_c, best_a, sel, scaler, cph = run_lasso_cv(X, y_arr)
    n_sel = len(sel)
    print(f"  C-index(CV)={best_c:.4f}, alpha={best_a:.4f}, selected={n_sel}")
    if n_sel > 0:
        print(f"  Features: {', '.join(sel.index[:5])}...")
    
    # Risk score
    if cph is not None and n_sel > 0:
        X_s = scaler.transform(X)
        risk = cph.predict(X_s).flatten()
    else:
        risk = np.zeros(len(X))
    
    # Log-rank
    grp = pd.cut(risk, 2, labels=['L','H'])
    lr = logrank_test(y['Time'][grp=='L'], y['Time'][grp=='H'],
                     event_observed_A=y['DFS'][grp=='L'],
                     event_observed_B=y['DFS'][grp=='H'])
    
    # AUC
    auc_36 = cumulative_dynamic_auc(y_arr, y_arr, risk, times=[36])[0][0]
    auc_60 = cumulative_dynamic_auc(y_arr, y_arr, risk, times=[60])[0][0]
    
    # Brier
    lr_mod = LogisticRegression(max_iter=1000)
    lr_mod.fit(risk.reshape(-1,1), y['DFS'])
    brier = np.mean((lr_mod.predict_proba(risk.reshape(-1,1))[:,1] - y['DFS'])**2)
    
    results.append({
        'Model': mn, 'Desc': desc,
        'N': len(X), 'Events': int(y['DFS'].sum()),
        'Features_in': X.shape[1], 'Features_selected': n_sel,
        'C_index_CV': round(best_c, 4),
        'AUC_3yr': round(auc_36, 4), 'AUC_5yr': round(auc_60, 4),
        'Brier': round(brier, 4),
        'Logrank_P': f'{lr.p_value:.4e}',
    })
    print(f"  AUC_3yr={auc_36:.4f}, AUC_5yr={auc_60:.4f}, Brier={brier:.4f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT, "model_comparison_k3.csv"), index=False, encoding='utf-8-sig')
print(f"\n{'='*60}")
print(df[['Model','Desc','Features_selected','C_index_CV','AUC_3yr','AUC_5yr','Brier']].to_string(index=False))
print(f"\n✅ Saved: {os.path.join(OUT, 'model_comparison_k3.csv')}")
