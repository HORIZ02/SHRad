import os
#!/usr/bin/env python3
"""Run every model in one pass: imaging only + clinical baseline + imaging+clinical"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
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
feats = [c for c in dwi.columns if c not in ['Case_No.','Habitat']]

def get_hab(df, hab):
    return df[df['Habitat']==hab][['Case_No.']+feats].drop_duplicates('Case_No.').set_index('Case_No.')

dwi_wt = get_hab(dwi,'Whole_tumor')
dwi_l  = get_hab(dwi,'habitat_L')
dwi_m  = get_hab(dwi,'habitat_M')
dwi_h  = get_hab(dwi,'habitat_H')
dce_wt = get_hab(dce,'Whole_tumor')
dce_l  = get_hab(dce,'habitat_L')
dce_m  = get_hab(dce,'habitat_M')
dce_h  = get_hab(dce,'habitat_H')

# Model definitions: (name, description, type, data source)
model_defs = []

# Imaging only (8)
pure_imaging = [
    ('DWI_WT', 'DWI Whole_tumor (imaging only)', dwi_wt),
    ('DWI_L',  'DWI Habitat_L (imaging only)', dwi_l),
    ('DWI_M',  'DWI Habitat_M (imaging only)', dwi_m),
    ('DWI_H',  'DWI Habitat_H (imaging only)', dwi_h),
    ('DCE_WT', 'DCE Whole_tumor (imaging only)', dce_wt),
    ('DCE_L',  'DCE Habitat_L (imaging only)', dce_l),
    ('DCE_M',  'DCE Habitat_M (imaging only)', dce_m),
    ('DCE_H',  'DCE Habitat_H (imaging only)', dce_h),
]
for mn, desc, src in pure_imaging:
    model_defs.append((mn, desc, 'pure', src))

# Clinical baseline (2)
model_defs.append(('M0_all', '11 clinical variables', 'clinical', clin_vars))
model_defs.append(('M0_sel', '6 clinical variables', 'clinical', sel_vars))

# Imaging + clinical (10, similar to the earlier M1~M4 + M5~M7)
image_clinical = [
    ('M1',   'DWI WT + clinical', dwi_wt),
    ('M2a',  'DWI L + clinical', dwi_l),
    ('M2b',  'DWI M + clinical', dwi_m),
    ('M2c',  'DWI H + clinical', dwi_h),
    ('M3',   'DCE WT + clinical', dce_wt),
    ('M4a',  'DCE L + clinical', dce_l),
    ('M4b',  'DCE M + clinical', dce_m),
    ('M4c',  'DCE H + clinical', dce_h),
    ('M5',   'DWI H+DCE H + clinical', pd.concat([dwi_h.add_prefix('DWI_H_'), dce_h.add_prefix('DCE_H_')], axis=1)),
    ('M6',   'DWI+DCE all + clinical', pd.concat([
        dwi_wt.add_prefix('DWI_WT_'), dwi_l.add_prefix('DWI_L_'),
        dwi_m.add_prefix('DWI_M_'), dwi_h.add_prefix('DWI_H_'),
        dce_wt.add_prefix('DCE_WT_'), dce_l.add_prefix('DCE_L_'),
        dce_m.add_prefix('DCE_M_'), dce_h.add_prefix('DCE_H_')], axis=1)),
    ('M7',   'DWI H+DCE L + clinical', pd.concat([dwi_h.add_prefix('DWI_H_'), dce_l.add_prefix('DCE_L_')], axis=1)),
]
for mn, desc, src in image_clinical:
    model_defs.append((mn, desc, 'combined', src))

def make_surv(y):
    return np.array([(bool(e),t) for e,t in zip(y['DFS'],y['Time'])], dtype=[('event','bool'),('time','float')])

def run_lasso_cv(X, y):
    alphas = np.logspace(-1.5, 0, 20)
    X_s = X.values
    kf = KFold(5, shuffle=True, random_state=42)
    cv_scores = {a:[] for a in alphas}
    for tr, te in kf.split(X_s):
        s = StandardScaler(); X_tr=s.fit_transform(X_s[tr]); X_te=s.transform(X_s[te])
        try:
            cph = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=alphas, max_iter=10000)
            cph.fit(X_tr, y[tr])
            for a in alphas:
                try:
                    p=cph.predict(X_te,alpha=a)
                    cv_scores[a].append(concordance_index_censored(y['event'][te],y['time'][te],p)[0])
                except: cv_scores[a].append(0.5)
        except:
            for a in alphas: cv_scores[a].append(0.5)
    cm={a:np.mean(v) for a,v in cv_scores.items()}
    ba=max(cm,key=cm.get)
    if all(v<=0.5 for v in cm.values()): return 0.5,0.1,pd.Series(dtype=float),StandardScaler(),None
    s=StandardScaler(); X_f=s.fit_transform(X_s)
    cph=CoxnetSurvivalAnalysis(l1_ratio=1,alphas=[ba],max_iter=10000).fit(X_f,y)
    coef=cph.coef_.flatten(); sel=pd.Series(coef,index=X.columns)[coef!=0]
    return cm[ba], ba, sel, s, cph

y=clin[['DFS','Time']]; y_arr=make_surv(y)
results = []

for mn, desc, mtype, src in model_defs:
    print(f"\n{mn}: {desc}", flush=True)
    
    if mtype == 'pure':
        X = src.loc[clin['Case_No.']].fillna(0).reset_index(drop=True)
    elif mtype == 'clinical':
        X = clin[src]
    else:
        X = pd.concat([clin[clin_vars], src.loc[clin['Case_No.']].reset_index(drop=True)], axis=1)
    
    print(f"  N={len(X)}, Events={int(y['DFS'].sum())}, Features={X.shape[1]}", flush=True)
    
    best_c, best_a, sel, scaler, cph = run_lasso_cv(X, y_arr)
    n_sel = len(sel)
    print(f"  C-index(CV)={best_c:.4f}, selected={n_sel}", flush=True)
    if n_sel > 0 and cph is not None:
        print(f"  Features: {', '.join(sel.index[:5])}...", flush=True)
    
    # Risk score
    if n_sel > 0 and cph is not None:
        risk=cph.predict(scaler.transform(X)).flatten()
    else:
        risk=np.zeros(len(X))
    
    # AUC
    auc_36=auc_60=np.nan
    if n_sel > 0:
        try: auc_36=round(cumulative_dynamic_auc(y_arr,y_arr,risk,times=[36])[0][0],4)
        except: pass
        try: auc_60=round(cumulative_dynamic_auc(y_arr,y_arr,risk,times=[60])[0][0],4)
        except: pass
    
    # Brier
    brier=np.nan
    if n_sel > 0:
        try:
            lr=LogisticRegression(max_iter=1000).fit(risk.reshape(-1,1),y['DFS'])
            brier=round(np.mean((lr.predict_proba(risk.reshape(-1,1))[:,1]-y['DFS'])**2),4)
        except: pass
    
    # Log-rank
    lr_p=np.nan
    if n_sel > 0 and len(np.unique(risk))>1:
        try:
            grp=pd.cut(risk,2,labels=['L','H'])
            lr_p=logrank_test(y['Time'][grp=='L'],y['Time'][grp=='H'],
                             event_observed_A=y['DFS'][grp=='L'],
                             event_observed_B=y['DFS'][grp=='H']).p_value
        except: pass
    
    results.append({
        'Model':mn,'Desc':desc,'Type':mtype,
        'N':len(X),'Events':int(y['DFS'].sum()),
        'Features_in':X.shape[1],'Features_selected':n_sel,
        'C_index_CV':round(best_c,4),
        'AUC_3yr':auc_36,'AUC_5yr':auc_60,
        'Brier':brier,
        'Logrank_P':f'{lr_p:.4e}' if not np.isnan(lr_p) else '',
    })

df=pd.DataFrame(results)
df.to_csv(os.path.join(OUT,"model_comparison_all.csv"),index=False,encoding='utf-8-sig')
print(f"\n{'='*70}")
print(df[['Model','Desc','Features_selected','C_index_CV','AUC_3yr','AUC_5yr','Brier']].to_string(index=False))
print(f"\n✅ Saved: model_comparison_all.csv")
