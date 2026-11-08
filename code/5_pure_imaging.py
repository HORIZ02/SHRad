import os
#!/usr/bin/env python3
"""LASSO-Cox on imaging features only (no clinical variables)"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

clin = pd.read_csv(os.path.join(BASE, '分析结果/merged_data_corrected.csv'))
dwi = pd.read_csv(os.path.join(BASE, 'data/features/dwi_pyradiomics_k3.csv'))
dce = pd.read_csv(os.path.join(BASE, 'data/features/dce_pyradiomics_k3.csv'))
feats = [c for c in dwi.columns if c not in ['Case_No.','Habitat']]

def get_hab(df, hab):
    return df[df['Habitat']==hab][['Case_No.']+feats].drop_duplicates('Case_No.').set_index('Case_No.')

models = {
    'DWI_WT': get_hab(dwi,'Whole_tumor'),
    'DWI_L':  get_hab(dwi,'habitat_L'),
    'DWI_M':  get_hab(dwi,'habitat_M'),
    'DWI_H':  get_hab(dwi,'habitat_H'),
    'DCE_WT': get_hab(dce,'Whole_tumor'),
    'DCE_L':  get_hab(dce,'habitat_L'),
    'DCE_M':  get_hab(dce,'habitat_M'),
    'DCE_H':  get_hab(dce,'habitat_H'),
}

def run_lasso_cv(X, y):
    alphas = np.logspace(-1.5, 0, 20)
    X_s = X.values
    kf = KFold(3, shuffle=True, random_state=42)
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
    if all(v<=0.5 for v in cm.values()): return 0.5,0.1,pd.Series(dtype=float),None,None
    s=StandardScaler(); X_f=s.fit_transform(X_s)
    cph=CoxnetSurvivalAnalysis(l1_ratio=1,alphas=[ba],max_iter=10000).fit(X_f,y)
    coef=cph.coef_.flatten(); sel=pd.Series(coef,index=X.columns)[coef!=0]
    return cm[ba], ba, sel, s, cph

y=clin[['DFS','Time']]
y_arr=np.array([(bool(e),t) for e,t in zip(y['DFS'],y['Time'])], dtype=[('event','bool'),('time','float')])

results = []
for mn, X_df in models.items():
    X = X_df.loc[clin['Case_No.']].fillna(0).reset_index(drop=True)
    best_c, best_a, sel, scaler, cph = run_lasso_cv(X, y_arr)
    n_sel = len(sel)
    line = f'{mn:8s} C-index={best_c:.4f}'
    
    if n_sel > 0 and cph is not None:
        X_s = scaler.transform(X)
        risk = cph.predict(X_s).flatten()
        try:
            auc_36 = cumulative_dynamic_auc(y_arr,y_arr,risk,times=[36])[0][0]
            line += f' AUC_36={auc_36:.4f}'
        except: line += f' AUC_36=NA'
        try:
            auc_60 = cumulative_dynamic_auc(y_arr,y_arr,risk,times=[60])[0][0]
            line += f' AUC_60={auc_60:.4f}'
        except: line += f' AUC_60=NA'
        lr = LogisticRegression(max_iter=1000)
        lr.fit(risk.reshape(-1,1), y['DFS'])
        brier = np.mean((lr.predict_proba(risk.reshape(-1,1))[:,1]-y['DFS'])**2)
        line += f' Brier={brier:.4f}'
        grp = pd.cut(risk,2,labels=['L','H'])
        try:
            lr_p = logrank_test(y['Time'][grp=='L'],y['Time'][grp=='H'],
                             event_observed_A=y['DFS'][grp=='L'],
                             event_observed_B=y['DFS'][grp=='H']).p_value
            line += f' P={lr_p:.4e}'
        except: pass
    line += f' selected={n_sel}'
    print(line)
