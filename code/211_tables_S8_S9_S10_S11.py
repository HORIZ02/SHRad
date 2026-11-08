# -*- coding: utf-8 -*-
"""
211_tables_S8_S9_S10_S11.py — v3 supplementary tables S8 / S9 / S10 / S11 (all written under <SHRAD_ROOT>)

S2 sample-definition sensitivity   source dwi288_sensitivity.csv (generated together with 209/210)
S6 family-completeness battery     source classic_spatial_*/dist_shape_*/fixed_spatial_* (HER2-low + prostate)
S7 multiplicity correction         source fdr_correction.csv (135 comparisons / 9 families)
S8 learner and screening sweep     source prostate all_classifiers + feature_selection
"""
import os, shutil
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
PROSTATE_ROOT = os.environ.get("PROSTATE_ROOT", os.path.abspath(os.getcwd()))

W2 = BASE
M = os.path.join(W2, 'results', 'models')
OUTD = os.path.join(W2, 'new_results')
PRO = os.path.join(PROSTATE_ROOT, 'results')
PCD = os.path.join(PRO, 'cross_dataset')
os.makedirs(OUTD, exist_ok=True)


def log(*s):
    print(*s, flush=True)


# ============================ three-line table helpers ============================
def new_doc():
    doc = Document()
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(9)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    sec = doc.sections[0]; sec.orientation = 1
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.left_margin = sec.right_margin = Cm(1.5)
    return doc


def add_table(doc, header, data):
    t = doc.add_table(rows=1 + len(data), cols=len(header))
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ''
        r = c.paragraphs[0].add_run(str(h)); r.bold = True
        r.font.name = 'Times New Roman'; r.font.size = Pt(9)
    for i, row in enumerate(data, 1):
        for j, v in enumerate(row):
            c = t.rows[i].cells[j]; c.text = ''
            r = c.paragraphs[0].add_run(str(v))
            r.font.name = 'Times New Roman'; r.font.size = Pt(9)
    n = len(data) + 1
    for i, row in enumerate(t.rows):
        for c in row.cells:
            tcPr = c._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn('w:tcBorders')):
                tcPr.remove(old)
            b = OxmlElement('w:tcBorders'); tcPr.append(b)
            if i == 0:
                e = OxmlElement('w:top'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '12'); e.set(qn('w:color'), '000000'); b.append(e)
                e = OxmlElement('w:bottom'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '6'); e.set(qn('w:color'), '000000'); b.append(e)
            elif i == n - 1:
                e = OxmlElement('w:bottom'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '12'); e.set(qn('w:color'), '000000'); b.append(e)
    return t


def para(doc, text, bold=False, size=9, before=5, after=3):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text); r.bold = bold; r.font.name = 'Times New Roman'; r.font.size = Pt(size)
    r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    return p


def f3(v):
    try:
        return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):.3f}'
    except Exception:
        return str(v)


# ============================ S2 sample-definition sensitivity ============================
d = pd.read_csv(os.path.join(M, 'dwi288_sensitivity.csv'), encoding='utf-8-sig')
assert set(d.subset) >= {'n345', 'n288'}, d.subset.unique()
d.to_csv(os.path.join(M, 'tableS2_sample_definition_345.csv'), index=False, encoding='utf-8-sig')
docS2 = new_doc()
rows = []
for sm in ['n345', 'n288']:
    for _, r in d[d.subset == sm].iterrows():
        rows.append([sm.replace('n', ''), r.model, r.n_feat, f'{r.EPV:.2f}',
                     f'{f3(r.C_OOF)} ({f3(r.CI_lo)}\u2013{f3(r.CI_hi)})',
                     f'{f3(r.C_mean)} \u00b1 {f3(r.C_sd)}',
                     ('—' if pd.isna(r.dC) else f'{r.dC:+.4f}'),
                     ('—' if pd.isna(r.p) else ('<.001' if r.p < .001 else f'{r.p:.3f}'))])
add_table(docS2, ['n', 'Model', 'Variables', 'EPV', 'C-index (95% CI)',
                  'C-index, mean \u00b1 SD', '\u0394C', 'P'], rows)
para(docS2, 'Values are the out-of-fold C-index; the 95% confidence interval is by patient-level bootstrap and the '
            'mean \u00b1 SD is across 20 random partitions. \u0394C and P are versus the reference model of the same '
            'sample definition, by paired bootstrap. The 288-patient subset excludes the 57 cases, all from Center 2, '
            'whose DWI volumes contained only two distinct intensity values.')
para(docS2, 'Table S2. Sensitivity of the model comparisons in cohort A to the exclusion of cases with an inadequate '
            'DWI intensity range. Model performance in all 345 patients and in the 288 patients for every model '
            'configuration: number of variables, events per variable, out-of-fold C-index at the primary partition with '
            'bootstrap 95% confidence interval, mean \u00b1 SD across 20 random partitions, and the difference in '
            'C-index versus the reference model with a paired-bootstrap P value.')
docS2.save(os.path.join(OUTD, 'TableS10_Sample_Definition.docx'))
log(f'S10 -> TableS10_Sample_Definition.docx ({len(rows)} rows)')

# ============================ S8 learner and screening sweep ============================
ac = pd.read_csv(os.path.join(PRO, 'all_classifiers', 'results.csv'), encoding='utf-8-sig')
fs = pd.read_csv(os.path.join(PRO, 'feature_selection', 'results.csv'), encoding='utf-8-sig')
ac.to_csv(os.path.join(M, 'tableS8_all_classifiers.csv'), index=False, encoding='utf-8-sig')
fs.to_csv(os.path.join(M, 'tableS8_feature_selection.csv'), index=False, encoding='utf-8-sig')
docS8 = new_doc()
para(docS8, '(a) Classifiers (cohort B, identical feature set)', bold=True, before=2)
add_table(docS8, ['Classifier', 'AUC', '95% CI'],
          [[r.Classifier, ('failed to converge' if pd.isna(r.AUC) else f3(r.AUC)),
            ('—' if pd.isna(r.AUC) else f'{r.CI_low:.3f}\u2013{r.CI_high:.3f}')] for _, r in ac.iterrows()])
para(docS8, '(b) Feature-selection rules (cohort B, matched pool size of 8 variables)', bold=True, before=8)
add_table(docS8, ['Rule', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'Brier', 'Variables selected (mean)'],
          [[r.Method, f3(r.AUC), f3(r.ACC), f3(r.Sens), f3(r.Spec), f3(r.F1), f3(r.Brier), f'{r.avg_n_selected:.0f}']
           for _, r in fs.iterrows()])
para(docS8, 'TabPFN failed to converge on this feature set and is reported as such rather than omitted. All rules '
            'were evaluated at the same average pool size so that the comparison is not confounded by model '
            'complexity. SVM = support vector machine; LDA = linear discriminant analysis; MLP = multilayer '
            'perceptron; mRMR = minimum redundancy maximum relevance.')
para(docS8, 'Table S8. Learner and screening-rule sweeps. (a) Twenty classifiers evaluated in cohort B on the same '
            'feature set, with the out-of-fold AUC; the three best were linear. (b) Five feature-selection rules '
            'evaluated in cohort B at the same pool size; univariable screening was best.')
docS8.save(os.path.join(OUTD, 'TableS11_Learner_Screening.docx'))
log(f'S11 -> TableS11_Learner_Screening.docx ({len(ac)} + {len(fs)} rows)')

# ============================ S7 multiplicity correction ============================
fd = pd.read_csv(os.path.join(M, 'fdr_correction.csv'), encoding='utf-8-sig')
assert len(fd) == 135, f'fdr rows {len(fd)} != 135'
docS7 = new_doc()
add_table(docS7, ['Family', 'Metric', 'Comparison', 'P', 'q (within family)', 'q (global)'],
          [[r.family, r.metric, r.comparison,
            ('<.001' if r.p < .001 else f3(r.p)), f3(r.q_within), f3(r.q_global)]
           for _, r in fd.sort_values(['family', 'p']).iterrows()])
para(docS7, 'Nine prespecified families; the within-family value is the Benjamini\u2013Hochberg correction applied '
            'within each family and the global value across all 135 comparisons. Of 135 comparisons, 23 were '
            'nominally significant and 8 survived within-family correction; all 8 were negative findings.')
para(docS7, 'Table S7. Multiplicity correction. All 135 comparisons grouped into the nine prespecified families, '
            'with the nominal P value, the within-family Benjamini\u2013Hochberg corrected value and the global '
            'corrected value.')
docS7.save(os.path.join(OUTD, 'TableS9_Multiplicity.docx'))
log(f'S9 -> TableS9_Multiplicity.docx ({len(fd)} rows)')

# ============================ S6 family-completeness battery ============================
docS6 = new_doc()
tests = [
    ('Cohort A', None, os.path.join(M, 'classic_spatial_tests_her2.csv'), 'Classic spatial statistics'),
    ('Cohort A', None, os.path.join(M, 'dist_shape_tests_her2.csv'), 'Distribution shape'),
    ('Cohort B', 'base', os.path.join(PCD, 'classic_spatial_tests_prostate.csv'), 'Classic spatial statistics'),
]
rowsA = []
for coh, basecol, p, lab in tests:
    t = pd.read_csv(p, encoding='utf-8-sig')
    for _, r in t.iterrows():
        rowsA.append([coh, (r[basecol] if basecol else '—'), r.k, lab, r.comparison,
                      f'{r.d_mean20:+.5f}', r.CI,
                      ('<.001' if r.p < .001 else f3(r.p)), r['concordant']])
para(docS6, '(a) Pool contrasts: the added family versus the same-size pool, and versus the base', bold=True, before=2)
add_table(docS6, ['Cohort', 'Base', 'k', 'Added family', 'Contrast', '\u0394', '95% CI', 'P', 'Concordant'],
          rowsA)
docS6.add_page_break()

rowsB = []
for coh, fam, p in [('Cohort A', 'Classic spatial statistics', os.path.join(M, 'classic_spatial_lr_her2.csv')),
                    ('Cohort A', 'Distribution shape', os.path.join(M, 'dist_shape_lr_her2.csv')),
                    ('Cohort B (base PSA)', 'Classic spatial statistics', os.path.join(PCD, 'classic_spatial_lr_prostate_PSA.csv')),
                    ('Cohort B (base 4)', 'Classic spatial statistics', os.path.join(PCD, 'classic_spatial_lr_prostate_4.csv'))
                    if os.path.exists(os.path.join(PCD, 'classic_spatial_lr_prostate_4.csv'))
                    else ('Cohort B (base 4)', 'Classic spatial statistics', os.path.join(PCD, 'classic_spatial_lr_prostate_临床4.csv'))]:
    t = pd.read_csv(p, encoding='utf-8-sig')
    for _, r in t.iterrows():
        rowsB.append([coh, fam, r.feature, f'{r.LR:.4f}',
                      ('<.001' if r.p < .001 else f3(r.p)), f3(r.p_FDR)])
para(docS6, '(b) Descriptor-by-descriptor likelihood-ratio tests beyond conventional shape', bold=True, before=2)
add_table(docS6, ['Cohort', 'Family', 'Descriptor', 'LR', 'P', 'FDR'], rowsB)
docS6.add_page_break()

ft = pd.read_csv(os.path.join(PCD, 'fixed_spatial_tests_prostate.csv'), encoding='utf-8-sig')
para(docS6, '(c) Forced-addition adjudication (cohort B): descriptors significant in the in-sample test, '
            'forced in without screening and evaluated on held-out partitions', bold=True, before=2)
add_table(docS6, ['Contrast', '\u0394', '95% CI', 'P', 'Concordant'],
          [[r.comparison, f'{r.d_mean20:+.5f}', r.CI,
            ('<.001' if r.p < .001 else f3(r.p)), r['concordant']] for _, r in ft.iterrows()])
para(docS6, '\u0394 values are means over 20 partitions; P values are nominal unless a false-discovery-rate column is '
            'shown. Three descriptors in cohort A (dce_L_surface, dce_M_surface, dce_H_surface) were not testable: '
            'each was collinear with the conventional shape block at |\u03c1| \u2265 0.9997, which makes the augmented '
            'design rank-deficient, and only the 39 testable descriptors were carried into the correction. '
            'LR = likelihood ratio.')
para(docS6, 'Table S6. Descriptor-family completeness battery. Pool contrasts and descriptor-by-descriptor '
            'likelihood-ratio tests for the classic spatial statistics and the distribution-shape descriptors, in both '
            'cohorts, with the forced-addition adjudication.')
docS6.save(os.path.join(OUTD, 'TableS8_Family_Completeness.docx'))
log(f'S8 -> TableS8_Family_Completeness.docx ((a){len(rowsA)} + (b){len(rowsB)} + (c){len(ft)} rows)')
log('done S2/S6/S7/S8.')
