# Spatial Habitat Radiomics — selection-position (attribution) analysis code

Code for the study:

> **Feature Selection Outside the Cross-Validation Loop Inflates Reported Performance of Radiomics and
> Deep-Learning Image Encoders: A Two-Cohort Quantification and Control Framework.**

The analysis asks a single question: how much of the *reported* increment of an imaging feature family is
produced by **where** feature selection is performed (inside the training fold vs on the full cohort),
rather than by the features themselves. Two families are compared under one protocol in two cohorts:
handcrafted spatial habitat descriptors (SHRad) and frozen 3D-encoder embeddings.

---

## Contents

```
code/
  1_habitat_clustering.py                 habitat clustering and K selection (cohort A)
  3_dce_habitat_clustering.py             DCE-specific clustering variant
  2_habitat_radiomics.py                  per-habitat radiomics aggregation
  21_spatial_features_k3.py               spatial habitat descriptor (SHRad) computation, K = 3
  22_univariate_spatial_cox_k3.py         univariable Cox screen of SHRad descriptors
  5_pure_imaging.py                       imaging-only models
  5_k3_modeling.py                        M0–M7 protocol, K = 3
  5_all_models.py                         all arms: imaging / clinical / combined
  26_parsimonious_models.py               parsimonious model search
  27_parsimonious_stability.py            stability of the parsimonious solution
  29_center_effect_diagnosis.py           centre effect diagnosis
  31_center_inflation_fair.py             centre effect on a common caliber
  32_auc_multiseed.py                     time-dependent AUC, multiple seeds
  33_center_subsets.py                    centre-subset comparisons
  42_calibration_brier_dca.py             calibration, Brier score, decision-curve analysis
  100_k_spatial_full_model.py             K sweep with the spatial pool included
  300_m1c_unified_metric.py               three-arm comparison, matched events per variable
  301_m2_tost.py                          TOST equivalence test
  302_m3m4m5_redundancy.py                redundancy, novelty, family partition
  303_e7_center_aware.py                  leave-one-centre-out, centre-aware evaluation
  306_e19_threshold_family.py             redundancy-threshold and nine-family robustness
  401_deep_leakage_protocol.py            selection position for frozen encoder embeddings
  211_tables_S8_S9_S10_S11.py             classifier sweep and feature-selection-rule sweep (cohort B)
  246_supp_figures.py                     supplementary figure S1 (study flow)
  50_figures_main.py                      main figures
  51_figures_main2.py                     main figures (continued)
  52_figures_supp.py                      supplementary figures
  540_build_deep_inflation_figure.py      selection-position figure (embeddings)
```

## Data are not included

No patient data, images, masks, feature tables or result tables are part of this repository. All scripts
expect the study directory tree to exist locally and read/write inside it.

Two environment variables define the roots:

| Variable | Meaning |
|:--|:--|
| `SHRAD_ROOT` | root of the study tree (features, masks, results) |
| `PROSTATE_ROOT` | root of the independent prostate cohort used as cohort B |

```bash
export SHRAD_ROOT=/path/to/study
export PROSTATE_ROOT=/path/to/prostate_cohort
python code/1_habitat_clustering.py
```

Every script falls back to the current working directory when the variable is unset.

### Directory names used by the code

The scripts were written against the study tree, whose folder names are partly in Chinese. They are
kept verbatim so that the code stays byte-compatible with the tree it was validated on; only the two
roots are configurable.

| Path (relative to `SHRAD_ROOT`) | Content |
|:--|:--|
| `使用数据/MRI数据/` | source DICOM/NIfTI images (not distributed) |
| `分析结果/merged_data_corrected.csv` | merged clinical + outcome table |
| `data/features/` | conventional radiomics feature tables |
| `data/habitat_masks/`, `data/dce_habitat_masks/` | per-case habitat masks |
| `data/file_mapping.csv` | case-to-file mapping |
| `results/models/` | result tables written by the analysis scripts |
| `results/figures_journal/` | rendered figures |

If your tree uses different names, change the constants at the top of the script you run (or rename
the folders); no other code change is required.

## Environment

```bash
python -m pip install -r requirements.txt
```

Python 3.11 was used. Versions are not pinned to a lockfile because the analyses are stable across recent
releases; `scikit-survival` and `lifelines` are required for the survival models.

## Protocol notes (for reading the numbers)

- Discrimination is evaluated **out-of-fold**; the reported increments are paired differences between two
  selection positions computed on the **same** partitions.
- Partition-based dispersion is the sample standard deviation across 20 random five-fold partitions.
  This is *not* a bootstrap confidence interval and is reported separately from it.
- Confidence intervals for paired contrasts come from patient-level bootstrap resampling (1,000
  replicates) on the primary partition.
- Multiplicity: nine prespecified families, Benjamini–Hochberg control within family.
- Absence of a significant difference is not reported as equivalence unless a TOST at the prespecified
  margin was met.

## License

MIT — see `LICENSE`.
