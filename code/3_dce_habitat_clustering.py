#!/usr/bin/env python3
"""
DCE habitat clustering.
Voxel-wise clustering based on the DCE P3 signal.
DCE images contain a large number of voxels, so only the signal intensity is
used as a feature (no local texture, to save computation).

Input:  data/file_mapping.csv
Output: data/dce_habitat_masks/{Case_No.}/habitat_K.nii.gz
        data/habitat_cluster_info.csv (DCE information appended)
"""
import pandas as pd
import numpy as np
import nibabel as nib
import os, sys
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

# ========== Configuration ==========
MAPPING_FILE = os.path.join(BASE, 'data/file_mapping.csv')
MRI_ROOT = os.path.join(BASE, '使用数据/MRI数据')
OUTPUT_DIR = os.path.join(BASE, 'data')
MASK_DIR = os.path.join(OUTPUT_DIR, "dce_habitat_masks")
os.makedirs(MASK_DIR, exist_ok=True)

K_MIN = 2
K_MAX = 5
RANDOM_STATE = 42
MIN_VOXELS = 30  # DCE has many voxels, but a lower bound is still kept

# ========== Load mapping table ==========
df = pd.read_csv(MAPPING_FILE)
print(f"Cases to process: {len(df)}")

# ========== Helper functions ==========
def load_nifti(rel_path):
    full_path = os.path.join(MRI_ROOT, rel_path)
    if not os.path.exists(full_path):
        return None, None
    try:
        nii = nib.load(full_path)
        arr = np.asanyarray(nii.dataobj)
        return arr, nii
    except Exception as e:
        print(f"    Load failed: {e}")
        return None, None

def find_dce_mask(row):
    """Locate the DCE mask file, tolerating both the masks and labels directories."""
    dce_path = row['DCE_path']
    if not dce_path or dce_path == 'nan':
        return None, None
    center = dce_path.split('/')[0]
    # DCE image names look like <CASE>P3.nii.gz and the mask carries the same name
    dce_basename = os.path.basename(dce_path)
    dce_stem = dce_basename.replace('.nii.gz', '').upper().replace('P3', '')
    
    for subdir in ['masks', 'labels']:
        md = f"{center}/DCE/{subdir}"
        fp = os.path.join(MRI_ROOT, md)
        if os.path.isdir(fp):
            for f in os.listdir(fp):
                if f.endswith('.nii.gz'):
                    f_stem = f.replace('.nii.gz', '').upper().replace('LABEL', '').replace('P3', '')
                    if f_stem == dce_stem:
                        m_arr, m_nii = load_nifti(f"{md}/{f}")
                        if m_arr is not None:
                            return m_arr, m_nii
    return None, None

def extract_voxel_features(dce_arr, mask_arr):
    """Extract voxel-wise features from the DCE ROI (signal intensity only)."""
    roi_mask = (mask_arr > 0)
    features = []
    indices = []
    
    for z in range(mask_arr.shape[0]):
        for y in range(mask_arr.shape[1]):
            for x in range(mask_arr.shape[2]):
                if roi_mask[z, y, x]:
                    features.append([dce_arr[z, y, x]])
                    indices.append([z, y, x])
    
    return np.array(features), np.array(indices)

def determine_optimal_k(features, k_min=2, k_max=5):
    bic_scores = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=RANDOM_STATE, n_init=5)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
    best_idx = np.argmin(bic_scores)
    return k_min + best_idx, bic_scores

def save_habitat_masks(labels, indices, nii_img, case_no, output_dir, n_clusters):
    case_dir = os.path.join(output_dir, case_no)
    os.makedirs(case_dir, exist_ok=True)
    
    shape = nii_img.shape[:3] if len(nii_img.shape) > 3 else nii_img.shape
    full_mask = np.zeros(shape, dtype=np.int32)
    
    for i, (z, y, x) in enumerate(indices):
        full_mask[z, y, x] = labels[i] + 1
    
    mask_nii = nib.Nifti1Image(full_mask, nii_img.affine, nii_img.header)
    nib.save(mask_nii, os.path.join(case_dir, "habitat_all.nii.gz"))
    
    for k in range(n_clusters):
        single_mask = (full_mask == (k + 1)).astype(np.int32)
        single_nii = nib.Nifti1Image(single_mask, nii_img.affine, nii_img.header)
        nib.save(single_nii, os.path.join(case_dir, f"habitat_{k+1}.nii.gz"))
    
    return os.path.join(case_dir, "habitat_all.nii.gz")


def plot_bic_curve(bic_scores, k_min, case_no, output_dir):
    ks = range(k_min, k_min + len(bic_scores))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, bic_scores, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('BIC score')
    plt.title(f'DCE Habitat BIC - {case_no}')
    plt.grid(alpha=0.3)
    best_k = k_min + np.argmin(bic_scores)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, label=f'Best K={best_k}')
    plt.legend()
    plt.tight_layout()
    case_dir = os.path.join(output_dir, case_no)
    os.makedirs(case_dir, exist_ok=True)
    plt.savefig(os.path.join(case_dir, "dce_bic_curve.png"), dpi=150)
    plt.close()


# ========== Main workflow ==========
# Resume existing results
completed = set()
for d in os.listdir(MASK_DIR):
    cd = os.path.join(MASK_DIR, d)
    if os.path.isdir(cd) and os.path.exists(os.path.join(cd, "habitat_all.nii.gz")):
        completed.add(d)

if completed:
    print(f"{len(completed)} DCE habitats already completed and will be skipped")

results = []
for idx, row in df.iterrows():
    case_no = row['Case_No.']
    name = row['Name']
    
    if case_no in completed:
        print(f"[{idx+1}/{len(df)}] {case_no} {name} ⏭ skipped")
        continue
    
    print(f"\n[{idx+1}/{len(df)}] {case_no} {name}")
    
    # 1. Load the DCE image
    dce_path = row['DCE_path']
    if not dce_path or dce_path == 'nan':
        print(f"  ⚠ No DCE path")
        continue
    dce_arr, dce_nii = load_nifti(dce_path)
    if dce_arr is None:
        print(f"  ⚠ DCE file missing")
        continue
    
    # 2. Locate the DCE mask
    mask_arr, mask_nii = find_dce_mask(row)
    if mask_arr is None:
        print(f"  ⚠ DCE mask not found")
        continue
    
    print(f"  DCE={dce_arr.shape}, Mask={mask_arr.shape}")
    
    # 3. Extract voxel features
    features, indices = extract_voxel_features(dce_arr, mask_arr)
    if len(features) < MIN_VOXELS:
        print(f"  ⚠ Too few ROI voxels: {len(features)}")
        continue
    
    print(f"  ROI voxels: {len(features)}")
    
    # 4. Standardization
    features = np.nan_to_num(features, nan=0.0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 5. PCA (a 1-D feature needs no reduction)
    n_components = min(features_scaled.shape[1], features_scaled.shape[0])
    if n_components > 1:
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        features_pca = pca.fit_transform(features_scaled)
        print(f"  PCA: {features_scaled.shape[1]}→{features_pca.shape[1]} dims")
    else:
        features_pca = features_scaled
        print(f"  Single-dimensional feature, PCA skipped")
    
    # 6. Determine K with GMM + BIC
    best_k, bic_scores = determine_optimal_k(features_pca)
    print(f"  Optimal K={best_k} (BIC: {[f'{b:.0f}' for b in bic_scores]})")
    
    # Plot the BIC curve
    try:
        plot_bic_curve(bic_scores, K_MIN, case_no, MASK_DIR)
    except:
        pass
    
    # 7. K-means
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(features_pca)
    cluster_counts = np.bincount(labels)
    print(f"  Voxels per habitat: {cluster_counts}")
    
    # 8. Save the mask
    ref_nii = dce_nii if dce_nii is not None else mask_nii
    save_habitat_masks(labels, indices, ref_nii, case_no, MASK_DIR, best_k)
    
    results.append({
        'Case_No.': case_no,
        'Name': name,
        'N_voxels': len(features),
        'DCE_Best_K': best_k,
    })
    
    if (idx + 1) % 50 == 0:
        print(f"  [intermediate save: {len(results)} cases]")

# ========== Save ==========
if results:
    df_r = pd.DataFrame(results)
    out_file = os.path.join(OUTPUT_DIR, "dce_cluster_info.csv")
    df_r.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\nNew DCE results: {len(results)} cases -> {out_file}")
    
    # Summary
    print(f"\nDCE K distribution:")
    print(df_r['DCE_Best_K'].value_counts().sort_index().to_string())

total = len(completed) + len(results)
print(f"\nDCE habitats in total: {total}/{len(df)} cases")
