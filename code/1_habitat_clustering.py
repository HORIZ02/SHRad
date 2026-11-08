#!/usr/bin/env python3
"""
Breast cancer multimodal habitat clustering.
Voxel-wise features from DCE + DWI -> PCA -> GMM + BIC to pick K -> K-means -> per-habitat masks.

Input:  data/file_mapping.csv
Output: data/habitat_masks/{Case_No.}/habitat_K.nii.gz (K=1,2,...)
        data/habitat_cluster_info.csv (clustering information per case)
"""
import pandas as pd
import numpy as np
import nibabel as nib
import os, sys
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tempfile
import warnings
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))
warnings.filterwarnings('ignore')

# ========== Configuration ==========
MAPPING_FILE = os.path.join(BASE, 'data/file_mapping.csv')
MRI_ROOT = os.path.join(BASE, '使用数据/MRI数据')
OUTPUT_DIR = os.path.join(BASE, 'data')
MASK_DIR = os.path.join(OUTPUT_DIR, "dwi_habitat_masks")
os.makedirs(MASK_DIR, exist_ok=True)

# Clustering parameters
K_MIN = 2
K_MAX = 5  # Try up to 5 clusters
RANDOM_STATE = 42

# ========== Load mapping table ==========
df = pd.read_csv(MAPPING_FILE)
print(f"Cases to process: {len(df)}")

# ========== Helper functions ==========
def load_nifti(rel_path):
    """Load a .nii.gz file and return (array, nibabel_image)."""
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

def extract_voxel_features(dwi_arr, mask_arr):
    """
    Extract voxel-wise features inside the ROI.
    Returns: (feature_matrix, roi_indices)
        feature_matrix: (N_voxels, n_features)
        roi_indices: (N_voxels, 3) coordinates in 3D space
    """
    from scipy import ndimage
    roi_mask = (mask_arr > 0)
    
    features = []
    indices = []
    
    # Precompute local texture features
    # Local entropy (reflects tissue heterogeneity)
    local_entropy = ndimage.generic_filter(dwi_arr, lambda x: -np.sum((x/np.sum(x+1e-10))*np.log(x/np.sum(x+1e-10)+1e-10)), 
                                           size=3, mode='constant', cval=0)
    # Local variance (reflects edge information)
    local_var = ndimage.uniform_filter(dwi_arr**2, size=3) - ndimage.uniform_filter(dwi_arr, size=3)**2
    
    for z in range(mask_arr.shape[0]):
        for y in range(mask_arr.shape[1]):
            for x in range(mask_arr.shape[2]):
                if roi_mask[z, y, x]:
                    features.append([
                        dwi_arr[z, y, x],        # Raw DWI signal
                        local_entropy[z, y, x],   # Local entropy
                        local_var[z, y, x],       # Local variance
                    ])
                    indices.append([z, y, x])
    
    return np.array(features), np.array(indices)

def determine_optimal_k(features, k_min=2, k_max=5):
    """Determine the optimal number of clusters with GMM + BIC."""
    bic_scores = []
    models = []
    
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', 
                              random_state=RANDOM_STATE, n_init=5)
        gmm.fit(features)
        bic_scores.append(gmm.bic(features))
        models.append(gmm)
    
    best_idx = np.argmin(bic_scores)
    return k_min + best_idx, bic_scores

def apply_kmeans(features_pca, n_clusters):
    """K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(features_pca)
    return labels, kmeans

def save_habitat_masks(labels, indices, nii_img, case_no, output_dir, n_clusters):
    """
    Save the clustering result as one mask per habitat.
    Uses nibabel so that the original affine information is preserved.
    """
    case_dir = os.path.join(output_dir, case_no)
    os.makedirs(case_dir, exist_ok=True)
    
    shape = nii_img.shape  # nibabel: (x, y, z) or (z, y, x)?
    if len(shape) == 3:
        shape_3d = shape
    else:
        shape_3d = shape[:3]
    
    # Create an all-zero array with the same shape as the source image
    full_mask = np.zeros(shape_3d, dtype=np.int32)
    
    # Fill in the cluster labels
    for i, (z, y, x) in enumerate(indices):
        full_mask[z, y, x] = labels[i] + 1  # Labels start at 1
    
    # Save the combined cluster mask
    mask_nii = nib.Nifti1Image(full_mask, nii_img.affine, nii_img.header)
    nib.save(mask_nii, os.path.join(case_dir, "habitat_all.nii.gz"))
    
    # Save one mask per habitat
    for k in range(n_clusters):
        single_mask = (full_mask == (k + 1)).astype(np.int32)
        single_nii = nib.Nifti1Image(single_mask, nii_img.affine, nii_img.header)
        nib.save(single_nii, os.path.join(case_dir, f"habitat_{k+1}.nii.gz"))
    
    return os.path.join(case_dir, "habitat_all.nii.gz")

def plot_bic_curve(bic_scores, k_min, case_no, output_dir):
    """Plot the BIC curve (optional)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    ks = range(k_min, k_min + len(bic_scores))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, bic_scores, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('BIC score')
    plt.title(f'GMM BIC - {case_no}')
    plt.grid(alpha=0.3)
    best_k = k_min + np.argmin(bic_scores)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, label=f'Best K={best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, case_no, "bic_curve.png"), dpi=150)
    plt.close()

# ========== Main workflow ==========
results = []
# Resume from existing results
completed = set()
for d in os.listdir(MASK_DIR):
    case_dir = os.path.join(MASK_DIR, d)
    if os.path.isdir(case_dir) and os.path.exists(os.path.join(case_dir, "habitat_all.nii.gz")):
        completed.add(d)

# Load previously saved records
partial_file = os.path.join(OUTPUT_DIR, "habitat_cluster_info_partial.csv")
if os.path.exists(partial_file):
    existing = pd.read_csv(partial_file)
    results = existing.to_dict('records')
    print(f"Loaded {len(results)} previous records")

if completed:
    print(f"{len(completed)} cases already completed and will be skipped")

for idx, row in df.iterrows():
    case_no = row['Case_No.']
    name = row['Name']
    
    if case_no in completed:
        print(f"[{idx+1}/{len(df)}] {case_no} {name} ⏭ skipped (result already exists)")
        continue
    
    print(f"\n[{idx+1}/{len(df)}] {case_no} {name}")
    
    # 1. Load the DWI image
    dwi_path = row['DWI_path']
    dwi_arr, dwi_img = load_nifti(dwi_path)
    if dwi_arr is None:
        print(f"  ⚠ DWI file missing: {dwi_path}")
        continue
    
    # DWI mask
    center = dwi_path.split('/')[0]
    dwi_basename = os.path.basename(dwi_path)
    mask_dir = f"{center}/DWI/masks"
    
    mask_file = None
    full_mp = os.path.join(MRI_ROOT, mask_dir)
    if os.path.isdir(full_mp):
        for f in os.listdir(full_mp):
            if f.endswith('.nii.gz'):
                f_stem = f.replace('.nii.gz', '').upper().replace('LABEL', '').replace('label', '')
                # Some masks carry a DWI suffix that the image lacks, and vice versa
                f_clean = f_stem.replace('DWI', '')
                dwi_stem = dwi_basename.replace('.nii.gz', '').upper()
                dwi_clean = dwi_stem.replace('DWI', '')
                if f_stem == dwi_stem or f_clean == dwi_stem or f_stem == dwi_clean or f_clean == dwi_clean:
                    mask_file = f
                    break
    
    if mask_file is None:
        print(f"  ⚠ Mask not found: {mask_dir}/ matching {dwi_basename}")
        continue
    
    mask_arr, mask_img = load_nifti(f"{mask_dir}/{mask_file}")
    if mask_arr is None:
        print(f"  ⚠ Failed to load mask: {mask_dir}/{mask_file}")
        continue
    print(f"  Mask: {mask_dir}/{mask_file}")
    
    # 3. Extract voxel-wise features (raw DWI signal + local texture only)
    features, indices = extract_voxel_features(dwi_arr, mask_arr)
    if len(features) < 30:
        print(f"  ⚠ Too few ROI voxels: {len(features)}")
        continue
    
    print(f"  ROI voxels: {len(features)}, feature dimensions: {features.shape[1]}")
    
    # 4. Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 5. Z-score standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 6. PCA (retain 95% of the variance)
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    features_pca = pca.fit_transform(features_scaled)
    print(f"  PCA: {features.shape[1]}→{features_pca.shape[1]} dims, explained variance={pca.explained_variance_ratio_.sum():.3f}")
    
    # 7. Determine the optimal K with GMM + BIC
    best_k, bic_scores = determine_optimal_k(features_pca, K_MIN, K_MAX)
    print(f"  Optimal K={best_k} (BIC: {[f'{b:.0f}' for b in bic_scores]})")
    
    # 8. K-means clustering
    labels, kmeans_model = apply_kmeans(features_pca, best_k)
    
    cluster_counts = np.bincount(labels)
    print(f"  Voxels per habitat: {cluster_counts}")
    
    # 9. Save the habitat masks
    ref_img = dwi_img if dwi_img is not None else mask_img
    mask_path = save_habitat_masks(labels, indices, ref_img, case_no, MASK_DIR, best_k)
    
    # 10. Try to plot the BIC curve
    try:
        plot_bic_curve(bic_scores, K_MIN, case_no, MASK_DIR)
    except Exception as e:
        pass
    
    # Record the result
    results.append({
        'Case_No.': case_no,
        'Name': name,
        'N_voxels': len(features),
        'N_features': features.shape[1],
        'PCA_dims': features_pca.shape[1],
        'Best_K': best_k,
        'BIC_scores': str([f'{b:.0f}' for b in bic_scores]),
        'Cluster_sizes': str(cluster_counts.tolist()),
        'Habitat_dir': os.path.join('habitat_masks', case_no),
    })
    
    # Save intermediate results every 50 cases
    if (idx + 1) % 50 == 0:
        pd.DataFrame(results).to_csv(
            os.path.join(OUTPUT_DIR, "habitat_cluster_info_partial.csv"), 
            index=False, encoding='utf-8-sig')
        print(f"  [intermediate save: {len(results)} cases]")

# ========== Save results ==========
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "habitat_cluster_info.csv"), 
                  index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"Done! Successfully processed {len(results)}/{len(df)} cases")
print(f"Results: {os.path.join(OUTPUT_DIR, 'habitat_cluster_info.csv')}")
print(f"Masks: {MASK_DIR}/")

# Distribution of K
print(f"\nK distribution:")
k_dist = df_results['Best_K'].value_counts().sort_index()
for k, v in k_dist.items():
    print(f"  K={k}: {v} cases ({v/len(df_results)*100:.1f}%)")
