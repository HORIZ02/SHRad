#!/usr/bin/env python3
"""
Habitat 3D radiomics feature extraction (replaces PyRadiomics).
Implemented by hand: shape, first-order statistics, GLCM texture (2D slices merged to 3D).
"""
import pandas as pd
import numpy as np
import nibabel as nib
import os, glob, warnings
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

warnings.filterwarnings('ignore')

# ========== Configuration ==========
MRI_ROOT = os.path.join(BASE, '使用数据/MRI数据')
MAPPING_FILE = os.path.join(BASE, 'data/file_mapping.csv')
OUTPUT_DIR = os.path.join(BASE, 'data/features')
os.makedirs(OUTPUT_DIR, exist_ok=True)

mapping = pd.read_csv(MAPPING_FILE)

def extract_features_from_mask(image_arr, mask_arr):
    """Extract features from a 3D image and a mask."""
    mask_bin = (mask_arr > 0)
    voxels = image_arr[mask_bin]
    
    if len(voxels) < 10:
        return None
    
    features = {}
    
    # ===== First-order statistics =====
    features['mean'] = float(np.mean(voxels))
    features['median'] = float(np.median(voxels))
    features['std'] = float(np.std(voxels))
    features['variance'] = float(np.var(voxels))
    features['skewness'] = float(np.mean(((voxels - np.mean(voxels)) / np.std(voxels)) ** 3))
    features['kurtosis'] = float(np.mean(((voxels - np.mean(voxels)) / np.std(voxels)) ** 4) - 3)
    features['range'] = float(np.ptp(voxels))
    features['min'] = float(np.min(voxels))
    features['max'] = float(np.max(voxels))
    features['p10'] = float(np.percentile(voxels, 10))
    features['p25'] = float(np.percentile(voxels, 25))
    features['p75'] = float(np.percentile(voxels, 75))
    features['p90'] = float(np.percentile(voxels, 90))
    features['energy'] = float(np.sum(voxels ** 2))
    features['entropy'] = float(-np.sum((voxels / np.sum(voxels)) * np.log(voxels / np.sum(voxels) + 1e-10)))
    features['rms'] = float(np.sqrt(np.mean(voxels ** 2)))
    features['uniformity'] = float(np.sum((voxels / np.sum(voxels)) ** 2))
    
    # ===== Shape features =====
    # Volume (voxel count)
    features['volume_voxels'] = int(np.sum(mask_bin))
    
    # Maximum 3D diameter
    coords = np.argwhere(mask_bin)
    if len(coords) > 1:
        max_dist = 0
        centroid = np.mean(coords, axis=0)
        # Ellipsoid axis lengths (PCA)
        coords_centered = coords - centroid
        cov = np.cov(coords_centered.T)
        evals, _ = np.linalg.eigh(cov)
        evals = np.sqrt(np.abs(evals)) * 2  # 2 standard deviations
        evals = np.sort(evals)[::-1]
        features['major_axis'] = float(evals[0]) if len(evals) > 0 else 0
        features['minor_axis'] = float(evals[1]) if len(evals) > 1 else 0
        features['least_axis'] = float(evals[2]) if len(evals) > 2 else 0
        features['elongation'] = float(evals[0] / evals[1]) if len(evals) > 1 and evals[1] > 0 else 1
        features['flatness'] = float(evals[1] / evals[2]) if len(evals) > 2 and evals[2] > 0 else 1
        
        # Compactness
        surface = np.sum(mask_bin) - np.sum(ndimage.binary_erosion(mask_bin))
        features['surface_area'] = int(surface)
        if surface > 0:
            features['compactness'] = float((surface ** 3) / (36 * np.pi * (np.sum(mask_bin) ** 2)))
            features['sphericity'] = float((36 * np.pi * (np.sum(mask_bin) ** 2)) ** (1/3) / surface)
    
    # ===== GLCM texture (2D per slice, then merged) =====
    # Discretization
    bin_width = 25
    img_discrete = np.floor(image_arr / bin_width).astype(np.int32)
    img_discrete = np.clip(img_discrete, 0, 255)
    
    glcm_props = []
    for z in range(mask_bin.shape[0]):
        slice_mask = mask_bin[z]
        if np.sum(slice_mask) < 10:
            continue
        slice_img = img_discrete[z]
        
        # Keep only the region inside the mask
        y_min, y_max = np.where(slice_mask)[0].min(), np.where(slice_mask)[0].max()
        x_min, x_max = np.where(slice_mask)[1].min(), np.where(slice_mask)[1].max()
        sub_img = slice_img[y_min:y_max+1, x_min:x_max+1]
        sub_mask = slice_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Set everything outside the mask to 0
        sub_img[~sub_mask] = 0
        
        try:
            glcm = graycomatrix(sub_img, distances=[1], angles=[0], levels=256, symmetric=True)
            props = {}
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                props[prop] = graycoprops(glcm, prop)[0, 0]
            glcm_props.append(props)
        except:
            pass
    
    if glcm_props:
        df_glcm = pd.DataFrame(glcm_props)
        for col in df_glcm.columns:
            features[f'glcm_{col}_mean'] = float(df_glcm[col].mean())
            features[f'glcm_{col}_std'] = float(df_glcm[col].std())
    
    return features

# ========== Main loop ==========
sequences = {
    'DWI': (os.path.join(BASE, 'data/dwi_habitat_masks'), 'DWI_path'),
    'DCE': (os.path.join(BASE, 'data/dce_habitat_masks'), 'DCE_path'),
}

for seq_name, (mask_root, img_col) in sequences.items():
    print(f"\n===== {seq_name} Habitat Radiomics =====")
    
    all_records = []
    case_list = sorted([d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))])
    
    for idx, case_no in enumerate(case_list):
        if (idx+1) % 50 == 0:
            print(f"  [{idx+1}/{len(case_list)}]")
        
        case_dir = os.path.join(mask_root, case_no)
        row = mapping[mapping['Case_No.'] == case_no]
        if len(row) == 0: continue
        row = row.iloc[0]
        
        img_rel = row[img_col]
        if not img_rel or img_rel == 'nan': continue
        img_full = os.path.join(MRI_ROOT, img_rel)
        if not os.path.exists(img_full): continue
        
        try:
            nii = nib.load(img_full)
            img_arr = np.asanyarray(nii.dataobj).astype(np.float64)
        except:
            continue
        
        # Whole tumor
        if seq_name == 'DWI':
            center = img_rel.split('/')[0]
            stem = os.path.basename(img_rel).replace('.nii.gz','').upper().replace('DWI','')
            mask_dir = os.path.join(MRI_ROOT, f"{center}/DWI/masks")
            if os.path.isdir(mask_dir):
                for f in os.listdir(mask_dir):
                    fs = f.upper().replace('LABEL','').replace('DWI','').replace('.NII.GZ','')
                    if fs == stem:
                        whole_mask = nib.load(os.path.join(mask_dir, f))
                        whole_arr = np.asanyarray(whole_mask.dataobj)
                        feats = extract_features_from_mask(img_arr, whole_arr)
                        if feats:
                            feats['Case_No.'] = case_no
                            feats['Sequence'] = seq_name
                            feats['Habitat'] = 'Whole_tumor'
                            all_records.append(feats)
                        break
        
        # Individual habitats
        for hf in sorted(glob.glob(os.path.join(case_dir, 'habitat_*.nii.gz'))):
            if 'habitat_all' in hf: continue
            hname = os.path.basename(hf).replace('.nii.gz','')
            try:
                h_nii = nib.load(hf)
                h_arr = np.asanyarray(h_nii.dataobj)
                feats = extract_features_from_mask(img_arr, h_arr)
                if feats:
                    feats['Case_No.'] = case_no
                    feats['Sequence'] = seq_name
                    feats['Habitat'] = hname
                    all_records.append(feats)
            except:
                continue
    
    if all_records:
        df = pd.DataFrame(all_records)
        out_file = os.path.join(OUTPUT_DIR, f"{seq_name.lower()}_habitat_radiomics.csv")
        df.to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f"\n{seq_name}: {len(df)} records -> {out_file}")
        print(f"  Covers {len(df['Case_No.'].unique())}/{len(case_list)} cases")
        print(f"  Number of features: {len([c for c in df.columns if c not in ['Case_No.','Sequence','Habitat']])}")
        print(f"  Counts by habitat type:")
        print(f"  {df['Habitat'].value_counts().to_string()}")

print(f"\n{'='*60}")
print("Done!")
