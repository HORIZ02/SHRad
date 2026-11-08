# -*- coding: utf-8 -*-
"""
21_spatial_features_k3.py  —  HER2-low habitat spatial-structure feature extraction (K=3)

Adapted from the validated 21_spatial_features_multik.py script of the prostate
project (<PROSTATE_ROOT>). The feature mathematics are identical; only the
following were changed:
  1) mask lookup: subdirectory layout data/{mod}_habitat_masks_k3/{Case}/habitat_{L}.nii.gz
  2) case list: scan the mask subdirectory names
  3) modalities: DWI / DCE (tag = dwi / dce)
  4) labels: K=3 -> ['L','M','H']

42 features per case per modality:
  single-habitat morphology 11 x 3 = 33
  pairwise habitat interactions 3 x 3 = 9   (interface, interface_norm, centroid_dist for LM / LH / MH)
DWI + DCE = 84 features per case

Key conventions (same as the prostate project, do not change):
  - one shared bbox crop for all habitats (based on the L|M|H union, with 2-voxel padding)
  - surface area via marching_cubes (counting voxel faces systematically underestimates it -> sphericity > 1)
  - distances via distance_transform_edt(sampling=spacing) -> physical units in mm
  - interface computed as binary_dilation(A) & B
  - radial quantities divided by the equivalent radius R_eq to remove the effect of tumor size

Output: data/features/habitat_spatial_features_k3.csv
Supports resuming (reads the existing CSV and skips (Case, modality) pairs)
+ incremental write-out after each modality
"""
import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage import measure
from skimage.measure import euler_number
BASE = os.environ.get("SHRAD_ROOT", os.path.abspath(os.getcwd()))

warnings.filterwarnings('ignore')

BASE_DIR = BASE
MODS     = ['dwi', 'dce']          # lowercase directory name = feature-name prefix
LABELS   = ['L', 'M', 'H']         # K=3
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

OUT_CSV  = os.path.join(BASE_DIR, 'data', 'features', 'habitat_spatial_features_k3.csv')
LOG_TXT  = os.path.join(BASE_DIR, 'data', 'features', 'spatial_features_k3_log.txt')

SINGLE_FEATS = ['n_voxels', 'n_components', 'largest_frac', 'euler', 'surface', 'sphericity',
                'fractal_dim', 'centroid_dist', 'radial_mean', 'radial_sd', 'edge_dist_mean']


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_TXT, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def surface_area(mask, spacing):
    if not mask.any():
        return 0.0
    try:
        padded = np.pad(mask.astype(np.float32), 1)
        v, f, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing)
        return float(measure.mesh_surface_area(v, f))
    except Exception:
        return np.nan


def fractal_dim(mask):
    if not mask.any():
        return np.nan
    coords = np.argwhere(mask)
    mn, mx = coords.min(0), coords.max(0) + 1
    crop = mask[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    counts, sizes = [], []
    for box in [2, 4, 8, 16]:
        if any(s < box for s in crop.shape):
            continue
        d = [crop.shape[i] // box * box for i in range(3)]
        if min(d) == 0:
            continue
        sub = crop[:d[0], :d[1], :d[2]]
        counts.append(sub.reshape(d[0] // box, box, d[1] // box, box, d[2] // box, box)
                         .max(axis=(1, 3, 5)).sum())
        sizes.append(box)
    if len(counts) < 2:
        return np.nan
    return float(np.polyfit(np.log(1.0 / np.array(sizes, float)),
                            np.log(np.array(counts, float)), 1)[0])


def process_case(case, mod):
    """Return a feature dict, or None (missing file / empty tumor)."""
    mdir = os.path.join(BASE_DIR, 'data', f'{mod}_habitat_masks_k3', case)

    masks, spacing = {}, None
    for L in LABELS:
        fp = os.path.join(mdir, f'habitat_{L}.nii.gz')
        if not os.path.exists(fp):
            return None, f'missing habitat_{L}.nii.gz'
        im = nib.load(fp)
        masks[L] = im.get_fdata() > 0.5
        if spacing is None:
            spacing = tuple(float(s) for s in im.header.get_zooms()[:3])

    tumor = np.zeros_like(masks[LABELS[0]], dtype=bool)
    for L in LABELS:
        tumor |= masks[L]
    nv_t = int(tumor.sum())
    if nv_t == 0:
        return None, 'empty tumor'

    # ---- Shared bbox crop (critical: must be based on the L|M|H union) ----
    cs = np.argwhere(tumor)
    mn = np.clip(cs.min(0) - 2, 0, None).astype(int)
    mx = (cs.max(0) + 3).astype(int)
    sl = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    M = {L: masks[L][sl] for L in LABELS}
    Tc = np.zeros_like(M[LABELS[0]], dtype=bool)
    for L in LABELS:
        Tc |= M[L]

    vox_vol = spacing[0] * spacing[1] * spacing[2]
    R_eq = (3 * nv_t * vox_vol / (4 * np.pi)) ** (1 / 3)
    tcent = (np.argwhere(Tc) * np.array(spacing)).mean(axis=0)
    boundary = Tc & ~ndimage.binary_erosion(Tc, structure=STRUCT26)
    dist_map = ndimage.distance_transform_edt(~boundary, sampling=spacing) if boundary.any() else None

    rec = {'Case_No.': case, 'modality': mod.upper()}
    surf = {}

    # ---- Single-habitat morphology (11 x 3) ----
    for L in LABELS:
        msk = M[L]
        nv = int(msk.sum())
        if nv == 0:
            for kk in SINGLE_FEATS:
                rec[f'{mod}_{L}_{kk}'] = 0 if kk in ('n_voxels', 'n_components', 'euler') else np.nan
            surf[L] = 0.0
            continue
        lab, ncomp = ndimage.label(msk, structure=STRUCT26)
        sizes = ndimage.sum(msk, lab, range(1, ncomp + 1))
        eul = float(euler_number(msk, connectivity=3))
        sa = surface_area(msk, spacing)
        surf[L] = sa if sa == sa else 0.0
        Rh = (3 * nv * vox_vol / (4 * np.pi)) ** (1 / 3)
        sph = (4 * np.pi * Rh ** 2) / sa if (sa and sa > 0) else np.nan
        fd = fractal_dim(msk)
        hc = (np.argwhere(msk) * np.array(spacing)).mean(axis=0)
        cdist = float(np.linalg.norm(hc - tcent) / R_eq) if R_eq > 0 else np.nan
        d = np.linalg.norm((np.argwhere(msk) * np.array(spacing)) - tcent, axis=1)
        rmean = float(d.mean() / R_eq) if R_eq > 0 else np.nan
        rsd = float(d.std(ddof=1) / R_eq) if (len(d) > 1 and R_eq > 0) else np.nan
        edm = float(dist_map[msk].mean()) if dist_map is not None else np.nan
        rec.update({
            f'{mod}_{L}_n_voxels': nv,
            f'{mod}_{L}_n_components': int(ncomp),
            f'{mod}_{L}_largest_frac': round(float(sizes.max()) / nv, 4),
            f'{mod}_{L}_euler': eul,
            f'{mod}_{L}_surface': round(sa, 3) if sa == sa else np.nan,
            f'{mod}_{L}_sphericity': round(sph, 4) if sph == sph else np.nan,
            f'{mod}_{L}_fractal_dim': round(fd, 4) if fd == fd else np.nan,
            f'{mod}_{L}_centroid_dist': round(cdist, 4) if cdist == cdist else np.nan,
            f'{mod}_{L}_radial_mean': round(rmean, 4) if rmean == rmean else np.nan,
            f'{mod}_{L}_radial_sd': round(rsd, 4) if rsd == rsd else np.nan,
            f'{mod}_{L}_edge_dist_mean': round(edm, 4) if edm == edm else np.nan,
        })

    # ---- Pairwise habitat interactions (3 x 3) ----
    sp2 = sorted(spacing)
    for A, B in itertools.combinations(LABELS, 2):
        A_dil = ndimage.binary_dilation(M[A], structure=STRUCT26)
        iface = int(np.sum(A_dil & M[B])) * sp2[0] * sp2[1]
        avg_s = (surf.get(A, 0) + surf.get(B, 0)) / 2
        rec[f'{mod}_{A}_{B}_interface'] = round(iface, 3)
        rec[f'{mod}_{A}_{B}_interface_norm'] = round(iface / avg_s, 4) if avg_s > 0 else np.nan
        if M[A].any() and M[B].any():
            ca = (np.argwhere(M[A]) * np.array(spacing)).mean(axis=0)
            cb = (np.argwhere(M[B]) * np.array(spacing)).mean(axis=0)
            rec[f'{mod}_{A}_{B}_centroid_dist'] = (
                round(float(np.linalg.norm(ca - cb) / R_eq), 4) if R_eq > 0 else np.nan)
        else:
            rec[f'{mod}_{A}_{B}_centroid_dist'] = np.nan

    return rec, None


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Case list = DWI mask subdirectory names (intersected with DCE)
    d_dwi = os.path.join(BASE_DIR, 'data', 'dwi_habitat_masks_k3')
    d_dce = os.path.join(BASE_DIR, 'data', 'dce_habitat_masks_k3')
    s_dwi = {x for x in os.listdir(d_dwi) if os.path.isdir(os.path.join(d_dwi, x))}
    s_dce = {x for x in os.listdir(d_dce) if os.path.isdir(os.path.join(d_dce, x))}
    cases = sorted(s_dwi & s_dce)
    log('=' * 66)
    log(f'K=3 habitat spatial-structure feature extraction | DWI+DCE')
    log(f'  Cases: DWI {len(s_dwi)} / DCE {len(s_dce)} / intersection {len(cases)}')
    if s_dwi ^ s_dce:
        log(f'  ⚠ Mismatch between modalities: DWI only {sorted(s_dwi - s_dce)[:5]} | DCE only {sorted(s_dce - s_dwi)[:5]}')
    log(f'  LABELS={LABELS}  {11*3 + 3*3} features per case per modality')
    log(f'  Output: {OUT_CSV}')

    # ---- Resume support ----
    rows, done = [], set()
    if os.path.exists(OUT_CSV):
        try:
            prev = pd.read_csv(OUT_CSV)
            if len(prev):
                rows = prev.to_dict('records')
                done = set(zip(prev['Case_No.'].astype(str), prev['modality'].astype(str)))
                log(f'  Resuming: {len(prev)} rows already present ({len(done)} Case x modality pairs)')
        except Exception as e:
            log(f'  ⚠ Failed to read the existing CSV, starting from scratch: {e}')

    n_ok = n_skip = n_fail = 0
    for i, case in enumerate(cases):
        for mod in MODS:
            if (case, mod.upper()) in done:
                n_skip += 1
                continue
            try:
                rec, why = process_case(case, mod)
            except Exception as e:
                n_fail += 1
                log(f'  ✗ [{case}/{mod.upper()}] exception: {type(e).__name__}: {e}')
                continue
            if rec is None:
                n_fail += 1
                log(f'  ✗ [{case}/{mod.upper()}] skipped: {why}')
                continue
            rows.append(rec)
            n_ok += 1
            # Incremental write-out after each modality (can be interrupted at any time)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        if (i + 1) % 25 == 0 or (i + 1) == len(cases):
            el = time.time() - t0
            eta = el / max(i + 1, 1) * (len(cases) - i - 1)
            log(f'  Progress {i+1}/{len(cases)} | added {n_ok} skipped {n_skip} failed {n_fail} '
                f'| elapsed {el/60:.1f}min | ETA {eta/60:.1f}min')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log('=' * 66)
    log(f'Done: {len(df)} rows -> {OUT_CSV}')
    log(f'  added {n_ok} / skipped {n_skip} / failed {n_fail}')
    if len(df):
        log(f'  Columns {df.shape[1]} (including Case_No./modality) | cases {df["Case_No."].nunique()}')
        log(f'  Modality distribution: {df["modality"].value_counts().to_dict()}')
    log(f'  Total time {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
