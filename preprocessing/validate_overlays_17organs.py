#!/usr/bin/env python3
import json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE = Path((__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + '/new_pipeline_800gb'))
RUN = BASE / "full_run_firstv1_rawlog1p_20260703_110639"
ROWMAP = RUN / "step3_rowmap_17organs"
OV = RUN / "step4_overlays_17organs"
FEATURES = RUN / "step1_features" / "master_feature_store.pt"

rowmap = pd.read_csv(ROWMAP / "rowmap_manifest.tsv", sep="\t", dtype={"feature_index":"int64"}).fillna("")
N = len(rowmap)
print("rowmap N", N, "sections", rowmap["section_id"].nunique(), "organs", rowmap["organ_id"].nunique(), "techs", rowmap["tech_id"].nunique())
print("feature_index range", int(rowmap["feature_index"].min()), int(rowmap["feature_index"].max()))

# feature store size from report
report = json.loads((RUN/"step1_features"/"feature_store_report.json").read_text())
F = int(report["features"])
print("feature_store_features", F)

m2 = torch.load(OV / "m2_pool32_homo_hetero_overlay.pt", map_location="cpu", weights_only=False)

print("\n=== 1. 形状检查 ===")
print("m2 keys", sorted(m2.keys()))
shapes = {
 "m2_cand_rows": m2["candidate_row_ids"].shape,
 "m2_cand_fidx": m2["candidate_feature_indices"].shape,
 "m2_cand_mask": m2["candidate_valid_mask"].shape,
 "m2_cand_dist": m2["candidate_spatial_distance"].shape,
 "m2_cand_cos": m2["candidate_feature_cosine_sim"].shape,
 "m2_homo_rows": m2["homo_row_ids"].shape,
 "m2_homo_fidx": m2["homo_feature_indices"].shape,
 "m2_homo_mask": m2["homo_valid_mask"].shape,
 "m2_hetero_rows": m2["hetero_row_ids"].shape,
 "m2_hetero_fidx": m2["hetero_feature_indices"].shape,
 "m2_hetero_mask": m2["hetero_valid_mask"].shape,
}
for k,v in shapes.items(): print(k, tuple(v))

print("\n=== 2. 有效数分布（完整） ===")
for name, mask in [
    ("m2_candidate", m2["candidate_valid_mask"]),
    ("m2_homo", m2["homo_valid_mask"]),
    ("m2_hetero", m2["hetero_valid_mask"]),
]:
    c = mask.sum(dim=1).numpy()
    print(f"{name}: min={c.min()} p50={np.percentile(c,50):.1f} p90={np.percentile(c,90):.1f} max={c.max()} full={int((c==mask.shape[1]).sum())} zero={int((c==0).sum())}")

print("\n=== 3. 越界检查（完整） ===")
def check_oob(name, vals, mask, lo, hi):
    valid = vals[mask]
    if valid.numel()==0:
        print(f"{name}: no valid entries"); return True
    ok = bool((valid>=lo).all() and (valid<hi).all())
    print(f"{name}: valid_min={int(valid.min())} valid_max={int(valid.max())} expected[{lo},{hi}) ok={ok}")
    return ok

check_oob("m2_cand_rows", m2["candidate_row_ids"], m2["candidate_valid_mask"], 0, N)
check_oob("m2_cand_fidx", m2["candidate_feature_indices"], m2["candidate_valid_mask"], 0, F)
check_oob("m2_homo_rows", m2["homo_row_ids"], m2["homo_valid_mask"], 0, N)
check_oob("m2_homo_fidx", m2["homo_feature_indices"], m2["homo_valid_mask"], 0, F)
check_oob("m2_hetero_rows", m2["hetero_row_ids"], m2["hetero_valid_mask"], 0, N)
check_oob("m2_hetero_fidx", m2["hetero_feature_indices"], m2["hetero_valid_mask"], 0, F)

print("\n=== 4. 距离/cosine 有限性检查（完整） ===")
for name, vals, mask in [
    ("m2_cand_dist", m2["candidate_spatial_distance"], m2["candidate_valid_mask"]),
    ("m2_cand_cos", m2["candidate_feature_cosine_sim"], m2["candidate_valid_mask"]),
]:
    v = vals[mask]
    finite = bool(torch.isfinite(v).all()) if v.numel() else True
    print(f"{name}: finite={finite} min={v.min().item():.6f} max={v.max().item():.6f}")

print("\n=== 5. 同 section 空间一致性 + homo/hetero 子集与互斥（抽样 20000 行） ===")
rng = random.Random(42)
idx = np.array(sorted(rng.sample(range(N), 20000)), dtype=np.int64)
sec = rowmap["section_id"].astype(str).to_numpy()
fail_sec = 0; fail_subset = 0; fail_disjoint = 0; fail_fidx = 0
for i in idx:
    cand = set(m2["candidate_row_ids"][i][m2["candidate_valid_mask"][i]].tolist())
    homo = set(m2["homo_row_ids"][i][m2["homo_valid_mask"][i]].tolist())
    hetero = set(m2["hetero_row_ids"][i][m2["hetero_valid_mask"][i]].tolist())
    # same section for all candidate rows
    for rr in cand:
        if sec[rr] != sec[i]: fail_sec += 1; break
    if not homo.issubset(cand): fail_subset += 1
    if not hetero.issubset(cand): fail_subset += 1
    if homo & hetero: fail_disjoint += 1
    # candidate feature index consistency: cand_fidx should equal rowmap feature_index of cand row
    cf = m2["candidate_feature_indices"][i].tolist()
    cr = m2["candidate_row_ids"][i].tolist()
    for rr, ff in zip(cr, cf):
        if rr >= 0 and int(rowmap["feature_index"].iloc[rr]) != ff:
            fail_fidx += 1; break
print(f"sample_rows={len(idx)} same_section_fail={fail_sec} subset_fail={fail_subset} disjoint_fail={fail_disjoint} fidx_mismatch={fail_fidx}")

print("\n=== 6. 结论 ===")
print("PASS_SAMPLE_CHECKS" if fail_sec==0 and fail_subset==0 and fail_disjoint==0 and fail_fidx==0 else "SAMPLE_CHECK_FAIL")
