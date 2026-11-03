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

spagenx = torch.load(OV / "spagenx_pool32_homo_hetero_overlay.pt", map_location="cpu", weights_only=False)

print("\n=== 1. 形状检查 ===")
print("spagenx keys", sorted(spagenx.keys()))
shapes = {
 "spagenx_cand_rows": spagenx["candidate_row_ids"].shape,
 "spagenx_cand_fidx": spagenx["candidate_feature_indices"].shape,
 "spagenx_cand_mask": spagenx["candidate_valid_mask"].shape,
 "spagenx_cand_dist": spagenx["candidate_spatial_distance"].shape,
 "spagenx_cand_cos": spagenx["candidate_feature_cosine_sim"].shape,
 "spagenx_homo_rows": spagenx["homo_row_ids"].shape,
 "spagenx_homo_fidx": spagenx["homo_feature_indices"].shape,
 "spagenx_homo_mask": spagenx["homo_valid_mask"].shape,
 "spagenx_hetero_rows": spagenx["hetero_row_ids"].shape,
 "spagenx_hetero_fidx": spagenx["hetero_feature_indices"].shape,
 "spagenx_hetero_mask": spagenx["hetero_valid_mask"].shape,
}
for k,v in shapes.items(): print(k, tuple(v))

print("\n=== 2. 有效数分布（完整） ===")
for name, mask in [
    ("spagenx_candidate", spagenx["candidate_valid_mask"]),
    ("spagenx_homo", spagenx["homo_valid_mask"]),
    ("spagenx_hetero", spagenx["hetero_valid_mask"]),
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

check_oob("spagenx_cand_rows", spagenx["candidate_row_ids"], spagenx["candidate_valid_mask"], 0, N)
check_oob("spagenx_cand_fidx", spagenx["candidate_feature_indices"], spagenx["candidate_valid_mask"], 0, F)
check_oob("spagenx_homo_rows", spagenx["homo_row_ids"], spagenx["homo_valid_mask"], 0, N)
check_oob("spagenx_homo_fidx", spagenx["homo_feature_indices"], spagenx["homo_valid_mask"], 0, F)
check_oob("spagenx_hetero_rows", spagenx["hetero_row_ids"], spagenx["hetero_valid_mask"], 0, N)
check_oob("spagenx_hetero_fidx", spagenx["hetero_feature_indices"], spagenx["hetero_valid_mask"], 0, F)

print("\n=== 4. 距离/cosine 有限性检查（完整） ===")
for name, vals, mask in [
    ("spagenx_cand_dist", spagenx["candidate_spatial_distance"], spagenx["candidate_valid_mask"]),
    ("spagenx_cand_cos", spagenx["candidate_feature_cosine_sim"], spagenx["candidate_valid_mask"]),
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
    cand = set(spagenx["candidate_row_ids"][i][spagenx["candidate_valid_mask"][i]].tolist())
    homo = set(spagenx["homo_row_ids"][i][spagenx["homo_valid_mask"][i]].tolist())
    hetero = set(spagenx["hetero_row_ids"][i][spagenx["hetero_valid_mask"][i]].tolist())
    # same section for all candidate rows
    for rr in cand:
        if sec[rr] != sec[i]: fail_sec += 1; break
    if not homo.issubset(cand): fail_subset += 1
    if not hetero.issubset(cand): fail_subset += 1
    if homo & hetero: fail_disjoint += 1
    # candidate feature index consistency: cand_fidx should equal rowmap feature_index of cand row
    cf = spagenx["candidate_feature_indices"][i].tolist()
    cr = spagenx["candidate_row_ids"][i].tolist()
    for rr, ff in zip(cr, cf):
        if rr >= 0 and int(rowmap["feature_index"].iloc[rr]) != ff:
            fail_fidx += 1; break
print(f"sample_rows={len(idx)} same_section_fail={fail_sec} subset_fail={fail_subset} disjoint_fail={fail_disjoint} fidx_mismatch={fail_fidx}")

print("\n=== 6. 结论 ===")
print("PASS_SAMPLE_CHECKS" if fail_sec==0 and fail_subset==0 and fail_disjoint==0 and fail_fidx==0 else "SAMPLE_CHECK_FAIL")
