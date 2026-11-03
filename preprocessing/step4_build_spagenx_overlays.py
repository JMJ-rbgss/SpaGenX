#!/usr/bin/env python3
"""Build SpaGenX index-only overlays from the shared rowmap cache.

SpaGenX: spatial candidate K32, then image-feature cosine homo12/hetero8.
No feature tensors are copied into overlays.
"""
import argparse, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

K_M2 = 32
K_HOMO = 12
K_HETERO = 8


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rowmap_root", required=True)
    ap.add_argument("--feature_store_pt", required=True)
    ap.add_argument("--feature_manifest_tsv", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--sim_block", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rowmap_root = Path(args.rowmap_root)
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    rowmap = pd.read_csv(rowmap_root / "rowmap_manifest.tsv", sep="\t").fillna("")
    N = len(rowmap)
    if N == 0:
        raise SystemExit("empty rowmap")
    for c in ["row_id", "feature_index", "section_id", "coord_x", "coord_y"]:
        if c not in rowmap.columns:
            raise SystemExit(f"rowmap missing {c}")
    rowmap = rowmap.sort_values("row_id").reset_index(drop=True)
    if rowmap["row_id"].astype(int).tolist() != list(range(N)):
        raise SystemExit("row_id must be 0..N-1")
    feature_idx = rowmap["feature_index"].astype(np.int64).to_numpy()
    coords = rowmap[["coord_x", "coord_y"]].astype(float).to_numpy(dtype=np.float32)
    sections = rowmap["section_id"].astype(str).tolist()
    if not np.isfinite(coords).all():
        raise SystemExit("rowmap has non-finite coordinates; cannot build spatial overlays")
    by_section = defaultdict(list)
    section_code = {}
    section_ids = []
    for rid, sec in enumerate(sections):
        if sec not in section_code:
            section_code[sec] = len(section_ids); section_ids.append(sec)
        by_section[section_code[sec]].append(rid)
    row_ids = torch.arange(N, dtype=torch.long)
    sec_tensor = torch.as_tensor([section_code[s] for s in sections], dtype=torch.int32)
    cand_rows = torch.full((N, K_M2), -1, dtype=torch.long)
    cand_fidx = torch.full((N, K_M2), -1, dtype=torch.long)
    cand_mask = torch.zeros((N, K_M2), dtype=torch.bool)
    cand_dist = torch.full((N, K_M2), float("inf"), dtype=torch.float32)
    cand_sim = torch.full((N, K_M2), float("nan"), dtype=torch.float32)
    homo_rows = torch.full((N, K_HOMO), -1, dtype=torch.long)
    homo_fidx = torch.full((N, K_HOMO), -1, dtype=torch.long)
    homo_mask = torch.zeros((N, K_HOMO), dtype=torch.bool)
    hetero_rows = torch.full((N, K_HETERO), -1, dtype=torch.long)
    hetero_fidx = torch.full((N, K_HETERO), -1, dtype=torch.long)
    hetero_mask = torch.zeros((N, K_HETERO), dtype=torch.bool)
    resource = []
    t_all = time.time()
    for sc, ridx_list in by_section.items():
        t0 = time.time(); ridx = np.asarray(ridx_list, dtype=np.int64); n = len(ridx)
        local_coords = coords[ridx]
        if n > 1:
            tree = cKDTree(local_coords)
            qk = min(n, K_M2 + 1)
            dist, loc = tree.query(local_coords, k=qk)
            if qk == 1:
                dist = dist[:, None]; loc = loc[:, None]
            for i in range(n):
                keep = loc[i] != i
                loc_i = loc[i][keep]; dist_i = dist[i][keep]
                if loc_i.size:
                    take2 = min(K_M2, loc_i.size)
                    rr2 = ridx[loc_i[:take2]]
                    cand_rows[ridx[i], :take2] = torch.from_numpy(rr2)
                    cand_fidx[ridx[i], :take2] = torch.from_numpy(feature_idx[rr2])
                    cand_mask[ridx[i], :take2] = True
                    cand_dist[ridx[i], :take2] = torch.from_numpy(dist_i[:take2].astype(np.float32))
        resource.append({"stage": "spatial_kdtree", "section_id": section_ids[sc], "n_rows": n, "time_sec": round(time.time() - t0, 4)})
    fs = load_pt(args.feature_store_pt)
    features = fs["features"] if isinstance(fs, dict) else fs
    if int(features.shape[0]) <= int(feature_idx.max()) or int(features.shape[1]) != 1536:
        raise SystemExit(f"feature store shape invalid: {tuple(features.shape)} max_feature_index={int(feature_idx.max())}")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    for sc, ridx_list in by_section.items():
        t0 = time.time(); ridx = np.asarray(ridx_list, dtype=np.int64)
        for start in range(0, len(ridx), args.sim_block):
            part = ridx[start:start + args.sim_block]
            valid = cand_mask[part].clone()
            safe = cand_fidx[part].clone(); safe[~valid] = 0
            with torch.no_grad():
                c = features[torch.from_numpy(feature_idx[part])].to(device=device, dtype=torch.float32)
                nf = features[safe.reshape(-1)].reshape(len(part), K_M2, 1536).to(device=device, dtype=torch.float32)
                c = torch.nn.functional.normalize(c, dim=1, eps=1e-12)
                nf = torch.nn.functional.normalize(nf, dim=2, eps=1e-12)
                sim = (nf * c[:, None, :]).sum(dim=2).cpu()
            sim[~valid] = float("nan")
            cand_sim[part] = sim
            hi = sim.clone(); hi[~valid] = -float("inf")
            hv, hp = torch.topk(hi, K_HOMO, dim=1, largest=True)
            hm = torch.isfinite(hv)
            cr = cand_rows[part]; cf = cand_fidx[part]
            hr = torch.gather(cr, 1, hp); hf = torch.gather(cf, 1, hp)
            hr[~hm] = -1; hf[~hm] = -1
            homo_rows[part] = hr; homo_fidx[part] = hf; homo_mask[part] = hm
            selected = torch.zeros_like(valid); selected.scatter_(1, hp, hm)
            hetero_valid = valid & (~selected)
            lo = sim.clone(); lo[~hetero_valid] = float("inf")
            lv, lp = torch.topk(lo, K_HETERO, dim=1, largest=False)
            lm = torch.isfinite(lv)
            lr = torch.gather(cr, 1, lp); lf = torch.gather(cf, 1, lp)
            lr[~lm] = -1; lf[~lm] = -1
            hetero_rows[part] = lr; hetero_fidx[part] = lf; hetero_mask[part] = lm
        resource.append({"stage": "m2_feature_cosine", "section_id": section_ids[sc], "n_rows": len(ridx), "time_sec": round(time.time() - t0, 4)})
    common_meta = {"rowmap_root": str(rowmap_root), "feature_store_pt": str(args.feature_store_pt), "N": N, "n_sections": len(section_ids), "section_codebook": section_ids, "uses_gene_expression_for_neighbors": False, "copies_neighbor_feature_tensor": False, "total_time_sec": round(time.time() - t_all, 3)}
    m2 = {"version": "m2_pool32_homo_hetero_overlay", "row_ids": row_ids, "candidate_row_ids": cand_rows, "candidate_feature_indices": cand_fidx, "candidate_valid_mask": cand_mask, "candidate_spatial_distance": cand_dist, "candidate_feature_cosine_sim": cand_sim, "homo_row_ids": homo_rows, "homo_feature_indices": homo_fidx, "homo_valid_mask": homo_mask, "hetero_row_ids": hetero_rows, "hetero_feature_indices": hetero_fidx, "hetero_valid_mask": hetero_mask, "metadata": {**common_meta, "candidate_k": K_M2, "homo_k": K_HOMO, "hetero_k": K_HETERO}}
    torch.save(m2, out / "m2_pool32_homo_hetero_overlay.pt")
    pd.DataFrame(resource).to_csv(out / "overlay_build_resource_profile.tsv", sep="\t", index=False)
    def count_summary(name, mask, k):
        c = mask.sum(dim=1).numpy()
        return {"name": name, "n_rows": N, "k": k, "min": int(c.min()), "p50": float(np.percentile(c, 50)), "p90": float(np.percentile(c, 90)), "max": int(c.max()), "n_full": int((c == k).sum()), "n_zero": int((c == 0).sum())}
    pd.DataFrame([count_summary("m2_candidate", cand_mask, K_M2), count_summary("m2_homo", homo_mask, K_HOMO), count_summary("m2_hetero", hetero_mask, K_HETERO)]).to_csv(out / "m2_homo_hetero_distribution.tsv", sep="\t", index=False)
    final = {"task_status": "PASS", "m2_overlay_built": True, "m2_candidate_k": K_M2, "m2_homo_k": K_HOMO, "m2_hetero_k": K_HETERO, "m2_full_pool32_ready": True, "copies_neighbor_feature_tensor": False, "ready_for_training": False}
    (out / "final_decision.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
