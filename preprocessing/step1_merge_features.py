#!/usr/bin/env python3
"""Merge first-version-compatible UNI2-h sample shards."""
import argparse, json
from pathlib import Path
from collections import Counter
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--output_pt", required=True)
    ap.add_argument("--manifest_tsv", required=True)
    args = ap.parse_args()
    shard_dir = Path(args.shard_dir)
    out_pt = Path(args.output_pt)
    manifest_path = Path(args.manifest_tsv)
    paths = [p for p in sorted(shard_dir.glob("*.pt")) if p.resolve() != out_pt.resolve()]
    feats, coords = [], []
    skipped_empty = []
    store_lists = {k: [] for k in ["sid", "sample_id", "dataset_id", "organ_id", "tech_id", "patient_id", "section_id", "spot_id", "image_route", "source_image_path"]}
    rows, failures = [], []
    global_idx = 0
    for p in paths:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            f = obj["features"].float()
            c = obj["coords"].float()
            n = int(f.shape[0])
            if n == 0:
                skipped_empty.append({"path": str(p), "reason": "zero_valid_features_after_first_version_white_filter"})
                continue
            if f.ndim != 2 or f.shape[1] != 1536 or not torch.isfinite(f).all().item() or c.shape[0] != n:
                failures.append({"path": str(p), "error": f"bad_or_nonfinite_shape_features={tuple(f.shape)} coords={tuple(c.shape)}"})
                continue
            manifest_rows = obj.get("manifest_rows", [])
            if len(manifest_rows) != n:
                spots = [str(x) for x in obj.get("spot_id", [str(i) for i in range(n)])]
                sid = str(obj.get("sid", obj.get("sample_id", p.stem)))
                manifest_rows = []
                for i, xy in enumerate(c.tolist()):
                    manifest_rows.append({"sid": sid, "sample_id": sid, "spot_id": spots[i], "coord_x": xy[0], "coord_y": xy[1], "dataset_id": obj.get("dataset", ""), "organ_id": obj.get("organ", ""), "tech_id": obj.get("tech", ""), "section_id": sid, "image_path": obj.get("wsi_path", ""), "extraction_status": "OK", "feature_dim": 1536, "shard_path": str(p), "local_feature_index": i})
            feats.append(f); coords.append(c)
            for i, rec in enumerate(manifest_rows):
                rec = dict(rec)
                rec["global_feature_index"] = global_idx + i
                rec.setdefault("sample_id", rec.get("sid", obj.get("sample_id", p.stem)))
                rec.setdefault("sid", rec.get("sample_id", obj.get("sid", p.stem)))
                rec.setdefault("shard_path", str(p))
                rec.setdefault("local_feature_index", i)
                rows.append(rec)
                store_lists["sid"].append(str(rec.get("sid", "")))
                store_lists["sample_id"].append(str(rec.get("sample_id", rec.get("sid", ""))))
                store_lists["dataset_id"].append(str(rec.get("dataset_id", rec.get("dataset", ""))))
                store_lists["organ_id"].append(str(rec.get("organ_id", rec.get("organ", ""))))
                store_lists["tech_id"].append(str(rec.get("tech_id", rec.get("tech", ""))))
                store_lists["patient_id"].append(str(rec.get("patient_id", "")))
                store_lists["section_id"].append(str(rec.get("section_id", rec.get("sid", ""))))
                store_lists["spot_id"].append(str(rec.get("spot_id", "")))
                store_lists["image_route"].append(str(rec.get("image_route", "")))
                store_lists["source_image_path"].append(str(rec.get("image_path", rec.get("wsi_path", ""))))
            global_idx += n
        except Exception as e:
            failures.append({"path": str(p), "error": repr(e)})
    features = torch.cat(feats, 0) if feats else torch.empty((0, 1536), dtype=torch.float32)
    coords_t = torch.cat(coords, 0) if coords else torch.empty((0, 2), dtype=torch.float32)
    master = {"features": features.float(), "coords": coords_t.float(), "patch_valid": torch.ones(features.shape[0], dtype=torch.bool), **store_lists, "metadata": {"feature_store_type": "master_unique_valid_spot_store", "feature_dim": 1536, "dtype": "float32", "encoder": "UNI2-h vit_giant_patch14_dinov2", "patch_size": 224, "view_policy": "orig_only", "white_gray_thresh": 205, "white_ratio_thresh": 0.8, "coords_norm_policy": "sid_local_minmax_for_index_and_audit_only_not_default_model_input", "merged_from_shards": [str(p) for p in paths]}}
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(master, out_pt)
    df = pd.DataFrame(rows)
    preferred = ["global_feature_index", "sid", "sample_id", "dataset_id", "organ_id", "tech_id", "patient_id", "section_id", "spot_id", "image_route", "image_path", "coord_x", "coord_y", "coord_space", "coord_transform_source", "coord_transform_confidence", "read_backend", "patch_size", "read_size", "target_mpp", "mpp_used", "mpp_policy", "physical_scale_confidence", "best_level", "white_ratio", "boundary_padded", "feature_dim", "extraction_status", "shard_path", "local_feature_index"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df.to_csv(manifest_path, sep="\t", index=False, columns=cols)
    fail_path = manifest_path.with_suffix(".failures.json")
    fail_path.write_text(json.dumps(failures, indent=2) + "\n")
    (manifest_path.parent / "feature_store_skipped_empty_shards.json").write_text(json.dumps(skipped_empty, indent=2) + "\n")
    route_counts = Counter(df.get("image_route", pd.Series(dtype=str)).astype(str).tolist()) if not df.empty else Counter()
    report = {"output_pt": str(out_pt), "features": int(features.shape[0]), "samples": int(df["sid"].nunique()) if "sid" in df else 0, "failures": len(failures), "skipped_empty_shards": len(skipped_empty), "route_counts": dict(route_counts), "feature_dim_ok": bool(features.ndim == 2 and features.shape[1] == 1536), "no_nan_inf": bool(torch.isfinite(features).all().item()) if features.numel() else True}
    (manifest_path.parent / "feature_store_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
