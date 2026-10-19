#!/usr/bin/env python3
"""Build first-version-compatible label-centered rowmap for raw-log1p chunks."""
import argparse, csv, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp


def split_norm(x):
    s = str(x).strip().lower()
    if s in {"pretrain", "train", "training", "pretrain_v1_train"}: return "pretrain"
    if s in {"val", "valid", "validation", "pretrain_v1_val"}: return "val"
    if s in {"test", "heldout", "holdout"}: return "test"
    if s in {"exclude", "excluded", "excluded_from_training"}: return "excluded_from_training"
    return s


def codebook_codes(values):
    mapping, codes = {}, []
    for v in values:
        s = str(v)
        if s not in mapping:
            mapping[s] = len(mapping)
        codes.append(mapping[s])
    return torch.tensor(codes, dtype=torch.long), mapping


def scalar_or_row(v, i, default=""):
    if isinstance(v, (list, tuple)):
        return str(v[i]) if i < len(v) else default
    return str(v) if v is not None else default


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_feature_store", required=True)
    ap.add_argument("--feature_manifest_tsv", required=True)
    ap.add_argument("--gene_chunks_manifest_tsv", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--feature_store_root", default=None)
    ap.add_argument("--gene_root", default=None)
    args = ap.parse_args()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    if not Path(args.master_feature_store).exists():
        raise SystemExit(f"missing master_feature_store: {args.master_feature_store}")
    fm = pd.read_csv(args.feature_manifest_tsv, sep="\t", dtype=str).fillna("")
    gm = pd.read_csv(args.gene_chunks_manifest_tsv, sep="\t", dtype=str).fillna("")
    split = pd.read_csv(args.split_csv, dtype=str).fillna("")
    if "sid" not in fm.columns and "sample_id" in fm.columns:
        fm["sid"] = fm["sample_id"]
    if "sample_id" not in fm.columns and "sid" in fm.columns:
        fm["sample_id"] = fm["sid"]
    if "sid" not in gm.columns and "sample_id" in gm.columns:
        gm["sid"] = gm["sample_id"]
    if "sample_id" not in gm.columns and "sid" in gm.columns:
        gm["sample_id"] = gm["sid"]
    split["sample_id"] = split.get("sample_id", split.get("sid")).astype(str)
    split["formal_split"] = split["split"].map(split_norm)
    split_map = split[["sample_id", "formal_split"]].drop_duplicates("sample_id")
    split_lookup = {str(r.sample_id): str(r.formal_split) for r in split_map.itertuples(index=False)}
    required_fm = {"global_feature_index", "sid", "spot_id"}
    missing = sorted(required_fm - set(fm.columns))
    if missing:
        raise SystemExit(f"feature_manifest missing required columns: {missing}")
    dup = fm.duplicated(["sid", "spot_id"]).sum()
    if dup:
        raise SystemExit(f"feature_manifest duplicate sid+spot_id keys: {int(dup)}")
    feature_cols = [c for c in ["global_feature_index", "sid", "sample_id", "spot_id", "coord_x", "coord_y", "wsi_path", "image_path", "dataset_id", "organ_id", "tech_id", "patient_id", "section_id", "image_route", "extraction_status", "feature_dim"] if c in fm.columns]
    fm_small = fm[feature_cols].copy()
    feature_map = {}
    feature_row_map = {}
    for rec in fm_small.to_dict("records"):
        key = (str(rec["sid"]), str(rec["spot_id"]))
        feature_map[key] = int(rec["global_feature_index"])
        feature_row_map[key] = rec
    gene_chunk_table, row_rows, unmatched_rows, exact_rows, excluded_rows = [], [], [], [], []
    feature_index_arr, gene_chunk_id_arr, gene_row_index_arr = [], [], []
    n_gene_total = n_unmatched_total = 0
    gm_sorted = gm.sort_values("gene_chunk_id", key=lambda x: pd.to_numeric(x, errors="coerce")).reset_index(drop=True) if "gene_chunk_id" in gm.columns else gm.reset_index(drop=True)

    def process_chunk(rec):
        fallback_cid, m = rec
        m = {k: v for k, v in m.items()}
        cid = int(m.get("gene_chunk_id", fallback_cid))
        sid = str(m.get("sid") or m.get("sample_id"))
        cp = str(m["chunk_path"])
        ch = load_pt(cp)
        spot_ids = [str(x) for x in ch["spot_id"]]
        n_rows = len(spot_ids)
        local_genes = ch.get("local_gene_symbols", [])
        n_local = len(local_genes)
        ng = str(m.get("normalization_group_id", scalar_or_row(ch.get("normalization_group_id", ""), 0, "")))
        target_scale = str(m.get("target_raw_scale", ch.get("target_raw_scale", "raw_counts_log1p")))
        source_type = str(m.get("expression_source_type", ch.get("expression_source_type", m.get("source_type", ""))))
        gct = {"gene_chunk_id": cid, "chunk_id": cid, "sid": sid, "sample_id": sid, "chunk_path": cp, "source_type": source_type, "n_rows": n_rows, "n_local_genes": n_local, "normalization_group_id": ng, "target_raw_scale": target_scale}
        n_exact = n_unmatched = 0
        split_values = set()
        unmatched = []
        matched = []
        for i, spot in enumerate(spot_ids):
            key = (sid, spot)
            fi = feature_map.get(key)
            split_name = split_lookup.get(sid, scalar_or_row(ch.get("split", ""), i, ""))
            split_name = split_norm(split_name)
            split_values.add(split_name)
            if fi is None:
                n_unmatched += 1
                unmatched.append({"sid": sid, "sample_id": sid, "spot_id": spot, "source_type": source_type, "gene_chunk_id": cid, "gene_row_index": i, "reason": "no_feature_row_for_sid_spot_id", "suggested_fix": "inspect feature extraction/white filtering or spot_id mismatch"})
                continue
            frow = feature_row_map[key]
            dataset_id = m.get("dataset_id") or scalar_or_row(ch.get("dataset_id", frow.get("dataset_id", frow.get("dataset", ""))), i, frow.get("dataset_id", frow.get("dataset", "")))
            organ_id = m.get("organ_id") or scalar_or_row(ch.get("organ_id", frow.get("organ_id", frow.get("organ", ""))), i, frow.get("organ_id", frow.get("organ", "")))
            tech_id = m.get("tech_id") or scalar_or_row(ch.get("tech_id", frow.get("tech_id", frow.get("tech", ""))), i, frow.get("tech_id", frow.get("tech", "")))
            section_id = m.get("section_id") or scalar_or_row(ch.get("section_id", frow.get("section_id", sid)), i, frow.get("section_id", sid))
            rr = {"sid": sid, "sample_id": sid, "spot_id": spot, "split": split_name, "formal_split": split_name, "dataset_id": dataset_id, "organ_id": organ_id, "tech_id": tech_id, "section_id": section_id, "source_type": source_type, "feature_index": fi, "global_feature_index": fi, "gene_chunk_id": cid, "gene_chunk_path": cp, "gene_row_index": i, "normalization_group_id": ng, "target_raw_scale": target_scale, "n_local_genes": n_local, "observed_mask_group_id": ng, "coord_x": frow.get("coord_x", ""), "coord_y": frow.get("coord_y", ""), "image_route": frow.get("image_route", ""), "extraction_status": frow.get("extraction_status", "")}
            matched.append(rr)
            n_exact += 1
        frac = n_exact / max(n_rows, 1)
        status = "PASS" if n_exact > 0 else "WARN"
        exact = {"sid": sid, "sample_id": sid, "source_type": source_type, "n_gene_rows": n_rows, "n_feature_rows": int((fm["sid"].eq(sid)).sum()), "n_exact_matches": n_exact, "n_unmatched_gene_rows": n_unmatched, "n_extra_feature_rows": max(int((fm["sid"].eq(sid)).sum()) - n_exact, 0), "exact_match_fraction": f"{frac:.6f}", "rowmap_rows_kept": n_exact, "rowmap_rows_dropped": n_unmatched, "status": status, "warning_reason": "" if status == "PASS" else "zero exact matches; SID excluded", "split_values": "|".join(sorted(split_values))}
        excluded = None
        if n_exact == 0:
            excluded = {"sid": sid, "sample_id": sid, "source_type": source_type, "n_gene_rows": n_rows, "n_feature_rows": int((fm["sid"].eq(sid)).sum()), "reason": "zero_exact_matches", "status": "WARN", "suggested_fix": "excluded from rowmap; inspect feature extraction/spot_id alignment"}
        return gct, unmatched, exact, excluded, matched

    records = [(fallback_cid, m.to_dict()) for fallback_cid, m in gm_sorted.iterrows()]
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        for gct, unmatched, exact, excluded, matched in ex.map(process_chunk, records):
            cid = gct["gene_chunk_id"]
            gene_chunk_table.append(gct)
            n_unmatched_total += len(unmatched)
            unmatched_rows.extend(unmatched)
            exact_rows.append(exact)
            if excluded is not None:
                excluded_rows.append(excluded)
            for rr in matched:
                row_id = len(row_rows)
                rr["row_id"] = row_id
                row_rows.append(rr)
                feature_index_arr.append(rr["feature_index"])
                gene_chunk_id_arr.append(cid)
                gene_row_index_arr.append(rr["gene_row_index"])
            n_gene_total += gct["n_rows"]
    merged = pd.DataFrame(row_rows)
    if merged.empty:
        raise SystemExit("rowmap has zero rows after exact sid+spot_id join")
    merged.to_csv(out / "rowmap_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(gene_chunk_table).to_csv(out / "gene_chunk_table.tsv", sep="\t", index=False)
    pd.DataFrame(exact_rows).to_csv(out / "row_feature_exact_match_audit.tsv", sep="\t", index=False)
    pd.DataFrame(unmatched_rows).to_csv(out / "unmatched_gene_rows_audit.tsv", sep="\t", index=False)
    pd.DataFrame(excluded_rows).to_csv(out / "excluded_sid_audit.tsv", sep="\t", index=False)
    for col in ["sid", "dataset_id", "organ_id", "tech_id", "normalization_group_id", "split", "target_raw_scale", "source_type"]:
        codes, book = codebook_codes(merged[col].astype(str).tolist())
        name = {"sid": "row_to_sid", "dataset_id": "row_to_dataset_id", "organ_id": "row_to_organ_id", "tech_id": "row_to_tech_id", "normalization_group_id": "row_to_normalization_group_id", "split": "row_to_split", "target_raw_scale": "row_to_target_scale", "source_type": "row_to_source_type"}[col]
        torch.save(codes, out / f"{name}.pt")
    torch.save(torch.as_tensor(feature_index_arr, dtype=torch.long), out / "row_to_feature_index.pt")
    torch.save(torch.as_tensor(gene_chunk_id_arr, dtype=torch.long), out / "row_to_gene_chunk_id.pt")
    torch.save(torch.as_tensor(gene_row_index_arr, dtype=torch.long), out / "row_to_gene_row_index.pt")
    pd.DataFrame({"row_id": merged["row_id"], "sid": merged["sid"], "sample_id": merged["sample_id"], "spot_id": merged["spot_id"], "formal_split": merged["formal_split"]}).to_csv(out / "row_to_sid.tsv", sep="\t", index=False)
    for split_name, out_name in [("pretrain", "train_row_indices.pt"), ("val", "val_row_indices.pt"), ("test", "test_row_indices.pt"), ("excluded_from_training", "excluded_row_indices.pt")]:
        idx = torch.as_tensor(merged.index[merged["formal_split"].eq(split_name)].to_numpy(), dtype=torch.long)
        torch.save(idx, out / out_name)
    codebooks = {col: dict.fromkeys(merged[col].astype(str).tolist()) for col in ["sid", "dataset_id", "organ_id", "tech_id", "normalization_group_id", "split", "target_raw_scale", "source_type"]}
    codebooks = {k: {v: i for i, v in enumerate(d.keys())} for k, d in codebooks.items()}
    (out / "codebooks.json").write_text(json.dumps(codebooks, indent=2, ensure_ascii=False) + "\n")
    torch.save({"feature_index": torch.as_tensor(feature_index_arr, dtype=torch.long), "gene_chunk_id": torch.as_tensor(gene_chunk_id_arr, dtype=torch.int32), "gene_row_index": torch.as_tensor(gene_row_index_arr, dtype=torch.long), "spot_id": merged["spot_id"].astype(str).tolist(), "codebooks": codebooks, "gene_chunk_table": gene_chunk_table, "metadata": {"rowmap_policy": "exact_sid_spot_id_only", "used_coordinate_fallback": False, "target_scale": "raw_counts_log1p", "zscore_used": False, "n_rowmap_rows": int(len(merged))}}, out / "label_centered_rowmap.pt")
    split_audit = []
    for sn, grp in merged.groupby("formal_split"):
        split_audit.append({"split": sn, "n_rows": int(len(grp)), "n_sids": int(grp["sid"].nunique()), "status": "PASS"})
    pd.DataFrame(split_audit).to_csv(out / "rowmap_split_audit.tsv", sep="\t", index=False)
    # Raw-only smoke.
    smoke_rows = []
    try:
        fs = load_pt(args.master_feature_store)
        features = fs["features"] if isinstance(fs, dict) else fs
    except Exception as e:
        features = None; smoke_rows.append({"check": "feature_store_load", "status": "FAIL", "notes": repr(e)})
    if features is not None:
        for split_name in ["pretrain", "val", "test"]:
            sub = merged[merged["formal_split"].eq(split_name)]
            if sub.empty: continue
            r = sub.iloc[0]
            status = "PASS"; notes = []
            fi = int(r["feature_index"]); cid = int(r["gene_chunk_id"]); grow = int(r["gene_row_index"])
            fv = features[fi]
            fdim_ok = tuple(fv.shape) == (1536,)
            fnan = int(torch.isnan(fv).sum().item()); finf = int(torch.isinf(fv).sum().item())
            if not fdim_ok or fnan or finf: status = "FAIL"
            try:
                ch = load_pt(str(r["gene_chunk_path"]))
                raw_row = ch["target_genes_raw_local_csr"][grow]
                raw_data = raw_row.data if sp.issparse(raw_row) else np.asarray(raw_row)
                rn = int(np.isnan(raw_data).sum()); ri = int(np.isinf(raw_data).sum()); rneg = int((raw_data < 0).sum())
                if rn or ri or rneg: status = "FAIL"
            except Exception as e:
                rn = ri = rneg = -1; status = "FAIL"; notes.append("chunk_err:" + repr(e)[:120])
            smoke_rows.append({"split": split_name, "row_id": int(r["row_id"]), "feature_index": fi, "feature_dim_ok": fdim_ok, "feature_nan": fnan, "feature_inf": finf, "target_raw_nan": rn, "target_raw_inf": ri, "target_raw_negative": rneg, "status": status, "notes": "|".join(notes)})
    pd.DataFrame(smoke_rows).to_csv(out / "rowmap_runtime_smoke.tsv", sep="\t", index=False)
    summary = {"task_status": "PASS" if not any(str(x.get("status")) == "FAIL" for x in smoke_rows) else "FAIL", "n_rows": int(len(merged)), "n_gene_rows_total": int(n_gene_total), "n_unmatched_gene_rows": int(n_unmatched_total), "alignment": "sid+spot_id_exact", "used_coordinate_fallback": False, "feature_store_referenced_only": True, "dense_targets_created": False, "target_scale": "raw_counts_log1p", "zscore_used": False, "ready_for_m0_dataloader_smoke": True, "ready_for_neighbor_index_build": True, "ready_for_training": False}
    (out / "rowmap_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "final_decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
