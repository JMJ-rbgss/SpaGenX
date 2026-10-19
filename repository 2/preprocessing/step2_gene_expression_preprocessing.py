#!/usr/bin/env python3
"""First-version-compatible gene preprocessing with raw_counts->log1p targets.

Keeps first-version source selection/local CSR/observed-mask contract, but removes
z-score and normalize_total. Only raw-count sources are accepted for target chunks.
"""
import argparse, json, os, re, gc
from pathlib import Path
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp


_CHUNK_GENE_TO_GLOBAL = None
_CHUNK_NG_CODE = None
_CHUNK_DIR = None
_CHUNK_SOURCE_AUDIT_BY_SID = None


def init_chunk_worker(gene_to_global, ng_code, chunk_dir, source_audit_by_sid):
    global _CHUNK_GENE_TO_GLOBAL, _CHUNK_NG_CODE, _CHUNK_DIR, _CHUNK_SOURCE_AUDIT_BY_SID
    _CHUNK_GENE_TO_GLOBAL = gene_to_global
    _CHUNK_NG_CODE = ng_code
    _CHUNK_DIR = Path(chunk_dir)
    _CHUNK_SOURCE_AUDIT_BY_SID = source_audit_by_sid


def gene_norm(g):
    s = str(g).strip()
    if re.match(r"^ENS[A-Z]*G\d+\.\d+$", s):
        return s.rsplit(".", 1)[0]
    return s


def s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def split_norm(x):
    v = s(x).strip().lower()
    if v in {"pretrain", "train", "training", "pretrain_v1_train"}: return "pretrain"
    if v in {"val", "valid", "validation", "pretrain_v1_val"}: return "val"
    if v in {"test", "heldout", "holdout"}: return "test"
    if v in {"exclude", "excluded", "excluded_from_training"}: return "excluded_from_training"
    return v


def classify_values_sample(vals):
    arr = np.asarray(vals, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "ambiguous", "no_finite_values"
    frac_neg = float(np.mean(arr < 0))
    frac_int = float(np.mean(np.isclose(arr, np.round(arr), atol=1e-6)))
    vmax = float(np.max(arr)); vmin = float(np.min(arr))
    if frac_neg > 0.001:
        return "scaled_likely", f"negative_fraction={frac_neg:.6f}"
    if frac_int > 0.98 and vmax > 1.0:
        return "raw_counts_likely", f"integer_fraction={frac_int:.6f};vmax={vmax:.3f}"
    if vmin >= 0 and vmax <= 30 and frac_int < 0.98:
        return "lognorm_likely", f"noninteger_fraction={1-frac_int:.6f};vmax={vmax:.3f}"
    return "ambiguous", f"integer_fraction={frac_int:.6f};vmin={vmin:.3f};vmax={vmax:.3f}"


def sample_sparse_values(X, n_obs=128, max_values=20000):
    xs = X[:min(n_obs, X.shape[0])]
    if sp.issparse(xs):
        vals = xs.data[:max_values]
    else:
        vals = np.asarray(xs).ravel()[:max_values]
    return vals


def detect_h5ad_raw_source(path):
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    try:
        candidates = []
        if a.raw is not None:
            vals = sample_sparse_values(a.raw.X)
            cls, reason = classify_values_sample(vals)
            candidates.append(("raw.X", cls, reason, [str(g) for g in a.raw.var_names]))
        if "counts" in a.layers:
            vals = sample_sparse_values(a.layers["counts"])
            cls, reason = classify_values_sample(vals)
            candidates.append(('layers["counts"]', cls, reason, [str(g) for g in a.var_names]))
        vals = sample_sparse_values(a.X)
        cls, reason = classify_values_sample(vals)
        candidates.append(("a.X", cls, reason, [str(g) for g in a.var_names]))
        for source, cls, reason, genes in candidates:
            if cls == "raw_counts_likely":
                return source, cls, reason, genes
        return candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3]
    finally:
        try: a.file.close()
        except Exception: pass


def csv_gene_and_class(path):
    head = pd.read_csv(path, nrows=0)
    if len(head.columns) < 2:
        raise ValueError(f"CSV expression has <2 columns: {path}")
    sample = pd.read_csv(path, nrows=64)
    vals = sample.iloc[:, 1:min(sample.shape[1], 513)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64).ravel()
    cls, reason = classify_values_sample(vals)
    if Path(path).name.lower().endswith("_count.csv") and cls == "ambiguous":
        cls, reason = "raw_counts_likely", "count_csv_path_policy_sample_low_information"
    return [str(c) for c in head.columns[1:]], cls, reason


def row_dict(row):
    return {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}


def norm_group(row):
    parts = [s(row.get("dataset")) or s(row.get("dataset_id")), s(row.get("organ")) or s(row.get("organ_id")), s(row.get("tech")) or s(row.get("tech_id"))]
    return "|".join(p if p else "UNKNOWN" for p in parts)


def source_paths(row):
    h5ad = s(row.get("h5ad_path"))
    csv = s(row.get("gene_exp_path")) or s(row.get("expression_path")) or s(row.get("csv_path"))
    return h5ad, csv


def sample_source_audit(row):
    sid = s(row.get("sample_id")) or s(row.get("sid"))
    h5ad, csv = source_paths(row)
    if h5ad:
        source, cls, reason, genes = detect_h5ad_raw_source(h5ad)
        return {"sample_id": sid, "sid": sid, "expression_source_type": "h5ad", "expression_path": h5ad, "selected_source": source, "source_classification": cls, "classification_reason": reason, "usable_for_gene_preprocessing": cls == "raw_counts_likely", "blocking_reason": "" if cls == "raw_counts_likely" else "non_raw_counts_source_blocked", "genes": genes}
    if csv:
        genes, cls, reason = csv_gene_and_class(csv)
        return {"sample_id": sid, "sid": sid, "expression_source_type": "csv", "expression_path": csv, "selected_source": "csv_matrix", "source_classification": cls, "classification_reason": reason, "usable_for_gene_preprocessing": cls == "raw_counts_likely", "blocking_reason": "" if cls == "raw_counts_likely" else "non_raw_counts_source_blocked", "genes": genes}
    return {"sample_id": sid, "sid": sid, "expression_source_type": "", "expression_path": "", "selected_source": "", "source_classification": "missing", "classification_reason": "no_expression_source", "usable_for_gene_preprocessing": False, "blocking_reason": "missing_expression_source", "genes": []}


def vocab_worker(item):
    order, row = item
    audit = sample_source_audit(row)
    return order, audit


def load_csv_expr(path, gene_to_global, chunksize=2048):
    """Load CSV expression as sparse CSR without materializing the full dense matrix.

    First column is spot id. Kept gene columns are streamed in row chunks,
    transformed raw_counts->log1p, converted to CSR, then vstacked. This avoids
    VisiumHD-scale dense allocations.
    """
    header = pd.read_csv(path, nrows=0)
    first = header.columns[0]
    genes = [str(c) for c in header.columns[1:]]
    usecols = [first]
    local_genes, local_global, seen = [], [], set()
    for g in genes:
        gn = gene_norm(g)
        gid = gene_to_global.get(gn)
        if gid is not None and gid not in seen:
            seen.add(gid)
            usecols.append(g)
            local_genes.append(gn)
            local_global.append(gid)
    if not local_genes:
        spot_ids = pd.read_csv(path, usecols=[first])[first].astype(str).tolist()
        mat = sp.csr_matrix((len(spot_ids), 0), dtype=np.float32)
        return spot_ids, local_genes, np.asarray(local_global, dtype=np.int64), mat

    spot_ids = []
    blocks = []
    for df in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        spot_ids.extend(df[first].astype(str).tolist())
        vals = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=True)
        vals[~np.isfinite(vals)] = 0
        vals[vals < 0] = 0
        np.log1p(vals, out=vals)
        block = sp.csr_matrix(vals)
        block.eliminate_zeros()
        blocks.append(block)
        del vals, block, df
    if blocks:
        mat = sp.vstack(blocks, format="csr", dtype=np.float32)
    else:
        mat = sp.csr_matrix((0, len(local_genes)), dtype=np.float32)
    return spot_ids, local_genes, np.asarray(local_global, dtype=np.int64), mat


def load_h5ad_expr(path, selected_source, gene_to_global):
    import anndata as ad
    a = ad.read_h5ad(path)
    try:
        spot_ids = [str(x) for x in a.obs_names]
        if selected_source == "raw.X" and a.raw is not None:
            genes = [str(g) for g in a.raw.var_names]; X = a.raw.X
        elif selected_source == 'layers["counts"]' and "counts" in a.layers:
            genes = [str(g) for g in a.var_names]; X = a.layers["counts"]
        else:
            genes = [str(g) for g in a.var_names]; X = a.X
        keep, local_genes, local_global, seen = [], [], [], set()
        for j, g in enumerate(genes):
            gn = gene_norm(g)
            gid = gene_to_global.get(gn)
            if gid is not None and gid not in seen:
                seen.add(gid); keep.append(j); local_genes.append(gn); local_global.append(gid)
        if not keep:
            mat = sp.csr_matrix((a.n_obs, 0), dtype=np.float32)
        else:
            Xs = X[:, keep]
            if sp.issparse(Xs):
                mat = Xs.tocsr().astype(np.float32)
                mat.data[~np.isfinite(mat.data)] = 0; mat.data[mat.data < 0] = 0
                np.log1p(mat.data, out=mat.data)
                mat.eliminate_zeros()
            else:
                vals = np.asarray(Xs, dtype=np.float32)
                vals[~np.isfinite(vals)] = 0; vals[vals < 0] = 0
                np.log1p(vals, out=vals)
                mat = sp.csr_matrix(vals)
        return spot_ids, local_genes, np.asarray(local_global, dtype=np.int64), mat
    finally:
        del a


def chunk_worker(item, gene_to_global=None, ng_code=None, chunk_dir=None, source_audit_by_sid=None):
    if gene_to_global is None:
        gene_to_global = _CHUNK_GENE_TO_GLOBAL
    if ng_code is None:
        ng_code = _CHUNK_NG_CODE
    if chunk_dir is None:
        chunk_dir = _CHUNK_DIR
    else:
        chunk_dir = Path(chunk_dir)
    if source_audit_by_sid is None:
        source_audit_by_sid = _CHUNK_SOURCE_AUDIT_BY_SID
    if gene_to_global is None or ng_code is None or chunk_dir is None or source_audit_by_sid is None:
        raise RuntimeError("chunk worker globals are not initialized")

    cid, row = item
    sid = s(row.get("sample_id")) or s(row.get("sid"))
    audit = source_audit_by_sid[sid]
    ng = norm_group(row)
    chunk_path = chunk_dir / f"{sid}.pt"

    def build_manifest_from_obj(obj, status="OK"):
        csr = obj["target_genes_raw_local_csr"]
        n_rows, n_genes = int(csr.shape[0]), int(csr.shape[1])
        nnz = int(csr.nnz)
        manifest = {"gene_chunk_id": cid, "chunk_id": cid, "chunk_path": str(chunk_path), "sid": sid, "sample_id": sid, "dataset_id": s(row.get("dataset_id")) or s(row.get("dataset")), "organ_id": s(row.get("organ_id")) or s(row.get("organ")), "tech_id": s(row.get("tech_id")) or s(row.get("tech")), "section_id": s(row.get("section_id")) or sid, "normalization_group_id": ng, "normalization_group_code": int(ng_code[ng]), "split": split_norm(row.get("formal_split", row.get("split", ""))), "n_rows": n_rows, "n_local_genes": n_genes, "nnz": nnz, "nnz_raw": nnz, "target_raw_scale": "raw_counts_log1p", "target_transform": "raw_to_log1p", "storage_format": "csr_raw_log1p_only", "zscore_used": False, "status": status}
        data = csr.data if nnz else np.asarray([], dtype=np.float32)
        sample_audit = {"sid": sid, "sample_id": sid, "expression_source_type": audit["expression_source_type"], "selected_source": audit["selected_source"], "source_classification": audit["source_classification"], "normalization_action": "raw_counts_log1p", "target_raw_scale": "raw_counts_log1p", "n_spots": n_rows, "n_genes_local": n_genes, "nan_count": int(np.isnan(data).sum()) if nnz else 0, "inf_count": int(np.isinf(data).sum()) if nnz else 0, "negative_count_after_processing": int((data < 0).sum()) if nnz else 0, "status": status, "normalization_group_id": ng}
        return manifest, sample_audit

    if chunk_path.exists() and chunk_path.stat().st_size > 0:
        try:
            obj = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if "target_genes_raw_local_csr" in obj and "local_to_global_gene_idx" in obj:
                manifest, sample_audit = build_manifest_from_obj(obj, status="SKIPPED_EXISTING")
                return cid, manifest, sample_audit
        except Exception:
            pass

    if audit["expression_source_type"] == "h5ad":
        spot_ids, local_genes, local_global, csr = load_h5ad_expr(audit["expression_path"], audit["selected_source"], gene_to_global)
    else:
        spot_ids, local_genes, local_global, csr = load_csv_expr(audit["expression_path"], gene_to_global)
    obj = {
        "target_genes_raw_local_csr": csr,
        "local_gene_symbols": local_genes,
        "local_to_global_gene_idx": torch.as_tensor(local_global, dtype=torch.long),
        "observed_gene_mask_local": torch.ones(len(local_genes), dtype=torch.bool),
        "sid": [sid] * csr.shape[0],
        "sample_id": [sid] * csr.shape[0],
        "spot_id": [str(x) for x in spot_ids],
        "dataset_id": [s(row.get("dataset_id")) or s(row.get("dataset"))] * csr.shape[0],
        "organ_id": [s(row.get("organ_id")) or s(row.get("organ"))] * csr.shape[0],
        "tech_id": [s(row.get("tech_id")) or s(row.get("tech"))] * csr.shape[0],
        "section_id": [s(row.get("section_id")) or sid] * csr.shape[0],
        "normalization_group_id": [ng] * csr.shape[0],
        "normalization_group_code": int(ng_code[ng]),
        "target_raw_scale": "raw_counts_log1p",
        "target_transform": "raw_to_log1p",
        "zscore_used": False,
        "targets_are_zscored": False,
        "expression_source_type": audit["expression_source_type"],
        "selected_expression_source": audit["selected_source"],
        "split": [split_norm(row.get("formal_split", row.get("split", "")))] * csr.shape[0],
    }
    torch.save(obj, chunk_path)
    manifest, sample_audit = build_manifest_from_obj(obj, status="OK")
    return cid, manifest, sample_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory_tsv", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--limit_vocab_to_limit_samples", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    workers = max(1, int(args.num_workers))
    inv = pd.read_csv(args.inventory_tsv, sep="\t").fillna("")
    split = pd.read_csv(args.split_csv).fillna("")
    if "sid" not in inv.columns and "sample_id" in inv.columns:
        inv["sid"] = inv["sample_id"].astype(str)
    if "sample_id" not in inv.columns and "sid" in inv.columns:
        inv["sample_id"] = inv["sid"].astype(str)
    split["sample_id"] = split.get("sample_id", split.get("sid")).astype(str)
    inv["sample_id"] = inv["sample_id"].astype(str)
    split["formal_split"] = split["split"].map(split_norm)
    if "status" in inv.columns:
        inv = inv[inv["status"].astype(str).eq("PASS")].copy()
    inv = inv.merge(split[["sample_id", "formal_split"]].drop_duplicates("sample_id"), on="sample_id", how="left")
    if inv["formal_split"].isna().any():
        raise SystemExit(f"split_csv missing PASS samples: {int(inv['formal_split'].isna().sum())}")
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    chunk_dir = out / "gene_label_store" / "chunks"; chunk_dir.mkdir(parents=True, exist_ok=True)
    inv_run = inv.head(args.limit_samples).copy() if args.limit_samples and args.limit_samples > 0 else inv.copy()
    pre_source = inv_run if args.limit_vocab_to_limit_samples else inv
    cached_source_tsv = out / "sample_gene_source_selection.tsv"
    cached_vocab_tsv = out / "global_gene_vocab.tsv"
    cached_ng_masks = out / "observed_gene_mask_by_normalization_group.pt"
    cached_sid_masks = out / "observed_gene_mask_by_sid.pt"
    use_cached_source_vocab = (
        cached_source_tsv.exists()
        and cached_vocab_tsv.exists()
        and cached_ng_masks.exists()
        and cached_sid_masks.exists()
        and not args.limit_vocab_to_limit_samples
    )

    def truthy(v):
        return str(v).strip().lower() in {"true", "1", "yes", "y"}

    if use_cached_source_vocab:
        print("[resume] using cached source/vocab/observed-mask files", flush=True)
        visible_source_rows = pd.read_csv(cached_source_tsv, sep="\t").fillna("").to_dict("records")
        source_audit_by_sid = {str(r.get("sample_id") or r.get("sid")): r for r in visible_source_rows}
        blocked = [r for r in visible_source_rows if not truthy(r.get("usable_for_gene_preprocessing"))]
        vocab_df = pd.read_csv(cached_vocab_tsv, sep="\t").fillna("")
        gene_to_global = {str(r["gene_symbol"]): int(r["gene_index"]) for _, r in vocab_df.iterrows()}
        usable_mask = []
        for _, r in inv_run.iterrows():
            sid = str(r["sample_id"])
            audit = source_audit_by_sid.get(sid)
            usable_mask.append(bool(audit and truthy(audit.get("usable_for_gene_preprocessing"))))
        usable_run = inv_run[usable_mask].copy()
        ng_values = [norm_group(row_dict(r)) for _, r in usable_run.iterrows()]
        ng_code = {ng: i for i, ng in enumerate(dict.fromkeys(ng_values))}
    else:
        pre_df = pre_source[pre_source["formal_split"].eq("pretrain")].copy()
        if pre_df.empty:
            raise SystemExit("no pretrain/train samples in split")
        print(f"[source] auditing samples={len(inv)} workers={workers}", flush=True)
        source_rows = []
        items = [(i, row_dict(r)) for i, (_, r) in enumerate(inv.iterrows())]
        if workers == 1:
            for item in items:
                _, audit = vocab_worker(item); source_rows.append(audit)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(vocab_worker, item) for item in items]
                for fut in as_completed(futs):
                    _, audit = fut.result(); source_rows.append(audit)
        source_rows.sort(key=lambda r: r["sample_id"])
        source_audit_by_sid = {r["sample_id"]: r for r in source_rows}
        visible_source_rows = [{k: v for k, v in r.items() if k != "genes"} for r in source_rows]
        pd.DataFrame(visible_source_rows).to_csv(out / "sample_gene_source_selection.tsv", sep="\t", index=False)
        blocked = [r for r in visible_source_rows if not bool(r["usable_for_gene_preprocessing"])]
        pd.DataFrame(blocked).to_csv(out / "blocked_gene_samples.tsv", sep="\t", index=False)
        pre_usable = [r for _, r in pre_df.iterrows() if bool(source_audit_by_sid[str(r["sample_id"])] ["usable_for_gene_preprocessing"])]
        if not pre_usable:
            raise SystemExit("no raw-count usable pretrain samples for vocab")
        vocab = OrderedDict(); vocab_rows = []; sample_gene_sets = {}
        for r in pre_usable:
            sid = str(r["sample_id"]); genes = source_audit_by_sid[sid]["genes"]
            sample_gene_sets[sid] = set()
            seen = set()
            for g in genes:
                gn = gene_norm(g)
                if gn in seen: continue
                seen.add(gn); sample_gene_sets[sid].add(gn)
                if gn not in vocab:
                    vocab[gn] = len(vocab)
                    vocab_rows.append({"gene_index": vocab[gn], "global_gene_id": vocab[gn], "gene_symbol": gn, "first_seen_sample_id": sid, "gene_symbol_status": "normalized"})
        gene_to_global = dict(vocab)
        pd.DataFrame(vocab_rows).to_csv(out / "global_gene_vocab.tsv", sep="\t", index=False)
        (out / "global_gene_vocab_manifest.json").write_text(json.dumps({"vocab_policy": "union_all_measured_genes_from_raw_count_pretrain_sources", "n_global_genes": len(gene_to_global), "used_expression_values_for_selection": False, "used_val_expression_stats_for_selection": False, "target_transform": "raw_to_log1p"}, indent=2) + "\n")
        usable_run = inv_run[[bool(source_audit_by_sid[str(r["sample_id"])] ["usable_for_gene_preprocessing"]) for _, r in inv_run.iterrows()]].copy()
        ng_values = [norm_group(row_dict(r)) for _, r in usable_run.iterrows()]
        ng_code = {ng: i for i, ng in enumerate(dict.fromkeys(ng_values))}
        # observed masks by normalization group and sid
        G = len(gene_to_global)
        ng_masks = {ng: torch.zeros(G, dtype=torch.bool) for ng in ng_code}
        sid_masks = {}
        for _, r in usable_run.iterrows():
            rd = row_dict(r); sid = str(rd["sample_id"]); ng = norm_group(rd)
            genes = set(gene_norm(g) for g in source_audit_by_sid[sid]["genes"])
            mask = torch.zeros(G, dtype=torch.bool)
            for g in genes:
                gid = gene_to_global.get(g)
                if gid is not None:
                    mask[gid] = True; ng_masks[ng][gid] = True
            sid_masks[sid] = mask
        torch.save(ng_masks, out / "observed_gene_mask_by_normalization_group.pt")
        torch.save(sid_masks, out / "observed_gene_mask_by_sid.pt")
    chunk_items = [(cid, row_dict(r)) for cid, (_, r) in enumerate(usable_run.iterrows())]
    manifest_rows, sample_audit = [], []
    print(f"[chunks] writing samples={len(chunk_items)} workers={workers}", flush=True)
    if workers == 1:
        init_chunk_worker(gene_to_global, ng_code, chunk_dir, source_audit_by_sid)
        for item in chunk_items:
            cid, manifest, audit = chunk_worker(item)
            manifest_rows.append(manifest); sample_audit.append(audit)
            print(f"[chunk] {cid+1}/{len(chunk_items)} {manifest['sample_id']} rows={manifest['n_rows']} genes={manifest['n_local_genes']} nnz={manifest['nnz']} status={manifest['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_chunk_worker, initargs=(gene_to_global, ng_code, str(chunk_dir), source_audit_by_sid)) as ex:
            futs = [ex.submit(chunk_worker, item) for item in chunk_items]
            for done, fut in enumerate(as_completed(futs), start=1):
                cid, manifest, audit = fut.result()
                manifest_rows.append(manifest); sample_audit.append(audit)
                print(f"[chunk] {done}/{len(futs)} cid={cid} {manifest['sample_id']} rows={manifest['n_rows']} genes={manifest['n_local_genes']} nnz={manifest['nnz']} status={manifest['status']}", flush=True)
    manifest_rows.sort(key=lambda x: x["gene_chunk_id"])
    sample_audit.sort(key=lambda x: x["sample_id"])
    pd.DataFrame(manifest_rows).to_csv(out / "gene_label_chunks_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(sample_audit).to_csv(out / "gene_chunk_sample_audit.tsv", sep="\t", index=False)
    torch.save({"stats_scope": "none_for_target_transform", "target_scale": "raw_counts_log1p", "zscore_used": False}, out / "pretrain_gene_stats.pt")
    policy = {"target_source": "raw_count_expression_matrix", "target_transform": "raw_to_log1p", "target_scale": "raw_counts_log1p", "zscore_used": False, "formal_train_norm_stats_required": False, "implicit_zero_policy": "sample_local_panel_missing_csr_entry_is_measured_zero", "global_unobserved_gene_policy": "excluded_from_denominator", "global_vocab_source": "pretrain_raw_count_sample_gene_union", "n_global_genes": len(gene_to_global), "num_workers": workers}
    (out / "target_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    final = {"task_status": "WARN" if blocked else "PASS", "n_chunks": len(manifest_rows), "n_global_genes": len(gene_to_global), "n_blocked_gene_samples": len(blocked), "target_scale": "raw_counts_log1p", "zscore_used": False, "ready_for_rowmap": len(manifest_rows) > 0, "ready_for_training": False}
    (out / "final_decision.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
