#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def calculate(task):
    torch.set_num_threads(1)
    sid, organ, split, path, local_rows, expected_rows, old_ids = task
    obj = torch.load(path, map_location="cpu", weights_only=False)
    csr = obj["target_genes_raw_local_csr"].tocsr()
    rows = np.asarray(local_rows, dtype=np.int64)
    if rows.size != expected_rows or np.unique(rows).size != rows.size:
        raise RuntimeError(f"{sid}: active local-row count/uniqueness mismatch")
    if rows.min(initial=0) < 0 or rows.max(initial=-1) >= csr.shape[0]:
        raise RuntimeError(f"{sid}: active local row outside chunk")
    active = csr[rows]
    gids = obj["local_to_global_gene_idx"].long().numpy()
    observed = obj["observed_gene_mask_local"].bool().numpy()
    sums = np.asarray(active.sum(axis=0)).ravel().astype(np.float64)
    squared = active.astype(np.float64, copy=True)
    squared.data = np.square(squared.data)
    sumsqs = np.asarray(squared.sum(axis=0)).ravel()
    n = int(active.shape[0])
    variance = sumsqs / n - np.square(sums / n)
    variance[~observed] = -1.0
    valid = np.flatnonzero(variance >= 0)
    order = np.lexsort((gids[valid], -variance[valid]))
    selected = gids[valid[order[:384]]].astype(np.int64)
    if selected.size != 384 or np.unique(selected).size != 384:
        raise RuntimeError(f"{sid}: invalid active-row HVG384")
    old = set(map(int, old_ids))
    overlap = len(old.intersection(map(int, selected)))
    # Return NumPy rather than a torch tensor to avoid multiprocessing
    # shared-file-descriptor transfer failures after workers exit.
    return sid, selected, {
        "section_id": sid,
        "organ": organ,
        "split": split,
        "chunk_path": path,
        "full_chunk_rows": int(csr.shape[0]),
        "active_rows": n,
        "excluded_unmatched_rows": int(csr.shape[0] - n),
        "old_new_overlap": overlap,
        "changed_slots": 384 - overlap,
        "status": "PASS",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--rowmap-root", required=True, type=Path)
    ap.add_argument("--split-root", required=True, type=Path)
    ap.add_argument("--old-hvg", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest_rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    by_cid = {int(row["gene_chunk_id"]): row for row in manifest_rows}
    old_hvg = torch.load(args.old_hvg, map_location="cpu", weights_only=False)
    row_to_chunk = torch.load(args.rowmap_root / "row_to_gene_chunk_id.pt", map_location="cpu", weights_only=False).long()
    row_to_local = torch.load(args.rowmap_root / "row_to_gene_row_index.pt", map_location="cpu", weights_only=False).long()
    section_rows = list(csv.DictReader((args.split_root / "section_split_17organs_dedup.tsv").open(), delimiter="\t"))
    section_by_sid = {row["section_id"]: row for row in section_rows}

    jobs = []
    seen_sids = set()
    for split in ("train", "val", "test"):
        indices = torch.load(args.split_root / f"{split}_row_indices.pt", map_location="cpu", weights_only=False).long()
        chunks = row_to_chunk[indices].numpy()
        locals_ = row_to_local[indices].numpy()
        order = np.argsort(chunks, kind="stable")
        chunks, locals_ = chunks[order], locals_[order]
        cuts = np.flatnonzero(np.diff(chunks)) + 1
        for cids, local_rows in zip(np.split(chunks, cuts), np.split(locals_, cuts)):
            cid = int(cids[0])
            meta = by_cid[cid]
            sid = meta["sid"]
            section = section_by_sid[sid]
            if section["split"] != split:
                raise RuntimeError(f"{sid}: row-index split mismatch")
            if sid in seen_sids:
                raise RuntimeError(f"{sid}: section appears in multiple split tensors")
            seen_sids.add(sid)
            jobs.append((
                sid, section["organ"], split, meta["chunk_path"], local_rows,
                int(section["n_rows"]), torch.as_tensor(old_hvg[sid]).tolist(),
            ))

    if len(jobs) != 1205 or seen_sids != set(section_by_sid):
        raise RuntimeError("active section coverage is not exactly 1,205")
    torch.set_num_threads(1)
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(calculate, jobs)
    hvg = {sid: torch.from_numpy(ids) for sid, ids, _ in results}
    audits = [audit for _, _, audit in results]

    hvg_path = args.output_dir / "section_hvg384_active_rows.pt"
    tmp_path = hvg_path.with_suffix(".pt.tmp")
    torch.save(hvg, tmp_path)
    os.replace(tmp_path, hvg_path)
    fields = list(audits[0])
    with (args.output_dir / "section_hvg384_active_row_audit.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(audits, key=lambda row: row["section_id"]))
    by_split = Counter(row["split"] for row in audits)
    summary = {
        "status": "PASS_ACTIVE_ROW_HVG384_REBUILT",
        "sections": len(hvg),
        "sections_by_split": dict(by_split),
        "active_rows": sum(row["active_rows"] for row in audits),
        "excluded_unmatched_rows": sum(row["excluded_unmatched_rows"] for row in audits),
        "changed_sections": sum(row["changed_slots"] > 0 for row in audits),
        "changed_slots": sum(row["changed_slots"] for row in audits),
        "minimum_overlap": min(row["old_new_overlap"] for row in audits),
        "output": str(hvg_path),
        "output_sha256": sha256(hvg_path),
        "manifest_sha256": sha256(args.manifest),
        "train_indices_sha256": sha256(args.split_root / "train_row_indices.pt"),
        "row_to_gene_row_index_sha256": sha256(args.rowmap_root / "row_to_gene_row_index.pt"),
    }
    (args.output_dir / "active_row_hvg384_contract.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
