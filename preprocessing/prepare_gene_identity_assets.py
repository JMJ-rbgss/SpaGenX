#!/usr/bin/env python3
"""Build add-only canonical gene-identity assets for the 17-organ SpaGenX corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


BASE = Path((__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + ''))
DEFAULT_SPLIT = BASE / "section_split.tsv"
DEFAULT_MANIFEST = BASE / "gene_label_chunks_manifest.tsv"
DEFAULT_VOCAB = BASE / "global_gene_vocab.tsv"
DEFAULT_ROWMAP = BASE / "rowmap"
DEFAULT_H5AD = BASE / "h5ad"
DEFAULT_PROGRAM = BASE / "program_source"
DEFAULT_SPAGENX_OVERLAY = BASE / "neighbors.pt"

EXPECTED = {
    "active_sections": 1205,
    "active_rows": 2798414,
    "section_splits": {"train": 960, "val": 63, "test": 182},
    "row_splits": {"train": 2339452, "val": 114580, "test": 344382},
    "active_affected_sections": 158,
    "vocab_rows": 115055,
}

ENSG_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$")
_ENSG_MAP: Dict[str, str] = {}
_CONFLICTS: Set[str] = set()
_SYMBOL_TO_GID: Dict[str, int] = {}
_GID_TO_SYMBOL: Dict[int, str] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_tsv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def init_worker(ensg_map, conflicts, symbol_to_gid, gid_to_symbol) -> None:
    global _ENSG_MAP, _CONFLICTS, _SYMBOL_TO_GID, _GID_TO_SYMBOL
    _ENSG_MAP = dict(ensg_map)
    _CONFLICTS = set(conflicts)
    _SYMBOL_TO_GID = {str(k): int(v) for k, v in symbol_to_gid.items()}
    _GID_TO_SYMBOL = {int(k): str(v) for k, v in gid_to_symbol.items()}


def collect_source_mapping(active_sids: Set[str], h5ad_root: Path, out: Path):
    mapping: Dict[str, str] = {}
    provenance: Dict[str, Set[str]] = defaultdict(set)
    conflicts: Set[str] = set()
    source_rows = []
    for sid in sorted(active_sids):
        path = h5ad_root / f"{sid}.h5ad"
        if not path.exists():
            continue
        a = ad.read_h5ad(path, backed="r")
        try:
            if "SYMBOL" not in a.var.columns:
                continue
            ids = list(map(str, a.var["gene_ids"])) if "gene_ids" in a.var.columns else list(map(str, a.var.index))
            symbols = list(map(str, a.var["SYMBOL"]))
            used = 0
            for ensembl, symbol in zip(ids, symbols):
                stable = ensembl.split(".")[0]
                if not re.fullmatch(r"ENSG\d+", stable) or not symbol or symbol == "nan":
                    continue
                used += 1
                if stable in mapping and mapping[stable] != symbol:
                    conflicts.add(stable)
                else:
                    mapping[stable] = symbol
                provenance[stable].add(str(path))
            source_rows.append({
                "section_id": sid,
                "path": str(path),
                "mapping_rows": used,
                "sha256": sha256_file(path),
            })
        finally:
            if getattr(a, "file", None) is not None:
                a.file.close()
    for stable in conflicts:
        mapping.pop(stable, None)
    write_tsv(out / "mapping_source_files.tsv", source_rows)
    return mapping, conflicts, provenance, source_rows


def scan_chunk(task):
    sid, organ, split, path, active = task
    obj = torch.load(path, map_location="cpu", weights_only=False)
    symbols = list(map(str, obj["local_gene_symbols"]))
    csr = obj["target_genes_raw_local_csr"]
    n_ensg = sum(bool(ENSG_RE.fullmatch(x)) for x in symbols)
    return {
        "section_id": sid,
        "organ": organ,
        "split": split,
        "active": bool(active),
        "chunk_path": path,
        "n_rows_full_chunk": int(csr.shape[0]),
        "n_local_genes": int(csr.shape[1]),
        "n_ensg_columns": int(n_ensg),
        "source_chunk_bytes": int(Path(path).stat().st_size),
    }


def resolve_column(symbol: str, old_gid: int):
    if ENSG_RE.fullmatch(symbol):
        stable = symbol.split(".")[0]
        if stable in _CONFLICTS:
            return None, "DROP_CONFLICT"
        target_symbol = _ENSG_MAP.get(stable)
        if target_symbol is None:
            return None, "DROP_UNRESOLVED"
        target_gid = _SYMBOL_TO_GID.get(target_symbol)
        if target_gid is None:
            return None, "DROP_SYMBOL_NOT_IN_VOCAB"
        return int(target_gid), "MAP_ENSG_TO_CANONICAL"
    expected_gid = _SYMBOL_TO_GID.get(symbol)
    if expected_gid is None:
        return None, "DROP_LOCAL_SYMBOL_NOT_IN_VOCAB"
    if int(old_gid) != int(expected_gid):
        raise RuntimeError(f"non-ENSG symbol/global ID mismatch: {symbol}: {old_gid} != {expected_gid}")
    return int(expected_gid), "KEEP_CANONICAL"


def collapse_chunk_matrix(csr, symbols, gids, observed):
    """Canonicalize columns while preserving singleton values bit-for-bit."""
    groups: Dict[int, List[int]] = defaultdict(list)
    counts = defaultdict(int)
    dropped_raw_total = 0.0
    retained_raw_total = 0.0
    raw_csc = csr.tocsc().astype(np.float64, copy=True)
    if raw_csc.data.size:
        raw_csc.data = np.expm1(raw_csc.data)
    raw_column_totals = np.asarray(raw_csc.sum(axis=0)).ravel()
    for col, (symbol, gid, is_observed) in enumerate(zip(symbols, gids, observed)):
        if not bool(is_observed):
            counts["drop_unobserved"] += 1
            continue
        target_gid, action = resolve_column(str(symbol), int(gid))
        counts[action] += 1
        if target_gid is None:
            dropped_raw_total += float(raw_column_totals[col])
            continue
        groups[target_gid].append(col)
        retained_raw_total += float(raw_column_totals[col])

    ordered_gids = sorted(groups)
    singleton_gids = [gid for gid in ordered_gids if len(groups[gid]) == 1]
    duplicate_gids = [gid for gid in ordered_gids if len(groups[gid]) > 1]
    blocks = []
    block_gids = []
    if singleton_gids:
        singleton_cols = [groups[gid][0] for gid in singleton_gids]
        blocks.append(csr[:, singleton_cols].tocsr(copy=True))
        block_gids.extend(singleton_gids)
    for gid in duplicate_gids:
        sub = csr[:, groups[gid]].astype(np.float64, copy=True)
        if sub.data.size:
            sub.data = np.expm1(sub.data)
        raw_sum = np.asarray(sub.sum(axis=1)).ravel()
        nz = np.flatnonzero(raw_sum > 0)
        data = np.log1p(raw_sum[nz]).astype(np.float32)
        col = sp.csr_matrix((data, (nz, np.zeros(len(nz), dtype=np.int64))), shape=(csr.shape[0], 1), dtype=np.float32)
        blocks.append(col)
        block_gids.append(gid)
    if not blocks:
        raise RuntimeError("canonicalization removed every observed gene")
    merged = sp.hstack(blocks, format="csr", dtype=np.float32)
    order = np.argsort(np.asarray(block_gids, dtype=np.int64), kind="stable")
    merged = merged[:, order].tocsr()
    final_gids = np.asarray(block_gids, dtype=np.int64)[order]
    merged.eliminate_zeros()
    output_raw_total = float(np.expm1(merged.data.astype(np.float64)).sum()) if merged.data.size else 0.0
    raw_abs_error = abs(output_raw_total - retained_raw_total)
    raw_rel_error = raw_abs_error / max(1.0, abs(retained_raw_total))
    stats = {
        **counts,
        "n_output_genes": int(len(final_gids)),
        "n_collapsed_target_ids": int(len(duplicate_gids)),
        "max_collapse_size": int(max((len(v) for v in groups.values()), default=0)),
        "dropped_raw_count_total": dropped_raw_total,
        "retained_raw_count_total": retained_raw_total,
        "output_raw_count_total": output_raw_total,
        "raw_total_abs_error": raw_abs_error,
        "raw_total_rel_error": raw_rel_error,
    }
    return merged, final_gids, stats


def repair_chunk(task):
    sid, organ, split, source_path, output_path = task
    source_path = Path(source_path)
    output_path = Path(output_path)
    obj = torch.load(source_path, map_location="cpu", weights_only=False)
    csr = obj["target_genes_raw_local_csr"].tocsr()
    symbols = list(map(str, obj["local_gene_symbols"]))
    gids = obj["local_to_global_gene_idx"].long().numpy()
    observed = obj["observed_gene_mask_local"].bool().numpy()
    if len(symbols) != csr.shape[1] or len(gids) != csr.shape[1] or len(observed) != csr.shape[1]:
        raise RuntimeError(f"{sid}: chunk column metadata mismatch")
    merged, final_gids, stats = collapse_chunk_matrix(csr, symbols, gids, observed)
    if len(final_gids) != len(set(map(int, final_gids))):
        raise RuntimeError(f"{sid}: duplicate canonical global IDs remain")
    final_symbols = [_GID_TO_SYMBOL[int(gid)] for gid in final_gids]
    if any(ENSG_RE.fullmatch(symbol) for symbol in final_symbols):
        raise RuntimeError(f"{sid}: ENSG pseudo-symbol remains after canonicalization")
    fixed = dict(obj)
    fixed["target_genes_raw_local_csr"] = merged
    fixed["local_gene_symbols"] = final_symbols
    fixed["local_to_global_gene_idx"] = torch.as_tensor(final_gids, dtype=torch.long)
    fixed["observed_gene_mask_local"] = torch.ones(len(final_gids), dtype=torch.bool)
    fixed["selected_expression_source"] = str(obj.get("selected_expression_source", "")) + ":canonical_gene_identity_add_only"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".pt.tmp")
    torch.save(fixed, tmp)
    os.replace(tmp, output_path)
    reread = torch.load(output_path, map_location="cpu", weights_only=False)
    if list(map(str, reread["spot_id"])) != list(map(str, obj["spot_id"])):
        raise RuntimeError(f"{sid}: spot order changed")
    if int(reread["target_genes_raw_local_csr"].shape[0]) != int(csr.shape[0]):
        raise RuntimeError(f"{sid}: row count changed")
    if stats["raw_total_rel_error"] > 2e-7:
        raise RuntimeError(f"{sid}: retained raw-count total mismatch {stats['raw_total_rel_error']}")
    return {
        "section_id": sid,
        "organ": organ,
        "split": split,
        "source_chunk": str(source_path),
        "source_sha256": sha256_file(source_path),
        "corrected_chunk": str(output_path),
        "corrected_sha256": sha256_file(output_path),
        "n_rows_before": int(csr.shape[0]),
        "n_rows_after": int(merged.shape[0]),
        "n_genes_before": int(csr.shape[1]),
        "n_genes_after": int(merged.shape[1]),
        "nnz_before": int(csr.nnz),
        "nnz_after": int(merged.nnz),
        **stats,
        "status": "REPAIRED_ADD_ONLY",
    }


def hvg384_task(task):
    sid, organ, split, path = task
    obj = torch.load(path, map_location="cpu", weights_only=False)
    csr = obj["target_genes_raw_local_csr"].tocsr()
    gids = obj["local_to_global_gene_idx"].long().numpy()
    observed = obj["observed_gene_mask_local"].bool().numpy()
    n = int(csr.shape[0])
    s1 = np.asarray(csr.sum(axis=0)).ravel().astype(np.float64)
    sq = csr.astype(np.float64, copy=True)
    sq.data = np.square(sq.data)
    s2 = np.asarray(sq.sum(axis=0)).ravel()
    var = s2 / max(n, 1) - np.square(s1 / max(n, 1))
    var[~observed] = -1.0
    valid = np.flatnonzero(var >= 0)
    order = np.lexsort((gids[valid], -var[valid]))
    chosen = valid[order[: min(384, len(order))]]
    ids = gids[chosen].astype(np.int64)
    return sid, ids, {
        "section_id": sid,
        "organ": organ,
        "split": split,
        "chunk_path": path,
        "n_rows_full_chunk": n,
        "n_observed_genes": int(observed.sum()),
        "n_hvg384": int(len(ids)),
        "top10_gene_ids": ",".join(map(str, ids[:10])),
    }


def rebuild_program_overlay(program_root: Path, out: Path, vocab: pd.DataFrame, ensg_map, conflicts):
    edges = pd.read_csv(program_root / "runtime_gene_program_edges.tsv", sep="\t")
    programs = pd.read_csv(program_root / "runtime_programs_256.tsv", sep="\t")
    symbol_to_gid = dict(zip(vocab.gene_symbol.astype(str), vocab.global_gene_id.astype(int)))
    gid_to_symbol = dict(zip(vocab.global_gene_id.astype(int), vocab.gene_symbol.astype(str)))
    migrated = []
    dropped = 0
    remapped = 0
    for row in edges.itertuples():
        gid = int(row.gene_id)
        symbol = gid_to_symbol[gid]
        target_gid = gid
        if ENSG_RE.fullmatch(symbol):
            stable = symbol.split(".")[0]
            target_symbol = None if stable in conflicts else ensg_map.get(stable)
            target_gid = symbol_to_gid.get(target_symbol, -1)
            if target_gid < 0:
                dropped += 1
                continue
            remapped += 1
        migrated.append({"runtime_program_id": int(row.runtime_program_id), "gene_id": int(target_gid)})
    migrated_df = pd.DataFrame(migrated).drop_duplicates(["runtime_program_id", "gene_id"])
    qmap = programs.set_index("runtime_program_id")["Q_program"].astype(float).to_dict()
    migrated_df["Q_program"] = migrated_df.runtime_program_id.map(qmap)
    migrated_df = migrated_df.sort_values(
        ["gene_id", "Q_program", "runtime_program_id"], ascending=[True, False, True]
    )
    pre_cap = len(migrated_df)
    degree_pre = migrated_df.groupby("gene_id").size()
    pruned = migrated_df.groupby("gene_id", sort=False).head(16)[["runtime_program_id", "gene_id"]]
    pruned = pruned.sort_values(["runtime_program_id", "gene_id"]).reset_index(drop=True)
    g = len(vocab)
    ids = torch.zeros((g, 16), dtype=torch.long)
    mask = torch.zeros((g, 16), dtype=torch.bool)
    degree = torch.zeros(g, dtype=torch.long)
    for gid, group in pruned.groupby("gene_id"):
        values = group.runtime_program_id.astype(int).tolist()
        ids[int(gid), : len(values)] = torch.as_tensor(values, dtype=torch.long)
        mask[int(gid), : len(values)] = True
        degree[int(gid)] = len(values)
    ensg_gids = vocab.loc[vocab.gene_symbol.astype(str).str.match(ENSG_RE.pattern), "global_gene_id"].astype(int).to_numpy()
    if int(degree[torch.as_tensor(ensg_gids)].sum()) != 0:
        raise RuntimeError("rebuilt P256 still assigns programs to ENSG pseudo-symbol rows")
    frozen = out / "program256_canonical_identity_m16"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copy2(program_root / "runtime_programs_256.tsv", frozen / "runtime_programs_256.tsv")
    shutil.copy2(program_root / "canonical_to_runtime_program.tsv", frozen / "canonical_to_runtime_program.tsv")
    pruned.to_csv(frozen / "runtime_gene_program_edges.tsv", sep="\t", index=False)
    torch.save({
        "gene_to_program_ids": ids,
        "gene_to_program_mask": mask,
        "gene_program_degree": degree,
        "runtime_program_count": 256,
        "membership_padding": 16,
        "program_dim": 128,
        "source": "rawlog1p_program256_reselection_m16_canonical_gene_identity",
    }, frozen / "gene_to_program_m16_overlay.pt")
    torch.save(degree > 0, frozen / "program_known_gene_mask.pt")
    audit = {
        "source_edges": int(len(edges)),
        "remapped_ensg_edges": int(remapped),
        "dropped_unresolved_or_conflict_ensg_edges": int(dropped),
        "edges_after_remap_dedup_before_m16": int(pre_cap),
        "genes_degree_gt16_before_cap": int((degree_pre > 16).sum()),
        "edges_after_m16": int(len(pruned)),
        "covered_genes": int((degree > 0).sum()),
        "max_degree": int(degree.max()),
        "ensg_pseudo_rows_with_positive_degree": 0,
        "selection_rule": "union canonical and resolved ENSG memberships; sort Q_program desc then runtime_program_id asc; cap M16",
        "overlay_path": str(frozen / "gene_to_program_m16_overlay.pt"),
        "overlay_sha256": sha256_file(frozen / "gene_to_program_m16_overlay.pt"),
        "status": "PASS",
    }
    write_json(out / "program_overlay_migration_audit.json", audit)
    write_json(frozen / "program256_freeze_contract.json", audit)
    return audit


def validate_outputs(split, manifest, hvg_map, vocab, rowmap_root: Path, program_audit, repair_rows):
    failures = []
    active = set(split.section_id.astype(str))
    m = manifest[manifest.sid.astype(str).isin(active)]
    if len(m) != EXPECTED["active_sections"] or m.sid.astype(str).nunique() != EXPECTED["active_sections"]:
        failures.append("active manifest section count/uniqueness mismatch")
    ensg_gids = set(vocab.loc[vocab.gene_symbol.astype(str).str.match(ENSG_RE.pattern), "global_gene_id"].astype(int))
    if set(hvg_map) != active:
        failures.append("HVG384 section keys do not equal active split sections")
    bad_hvg = []
    for sid, values in hvg_map.items():
        ids = list(map(int, torch.as_tensor(values).tolist()))
        if len(ids) != len(set(ids)) or len(ids) > 384 or any(gid in ensg_gids for gid in ids):
            bad_hvg.append(sid)
    if bad_hvg:
        failures.append(f"invalid HVG384 entries: {bad_hvg[:10]}")
    row_to_chunk = torch.load(rowmap_root / "row_to_gene_chunk_id.pt", map_location="cpu", weights_only=False).long().numpy()
    row_to_local = torch.load(rowmap_root / "row_to_gene_row_index.pt", map_location="cpu", weights_only=False).long().numpy()
    all_indices = []
    for part in ("train", "val", "test"):
        path = Path(split.attrs["split_dir"]) / f"{part}_row_indices.pt"
        ids = torch.load(path, map_location="cpu", weights_only=False).long().numpy()
        all_indices.append(ids)
        if len(ids) != EXPECTED["row_splits"][part]:
            failures.append(f"{part} row count mismatch")
    all_indices = np.concatenate(all_indices)
    if len(np.unique(all_indices)) != EXPECTED["active_rows"]:
        failures.append("active row IDs overlap or count mismatch")
    chunk_rows = dict(zip(manifest.gene_chunk_id.astype(int), manifest.n_rows.astype(int)))
    active_chunk = row_to_chunk[all_indices]
    active_local = row_to_local[all_indices]
    max_local = np.full(int(active_chunk.max()) + 1, -1, dtype=np.int64)
    np.maximum.at(max_local, active_chunk, active_local)
    for cid in np.flatnonzero(max_local >= 0):
        if int(max_local[cid]) >= int(chunk_rows[int(cid)]):
            failures.append(f"rowmap local row OOB for chunk {cid}")
    if program_audit["max_degree"] > 16 or program_audit["ensg_pseudo_rows_with_positive_degree"] != 0:
        failures.append("program overlay contract failed")
    if any(float(r["raw_total_rel_error"]) > 2e-7 for r in repair_rows):
        failures.append("corrected chunk raw-domain conservation failed")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--rowmap", type=Path, default=DEFAULT_ROWMAP)
    ap.add_argument("--h5ad-root", type=Path, default=DEFAULT_H5AD)
    ap.add_argument("--program-root", type=Path, default=DEFAULT_PROGRAM)
    ap.add_argument("--neighbor-overlay", type=Path, default=DEFAULT_SPAGENX_OVERLAY)
    args = ap.parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("workers must be in [1,16]")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "code").mkdir()
    shutil.copy2(Path(__file__), args.out / "code" / Path(__file__).name)
    (args.out / "commands.log").write_text(f"{datetime.now(timezone.utc).isoformat()} {sys.executable} {' '.join(sys.argv)}\n")
    write_json(args.out / "final_decision.json", {"decision": "FAIL_SpaGenX_17_GENE_IDENTITY_DATA_NOT_FIXED", "completed": False})
    (args.out / "status.md").write_text("RUNNING: preflight and source mapping\n")
    try:
        split = pd.read_csv(args.split, sep="\t", dtype=str)
        if "organ_id" in split.columns and "organ" not in split.columns:
            split = split.rename(columns={"organ_id": "organ"})
        if "split_v3" in split.columns and "split" not in split.columns:
            split = split.rename(columns={"split_v3": "split"})
        split.attrs["split_dir"] = str(args.split.parent)
        section_counts = split.split.value_counts().to_dict()
        if len(split) != EXPECTED["active_sections"] or section_counts != EXPECTED["section_splits"]:
            raise RuntimeError(f"split contract mismatch: {len(split)} {section_counts}")
        manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
        if manifest.sid.astype(str).duplicated().any():
            raise RuntimeError("manifest has duplicate section IDs")
        vocab = pd.read_csv(args.vocab, sep="\t")
        if len(vocab) != EXPECTED["vocab_rows"] or vocab.global_gene_id.duplicated().any() or vocab.gene_symbol.astype(str).duplicated().any():
            raise RuntimeError("vocabulary contract mismatch")
        symbol_to_gid = dict(zip(vocab.gene_symbol.astype(str), vocab.global_gene_id.astype(int)))
        gid_to_symbol = dict(zip(vocab.global_gene_id.astype(int), vocab.gene_symbol.astype(str)))
        active = set(split.section_id.astype(str))
        ensg_map, conflicts, provenance, source_rows = collect_source_mapping(active, args.h5ad_root, args.out)
        if len(conflicts) != 4:
            raise RuntimeError(f"expected 4 source mapping conflicts, found {len(conflicts)}")

        split_meta = split.set_index("section_id")[["organ", "split"]].to_dict("index")
        scan_tasks = []
        for row in manifest.itertuples():
            sid = str(row.sid)
            meta = split_meta.get(sid, {"organ": str(row.organ_id), "split": str(row.split)})
            scan_tasks.append((sid, meta["organ"], meta["split"], str(row.chunk_path), sid in active))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            scans = list(pool.map(scan_chunk, scan_tasks))
        scan_df = pd.DataFrame(scans).sort_values(["active", "organ", "section_id"], ascending=[False, True, True])
        scan_df.to_csv(args.out / "source_chunk_identity_scan.tsv", sep="\t", index=False)
        affected = scan_df[(scan_df.active) & (scan_df.n_ensg_columns > 0)].copy()
        if len(affected) != EXPECTED["active_affected_sections"]:
            raise RuntimeError(f"expected 158 affected active sections, found {len(affected)}")
        encountered = set()
        pmap = dict(zip(manifest.sid.astype(str), manifest.chunk_path.astype(str)))
        for sid in affected.section_id:
            obj = torch.load(pmap[sid], map_location="cpu", weights_only=False)
            encountered.update(s.split(".")[0] for s in map(str, obj["local_gene_symbols"]) if ENSG_RE.fullmatch(s))
        mapping_rows = []
        for stable in sorted(encountered):
            mapped_symbol = ensg_map.get(stable)
            if stable in conflicts:
                status = "CONFLICT_SOURCE_SYMBOL"
            elif mapped_symbol is None:
                status = "ABSENT_SOURCE_MAPPING"
            elif mapped_symbol not in symbol_to_gid:
                status = "SOURCE_SYMBOL_NOT_IN_VOCAB"
            else:
                status = "RESOLVED_EXACT_SOURCE_SYMBOL"
            mapping_rows.append({
                "ensembl_id": stable,
                "mapped_symbol": mapped_symbol or "",
                "canonical_global_gene_id": symbol_to_gid.get(mapped_symbol, ""),
                "status": status,
                "source_file_count": len(provenance.get(stable, set())),
            })
        write_tsv(args.out / "ensembl_symbol_mapping_audit.tsv", mapping_rows)

        (args.out / "status.md").write_text("RUNNING: rebuilding affected gene chunks\n")
        chunk_dir = args.out / "corrected_gene_assets" / "chunks"
        repair_tasks = []
        for row in affected.itertuples():
            repair_tasks.append((row.section_id, row.organ, row.split, row.chunk_path, str(chunk_dir / f"{row.section_id}.pt")))
        initargs = (ensg_map, conflicts, symbol_to_gid, gid_to_symbol)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=initargs) as pool:
            repair_rows = list(pool.map(repair_chunk, repair_tasks))
        repair_df = pd.DataFrame(repair_rows).sort_values(["organ", "section_id"])
        repair_df.to_csv(args.out / "corrected_chunk_audit.tsv", sep="\t", index=False)
        corrected_path = dict(zip(repair_df.section_id, repair_df.corrected_chunk))
        corrected_stats = repair_df.set_index("section_id").to_dict("index")
        corrected_manifest = manifest.copy()
        corrected_manifest["identity_fix_action"] = "REUSED_UNAFFECTED_READ_ONLY"
        for idx, row in corrected_manifest.iterrows():
            sid = str(row.sid)
            if sid not in corrected_path:
                continue
            audit = corrected_stats[sid]
            corrected_manifest.at[idx, "chunk_path"] = corrected_path[sid]
            corrected_manifest.at[idx, "n_local_genes"] = str(audit["n_genes_after"])
            corrected_manifest.at[idx, "nnz"] = str(audit["nnz_after"])
            corrected_manifest.at[idx, "nnz_raw"] = str(audit["nnz_after"])
            corrected_manifest.at[idx, "status"] = "REPAIRED_CANONICAL_GENE_IDENTITY_ADD_ONLY"
            corrected_manifest.at[idx, "identity_fix_action"] = "REBUILT_CANONICAL_GENE_IDENTITY"
        manifest_out = args.out / "corrected_gene_assets" / "gene_label_chunks_manifest.tsv"
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        corrected_manifest.to_csv(manifest_out, sep="\t", index=False)

        (args.out / "status.md").write_text("RUNNING: rebuilding section-HVG384 and P256/M16 overlay\n")
        active_manifest = corrected_manifest[corrected_manifest.sid.astype(str).isin(active)].copy()
        cpath = dict(zip(active_manifest.sid.astype(str), active_manifest.chunk_path.astype(str)))
        hvg_tasks = [(str(r.section_id), str(r.organ), str(r.split), cpath[str(r.section_id)]) for r in split.itertuples()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            hvg_results = list(pool.map(hvg384_task, sorted(hvg_tasks)))
        hvg_map = {sid: torch.as_tensor(ids, dtype=torch.long) for sid, ids, _ in hvg_results}
        hvg_rows = [row for _, _, row in hvg_results]
        hvg_path = args.out / "corrected_gene_assets" / "section_hvg384_ids.pt"
        torch.save(hvg_map, hvg_path)
        pd.DataFrame(hvg_rows).sort_values(["organ", "section_id"]).to_csv(
            args.out / "section_hvg384_rebuild_audit.tsv", sep="\t", index=False
        )
        program_audit = rebuild_program_overlay(args.program_root, args.out, vocab, ensg_map, conflicts)

        failures = validate_outputs(split, corrected_manifest, hvg_map, vocab, args.rowmap, program_audit, repair_rows)
        if failures:
            raise RuntimeError("; ".join(failures))
        untouched = {
            "rowmap_root": str(args.rowmap),
            "row_to_gene_chunk_id_sha256": sha256_file(args.rowmap / "row_to_gene_chunk_id.pt"),
            "row_to_gene_row_index_sha256": sha256_file(args.rowmap / "row_to_gene_row_index.pt"),
            "spagenx_neighbor_overlay": str(args.neighbor_overlay),
            "spagenx_neighbor_overlay_sha256": sha256_file(args.neighbor_overlay),
            "row_ids_rebuilt": False,
            "features_rebuilt": False,
            "neighbors_rebuilt": False,
            "split_rebuilt": False,
            "reason": "gene-column identity changed; row identity, feature identity, and section-local neighbor topology did not change",
            "status": "PASS_REUSED_READ_ONLY",
        }
        write_json(args.out / "row_feature_neighbor_reuse_audit.json", untouched)
        input_rows = []
        for role, path in [
            ("split", args.split), ("source_manifest", args.manifest), ("vocabulary", args.vocab),
            ("source_program_edges", args.program_root / "runtime_gene_program_edges.tsv"),
            ("source_programs", args.program_root / "runtime_programs_256.tsv"),
            ("source_program_overlay", args.program_root / "gene_to_program_m16_overlay.pt"),
        ]:
            input_rows.append({"role": role, "path": str(path), "sha256": sha256_file(path)})
        write_tsv(args.out / "input_hashes.tsv", input_rows)
        summary = {
            "active_sections": len(active),
            "active_rows": EXPECTED["active_rows"],
            "affected_sections_rebuilt": len(repair_rows),
            "affected_sections_by_organ": repair_df.organ.value_counts().sort_index().to_dict(),
            "affected_sections_by_split": repair_df.split.value_counts().sort_index().to_dict(),
            "resolved_ensg_columns": int(repair_df.get("MAP_ENSG_TO_CANONICAL", pd.Series(dtype=int)).sum()),
            "unresolved_ensg_columns_dropped": int(repair_df.get("DROP_UNRESOLVED", pd.Series(dtype=int)).sum()),
            "conflict_ensg_columns_dropped": int(repair_df.get("DROP_CONFLICT", pd.Series(dtype=int)).sum()),
            "collapsed_target_ids": int(repair_df.n_collapsed_target_ids.sum()),
            "corrected_manifest": str(manifest_out),
            "corrected_manifest_sha256": sha256_file(manifest_out),
            "section_hvg384": str(hvg_path),
            "section_hvg384_sha256": sha256_file(hvg_path),
            "program_overlay": program_audit["overlay_path"],
            "program_overlay_sha256": program_audit["overlay_sha256"],
            "existing_training_processes_automatically_switched": False,
        }
        write_json(args.out / "spagenx_gene_identity_fix_contract.json", summary)
        decision = {
            "decision": "PASS_SpaGenX_17_GENE_IDENTITY_DATA_FIXED",
            "completed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        write_json(args.out / "final_decision.json", decision)
        (args.out / "status.md").write_text("PASS_SpaGenX_17_GENE_IDENTITY_DATA_FIXED\n")
        (args.out / "final_report.md").write_text(
            "# SpaGenX-17 canonical gene-identity data fix\n\n"
            f"- Decision: `PASS_SpaGenX_17_GENE_IDENTITY_DATA_FIXED`\n"
            f"- Active corpus: {len(active):,} sections / {EXPECTED['active_rows']:,} rows\n"
            f"- Rebuilt affected sections: {len(repair_rows)}\n"
            f"- Corrected manifest: `{manifest_out}`\n"
            f"- Rebuilt section-HVG384: `{hvg_path}`\n"
            f"- Rebuilt canonical P256/M16 overlay: `{program_audit['overlay_path']}`\n\n"
            "The split, row IDs, image features, and SpaGenX neighbor topology were reused read-only. "
            "Existing runs still point to the old assets and were not modified or promoted.\n"
        )
        artifacts = []
        for path in sorted(args.out.rglob("*")):
            if not path.is_file() or "corrected_gene_assets/chunks" in str(path) or path.name == "artifact_index.json":
                continue
            artifacts.append({"path": str(path.relative_to(args.out)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        artifacts.append({"path": "corrected_gene_assets/chunks/", "file_count": len(repair_rows), "indexed_by": "corrected_chunk_audit.tsv"})
        write_json(args.out / "artifact_index.json", {"artifacts": artifacts})
        print(json.dumps(decision, indent=2, sort_keys=True))
    except Exception as exc:
        write_json(args.out / "final_decision.json", {
            "decision": "FAIL_SpaGenX_17_GENE_IDENTITY_DATA_NOT_FIXED",
            "completed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        (args.out / "status.md").write_text(f"FAIL_SpaGenX_17_GENE_IDENTITY_DATA_NOT_FIXED\n\n{exc}\n")
        (args.out / "final_report.md").write_text(
            f"# SpaGenX-17 canonical gene-identity data fix\n\nDecision: `FAIL_SpaGenX_17_GENE_IDENTITY_DATA_NOT_FIXED`\n\nError: {exc}\n"
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
