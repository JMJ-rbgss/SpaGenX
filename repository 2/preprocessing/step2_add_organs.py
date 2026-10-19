#!/usr/bin/env python3

raise RuntimeError('ARCHIVAL_ONLY: historical in-place/site-specific protocol; use documented current CLI stages, not direct execution')
import sys, importlib.util, csv, json, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch

BASE = Path((__import__('os').environ.get('SPAGENX_DATA_ROOT', '/external/spagenx_data') + '/new_pipeline_800gb'))
RUN = BASE / "full_run_firstv1_rawlog1p_20260703_110639"
STEP2_DIR = RUN / "step2_gene_rawlog1p"
CHUNK_DIR = STEP2_DIR / "gene_label_store" / "chunks"
INV = BASE.parent / "hest_stimage_17organs_human_20260819/step0_sample_inventory_additional10.tsv"
SPLIT = BASE / "split_additional10_pretrain.csv"
VOCAB = STEP2_DIR / "global_gene_vocab.tsv"
MANIFEST = STEP2_DIR / "gene_label_chunks_manifest.tsv"
SOURCE_TSV = STEP2_DIR / "sample_gene_source_selection.tsv"

sys.path.insert(0, str(BASE/"scripts"))
import step2_gene_expression_preprocessing as S

# 1) old vocab
vocab = pd.read_csv(VOCAB, sep="\t").fillna("")
gene_to_global = {str(r["gene_symbol"]): int(r["gene_index"]) for _, r in vocab.iterrows()}
print("old vocab genes", len(gene_to_global))

# 2) old manifest cid max + old ng_code map
old_manifest = pd.read_csv(MANIFEST, sep="\t").fillna("")
max_cid = int(old_manifest["gene_chunk_id"].max())
ng_code = {}
if "normalization_group_id" in old_manifest and "normalization_group_code" in old_manifest:
    for ng, code in old_manifest.groupby("normalization_group_id")["normalization_group_code"].first().items():
        ng_code[str(ng)] = int(code)
print("old chunks", len(old_manifest), "max_cid", max_cid, "old ng groups", len(ng_code))

# 3) inventory + split
inv = pd.read_csv(INV, sep="\t").fillna("")
inv = inv[inv["status"].astype(str).eq("PASS")].copy()
split = pd.read_csv(SPLIT).fillna("")
split["sample_id"] = split["sample_id"].astype(str)
split["formal_split"] = split["split"].map(S.split_norm)
inv["sample_id"] = inv["sample_id"].astype(str)
inv = inv.merge(split[["sample_id","formal_split"]].drop_duplicates("sample_id"), on="sample_id", how="left")
assert not inv["formal_split"].isna().any(), "split missing"
print("new samples to process", len(inv))

# 4) source audit for new samples (parallel)
items = [(i, S.row_dict(r)) for i, (_, r) in enumerate(inv.iterrows())]
source_rows = []
with ProcessPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(S.vocab_worker, item) for item in items]
    for fut in as_completed(futs):
        _, audit = fut.result(); source_rows.append(audit)
source_rows.sort(key=lambda r: r["sample_id"])
source_audit_by_sid = {r["sample_id"]: r for r in source_rows}
blocked = [r for r in source_rows if not bool(r["usable_for_gene_preprocessing"])]
print("blocked new samples", len(blocked))
for b in blocked:
    print("BLOCKED", b["sample_id"], b["blocking_reason"])
inv = inv[[bool(source_audit_by_sid[str(r["sample_id"])]["usable_for_gene_preprocessing"]) for _, r in inv.iterrows()]].copy()
print("usable new samples", len(inv))

# 5) ng_code for new rows
for _, r in inv.iterrows():
    ng = S.norm_group(S.row_dict(r))
    if ng not in ng_code:
        ng_code[ng] = len(ng_code)

# 6) build chunk items
chunk_items = []
for idx, (_, r) in enumerate(inv.iterrows()):
    cid = max_cid + 1 + idx
    row = S.row_dict(r)
    row["sample_id"] = str(row.get("sample_id") or row.get("sid"))
    row["sid"] = row["sample_id"]
    row["formal_split"] = str(r["formal_split"])
    chunk_items.append((cid, row))

# 7) process chunks in parallel
manifest_rows = []
sample_audits = []
with ProcessPoolExecutor(max_workers=16, initializer=S.init_chunk_worker,
                         initargs=(gene_to_global, ng_code, str(CHUNK_DIR), source_audit_by_sid)) as ex:
    futs = [ex.submit(S.chunk_worker, item) for item in chunk_items]
    for done, fut in enumerate(as_completed(futs), start=1):
        cid, manifest, audit = fut.result()
        manifest_rows.append(manifest); sample_audits.append(audit)
        if done % 20 == 0:
            print(f"[chunk] {done}/{len(futs)} {manifest['sample_id']} rows={manifest['n_rows']} genes={manifest['n_local_genes']} nnz={manifest['nnz']} status={manifest['status']}", flush=True)

manifest_rows.sort(key=lambda x: x["gene_chunk_id"])
sample_audits.sort(key=lambda x: x["sample_id"])
new_manifest = pd.DataFrame(manifest_rows)
print("new manifest rows", len(new_manifest))
print(new_manifest["status"].value_counts().to_dict())

# 8) append to old manifest
combined = pd.concat([old_manifest, new_manifest], ignore_index=True)
combined.to_csv(MANIFEST, sep="\t", index=False)
print("combined manifest rows", len(combined))

# 9) append source selection rows (without genes) for future resume
visible = [{k:v for k,v in r.items() if k != "genes"} for r in source_rows]
new_source = pd.DataFrame(visible)
if SOURCE_TSV.exists():
    old_source = pd.read_csv(SOURCE_TSV, sep="\t").fillna("")
    combined_source = pd.concat([old_source, new_source], ignore_index=True)
else:
    combined_source = new_source
combined_source.to_csv(SOURCE_TSV, sep="\t", index=False)
print("combined source rows", len(combined_source))

# 10) final report
fail_rows = new_manifest[~new_manifest["status"].astype(str).isin(["OK","SKIPPED_EXISTING"])]
print("FAIL_CHUNKS", len(fail_rows))
print(json.dumps({"new_chunks": int(len(new_manifest)), "combined_chunks": int(len(combined)), "blocked": len(blocked), "vocab_size": len(gene_to_global), "status": "PASS_WITH_BLOCKED" if len(fail_rows)==0 else "FAIL", "blocked": len(blocked)}, indent=2))
