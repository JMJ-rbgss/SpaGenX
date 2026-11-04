#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

R = Path(os.environ.get("SPAGENX_DATA_ROOT", "/external/spagenx_data"))
SRC = R/"split"
SPLIT = SRC/"section_split_17organs_dedup.tsv"
SRC_TEST_ROWS = SRC/"test_row_indices.pt"
ROW_TO_CHUNK = R/"rowmap/row_to_gene_chunk_id.pt"
MANIFEST = R/"gene_label_chunks_manifest.tsv"
LEAK = R/"patient_audit/test_patient_leak_sections.tsv"
NOLEAK = R/"patient_audit/test_after_removing_known_leaks.tsv"

OUT = Path(sys.argv[1]).resolve()
assert OUT.exists(), OUT
assert not (OUT/"index").exists(), (OUT/"index")
(OUT/"index").mkdir(parents=True, exist_ok=False)

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def atomic_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")

split = pd.read_csv(SPLIT, sep="\t", dtype=str)
split["n_rows_int"]=split["n_rows"].astype(int)
assert set(["section_id","split","organ","dataset_id","tech_id","n_rows","replay_group"]).issubset(split.columns), split.columns.tolist()
test = split[split["split"]=="test"].copy()
noleak = pd.read_csv(NOLEAK, sep="\t", dtype=str)
leak = pd.read_csv(LEAK, sep="\t", dtype=str)
noleak_ids=set(noleak["section_id"].astype(str))
leak_ids=set(leak["section_id"].astype(str))
test_ids=set(test["section_id"].astype(str))
assert len(test)==182 and len(test_ids)==182, (len(test),len(test_ids))
assert len(noleak)==162 and len(noleak_ids)==162, (len(noleak),len(noleak_ids))
assert len(leak)==20 and len(leak_ids)==20, (len(leak),len(leak_ids))
assert noleak_ids==(test_ids-leak_ids), "162 != 182 - leak20"

manifest=pd.read_csv(MANIFEST, sep="\t", dtype=str)
assert manifest["gene_chunk_id"].astype(int).is_unique
chunk2sid=dict(zip(manifest["gene_chunk_id"].astype(int), manifest["sid"].astype(str)))
row_to_chunk=torch.load(ROW_TO_CHUNK, map_location="cpu", weights_only=False)
src_rows=torch.load(SRC_TEST_ROWS, map_location="cpu", weights_only=False)
assert src_rows.dtype==torch.int64 and len(src_rows)==344382
chunk_ids=row_to_chunk[src_rows].numpy()
sids=np.array([chunk2sid.get(int(c), "") for c in chunk_ids], dtype=object)
assert (sids!="").all()
assert set(np.unique(sids))==test_ids, (len(np.unique(sids)), len(test_ids))

# exact per-section counts on source universe
cnt=pd.Series(sids).value_counts()
for r in test.itertuples():
    assert int(cnt.get(r.section_id,0))==int(r.n_rows_int), (r.section_id, int(cnt.get(r.section_id,0)), int(r.n_rows_int))

mask=np.isin(sids, sorted(noleak_ids))
clean_rows=src_rows[mask]
clean_sids=sids[mask]
clean_section_ids=set(np.unique(clean_sids).tolist())
assert clean_section_ids==noleak_ids
assert int(clean_rows.numel())==int(noleak["n_rows"].astype(int).sum())==329612
assert len(np.unique(clean_rows.numpy()))==len(clean_rows)
cnt2=pd.Series(clean_sids).value_counts()
for r in noleak.itertuples():
    assert int(cnt2.get(r.section_id,0))==int(r.n_rows), (r.section_id, int(cnt2.get(r.section_id,0)), int(r.n_rows))

# frozen manifest: keep audit fields, ordered by section_id
order=sorted(noleak_ids)
man=noleak.copy()
man["n_rows_int"]=man["n_rows"].astype(int)
rep=test.set_index("section_id")["replay_group"].to_dict()
man["replay_group"]=man["section_id"].map(rep)
man=man.sort_values("section_id")
cols=["section_id","organ","dataset_id","tech_id","n_rows","patient_id","replay_group"]
man["n_rows"]=man["n_rows_int"]
man[cols].to_csv(OUT/"index/clean162_section_manifest.tsv", sep="\t", index=False)
torch.save(clean_rows.cpu(), OUT/"index/clean162_row_indices.pt")

row_map=pd.DataFrame({
    "row_index": clean_rows.numpy(),
    "section_id": clean_sids,
})
row_map["organ"]=row_map["section_id"].map(test.set_index("section_id")["organ"].to_dict())
row_map.to_csv(OUT/"index/clean162_row_to_section.tsv", sep="\t", index=False)

removed=leak.copy()
removed.to_csv(OUT/"index/clean162_removed_leak_sections.tsv", sep="\t", index=False)

contract={
    "status":"PASS_CANONICAL_CLEAN162_INDEX_FROZEN",
    "source_test_sections":182,
    "source_test_rows":int(len(src_rows)),
    "removed_leak_sections":20,
    "clean_sections":162,
    "clean_rows":int(clean_rows.numel()),
    "unique_rows":int(np.unique(clean_rows.numpy()).size),
    "duplicate_rows":int(clean_rows.numel()-np.unique(clean_rows.numpy()).size),
    "missing_sections":0,
    "expected_rows_from_manifest":int(noleak["n_rows"].astype(int).sum()),
    "row_identity_unique":True,
    "removed_section_ids":sorted(leak_ids),
    "input_sha256":{
        "section_split_17organs_dedup.tsv":sha(SPLIT),
        "source_test_row_indices.pt":sha(SRC_TEST_ROWS),
        "row_to_gene_chunk_id.pt":sha(ROW_TO_CHUNK),
        "gene_label_chunks_manifest.tsv":sha(MANIFEST),
        "patient_leak_sections.tsv":sha(LEAK),
        "test_after_removing_known_leaks.tsv":sha(NOLEAK),
    },
    "output_sha256":{
        "clean162_section_manifest.tsv":sha(OUT/"index/clean162_section_manifest.tsv"),
        "clean162_row_indices.pt":sha(OUT/"index/clean162_row_indices.pt"),
        "clean162_row_to_section.tsv":sha(OUT/"index/clean162_row_to_section.tsv"),
    },
    "assertions":{
        "162_equals_182_minus_20_leak":True,
        "source_rows_344382":True,
        "every_source_section_row_count_matches_split":True,
        "clean_rows_329612":True,
        "every_clean_section_row_count_matches_manifest":True,
        "no_missing_or_duplicate_sections":True,
        "no_duplicate_rows":True,
    },
}
atomic_json(OUT/"index/clean162_contract.json", contract)
print(json.dumps(contract, indent=2, sort_keys=True))
print("OUT", OUT)
