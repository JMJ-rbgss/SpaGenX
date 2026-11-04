# Native split and test-cleaning sources

This directory contains the original section-splitting and frozen test-index construction code. Changes are restricted to external path configuration and release filenames. Assignment rules, seeds, balancing objectives, filtering, assertions and output schemas are retained.

## 1. Section splitting

`build_split_v3.py` is the original base-organ splitter. `build_mixed_replay_split.py` retains the frozen base-organ assignments and builds the new-organ assignments. These are fixed research-contract scripts, not arbitrary-dataset split generators. Their original expected-count and source-exclusion checks remain in the code.

```bash
export SPAGENX_DATA_ROOT=/external/spagenx_data
python preprocessing/native_split/build_split_v3.py \
  --input_rowmap_manifest "$SPAGENX_DATA_ROOT/base_rowmap/rowmap_manifest.tsv" \
  --output_dir "$SPAGENX_DATA_ROOT/old_split"

python preprocessing/native_split/build_mixed_replay_split.py \
  --rowmap_tsv "$SPAGENX_DATA_ROOT/rowmap/rowmap_manifest.tsv" \
  --gene_manifest "$SPAGENX_DATA_ROOT/gene_label_chunks_manifest.tsv" \
  --exclude_sections "$SPAGENX_DATA_ROOT/excluded_sections.txt" \
  --output_dir "$SPAGENX_DATA_ROOT/split"
```

Use fresh output directories. The base-organ splitter also receives its original expected-size arguments when a matching frozen contract is being replayed. These commands execute immediately.

## 2. Frozen test-index cleaning

`build_clean_test_index.py` is the original index-construction script. It reads the frozen section split and verified patient-audit results, removes the already identified overlapping test sections, and preserves global row identities. It does not generate the upstream patient-overlap audit.

Required external paths under `SPAGENX_DATA_ROOT`:

- `split/section_split_17organs_dedup.tsv` and `split/test_row_indices.pt`;
- `rowmap/row_to_gene_chunk_id.pt`;
- `gene_label_chunks_manifest.tsv`;
- `patient_audit/test_patient_leak_sections.tsv`;
- `patient_audit/test_after_removing_known_leaks.tsv`.

```bash
mkdir -p runs/native_test_clean
python preprocessing/native_split/build_clean_test_index.py runs/native_test_clean
```

The destination must exist and must not contain an `index` directory. Outputs are written under `index/`. Audit tables and patient metadata are external inputs and are not distributed here.

`source_manifest.json` records the original and release hashes, plus the exact permitted path changes. No replacement algorithm or synthetic patient-audit generator is included.
