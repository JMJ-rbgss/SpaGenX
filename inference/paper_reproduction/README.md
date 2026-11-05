# Native paper inference and evaluation

These files preserve the original paired SpaGenX/STPath experiment flow. Only release naming, the native-runtime import name, and external filesystem paths are changed. The SpaGenX forward, gene panels, row ordering, precision, masks, PCC/MAE calculations and aggregation are not rewritten.

## Files

| File | Original role |
|---|---|
| `common.py` | Frozen asset paths, identity checks, coordinate readers and Pearson helper |
| `preflight.py` | Verify checkpoints and Program buffers; align the frozen sections, spot rows, HVG panels and measured truth |
| `worker.py` | Native frozen inference with original feature stores and neighborhoods; export ordered predictions |
| `run.py` | Original two-model smoke gate and worker coordination |
| `evaluate.py` | Recompute section–gene PCC, section MAE, aggregate results and paired comparisons |

The original comparison-model branch is retained so that this is the actual experimental entry point, not a newly written SpaGenX-only pipeline. It requires the corresponding frozen STPath assets and external STPath implementation when executing the full paired workflow.

## External assets and paths

Set `SPAGENX_DATA_ROOT` to the external frozen asset bundle, `SPAGENX_CODE_ROOT` to this repository's `training/` directory, and `RUN_ROOT` to a fresh experiment directory. `common.py` lists all expected paths and checkpoint hashes. In particular, the asset bundle contains:

- `weights/SpaGenX_weights.pt`;
- `pipeline/step1_features/master_feature_store.pt`, `pipeline/step3_rowmap_17organs/`, and `pipeline/step4_overlays_17organs/spagenx_pool32_homo_hetero_overlay.pt`;
- `canonical_assets/corrected_gene_assets/` and `canonical_assets/program256_canonical_identity_m16/`;
- `pipeline/step2_gene_rawlog1p/global_gene_vocab.tsv` and `split/section_split_17organs_dedup.tsv`;
- `test_contract/result/`, including the original row/spot alignment plan and section manifest;
- `frozen_panels/analysis/section_hvg50_selection_primary.tsv` and its original diagnostic prediction cache under `frozen_panels/predictions/`;
- the paired comparison model, feature stores, tokenizer assets and coordinate inputs at the paths declared in `common.py`.

Paths are external aliases for the original assets; file contents and embedded row/spot/gene identities must remain unchanged. The two coordinate roots are `coordinates_primary/` and `coordinates_secondary/`. No weights, images, patient tables, feature stores, prediction caches or expression matrices are included in this repository.

## Run the original flow

From the repository root:

```bash
export SPAGENX_DATA_ROOT=/external/spagenx_data
export SPAGENX_CODE_ROOT="$PWD/training"
export RUN_ROOT=/external/spagenx_runs/paper_reproduction

mkdir -p "$RUN_ROOT/code"
cp inference/paper_reproduction/*.py "$RUN_ROOT/code/"
python "$RUN_ROOT/code/preflight.py"
python "$RUN_ROOT/code/run.py"
```

Execution is immediate. `run.py` retains the original physical GPU assignments 6 and 7; inspect them and confirm resource availability before running. Preflight and worker smoke gates are retained. This command does not train or fine-tune either model.

With genuine completed prediction artifacts and completion logs already present in `RUN_ROOT`, evaluation can be rerun without model inference:

```bash
python "$RUN_ROOT/code/evaluate.py"
```

## Protocol and outputs

SpaGenX uses the frozen paper checkpoint, original UNI2-h feature store, stored candidate/homo/hetero neighborhoods, actual organ/technology IDs, BF16 and the original patch batch of 64. Expression prompts are empty and disabled; learned gene bias is enabled. The web WSI-grid protocol is a separate entry point and is not substituted here.

Predictions under `predictions/SpaGenX/` contain `pred`, `row_ids`, `spot_ids`, `gene_ids` and prompt-state metadata. Prepared inputs and truth are written under `02_cohort/`; metrics are written under `03_metrics/`.

For the paper's section-equal result, use `section_weighted_PCC` from `overall_metrics.tsv`: despite the original field name, it is an equal arithmetic average of section PCC values. Each section PCC is the strict average of the frozen 50 per-gene Pearson correlations over spots. `resource_collection_macro_PCC` is a different aggregation and is reported separately. Section MAE is stored in `section_hvg50_pcc.tsv`, with per-gene MAE in `section_gene_pcc.tsv`. No `nanmean` replaces the strict result.

`source_manifest.json` records original/release hashes and source-preservation checks. The runtime data-loading helpers were compared with the original experiment dependency and have identical ASTs after naming normalization. The evaluator was also rerun on the original saved predictions and truth in an isolated directory; validation details are recorded there.
