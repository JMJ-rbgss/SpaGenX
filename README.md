# SpaGenX

![SpaGenX data construction and framework overview](assets/SpaGenX_overview.png)

*SpaGenX data construction and framework overview. Expression prompts are disabled in current42K inference.*

[中文完整使用说明（PyTorch commands）](docs/USAGE.zh-CN.md) · [中文概览](README.zh-CN.md)

The Chinese guide includes installation, external assets, automated WSI inference, output inspection, preprocessing stages, a standard four-GPU `torchrun` template and troubleshooting. The optional `scripts/infer_wsi.sh` automation previews commands by default and runs only with `--execute`.

Research code for preprocessing, native SpaGenX training, and frozen current42K inference, with an overview illustration. This package contains **no weights, patient-level data, raw research images, gene dictionaries, feature caches, or optimizer states**. It is not a clinical service. Review the validation report before use. The repository is private and requires authorized GitHub access.

## Layout

- `preprocessing/`: tissue features, measured-expression masks and raw-log1p, rowmap, spatial neighbors, canonical identity/Program M16 correction, and actual active-row HVG384 construction.
- `training/`: unchanged native SpaGenX model and formal recovery trainer; configuration-based dry-run launcher.
- `inference/`: fixed-checkpoint WSI application CLI and frozen-feature experimental source archive.
- `configs/`, `tests/`, `docs/`: external-asset contracts, synthetic tests, provenance hashes and protocol explanations.

## Environment and external assets

Use Python 3.9 and install `requirements.txt` in a separate environment. OpenSlide also needs its platform shared library. The recorded test-environment versions are in `docs/validation.json`. Do not use untrusted pickle-based `.pt` uploads. Supply trusted, separately obtained assets listed in `docs/external_assets.json`; their inclusion here does not grant redistribution rights.

Copy `configs/bundle.example.json` to `bundle_config.json`, edit only the external asset locations, and keep the model/preprocessor code locations relative to this repository. Checkpoint identity is fixed to current42K. Obtain UNI2-h independently under its applicable terms; no automatic download occurs.

## Preprocessing and training

Edit placeholder paths in the example configs. Commands are dry-run by default:

```sh
python preprocessing/run.py --stage features --config configs/preprocessing.example.json
python preprocessing/run.py --stage canonical --config configs/preprocessing.example.json
python preprocessing/run.py --stage hvg384 --config configs/preprocessing.example.json
python training/run.py --config configs/train.example.json
```

Add `--execute` only after checking all input/output paths. Outputs must not already exist; `--overwrite` is rejected by the preprocessing launcher. `SPAGENX_DATA_ROOT` relocates historical source defaults, but manifests must also contain valid local paths. The 17-organ extension and older HVG rebuild scripts are **historical migration sources**, not safe one-command pipeline entrypoints: they originally modified a working dataset in place. They are guarded against direct execution and must not be run on source assets. Use the current canonical correction/HVG CLI stages for add-only construction. No full-dataset reconstruction is claimed by the small tests.

The training example documents the recovery/continuation contract and requires an external step1000 full-state checkpoint. It is not a from-scratch recipe for bitwise reproduction of current42K. Do not replace the missing recovery state with the published final weights and call it the same training lineage. Full training, optimizer history and training data are not shipped. The trainer retains its historical validation logic; latest section-HVG50 benchmark numbers are not the same as its early training metric. The new homo/hetero ablation trainer is deliberately excluded.

## Frozen inference

```sh
python inference/spagenx_cli.py --bundle . preprocess --image /external/slide.svs --out runs/new_slide --device cuda:0
python inference/spagenx_cli.py --bundle . infer --features runs/new_slide/features.pt --neighbors runs/new_slide/neighbors.pt --genes-file /external/genes.txt --organ Breast --out runs/new_predictions --device cuda:0
```

For images lacking MPP, supply `--mpp`. Gene input accepts unambiguous canonical symbols or IDs from the external frozen dictionary. Inference requires `global_step=42000`, gene bias enabled, tech_zero, empty expression prompt and explicit prompt disable. Native parameter names and dimensions are unchanged for strict checkpoint loading. An output ID's existence does not guarantee that it was supervised or reliably predicted.

The WSI wrapper uses a new non-overlapping 224px grid at 0.5 microns/px, grayscale threshold205/white fraction0.8, stops above16000 accepted patches, and uses slide-local normalized distances. It must not be described as reproducing the original spatial experimental sampling or raw-distance training graph. `experiment_pseudobulk_archive.py` preserves the experimental protocol as a reference and is guarded against direct execution; the public CLI can score trusted frozen feature/neighbor files without rebuilding their graph.

## Tests and publication

```sh
python tests/test_release.py
```

Small tests and native-equivalence checks do not establish full training reproducibility or clinical validity. No license grant is inferred: the owner must choose the license and confirm third-party redistribution eligibility before publication. See `docs/THIRD_PARTY_NOTICES.md`. Keep the sibling `private_audit/` directory out of GitHub.
