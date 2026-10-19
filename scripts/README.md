# Automatic WSI pipeline

`infer_wsi.sh` connects WSI preprocessing, frozen feature extraction, neighborhood construction and SpaGenX expression inference.

Set `WSI`, `ORGAN`, `GENES_FILE`, a new absolute `RUN_DIR`, and the configured bundle assets, then run:

```bash
bash scripts/infer_wsi.sh --dry-run
# Run only after inspecting the preview:
bash scripts/infer_wsi.sh --execute
```

The default is preview-only. Preprocessing failures stop the pipeline. [Full setup and examples](../docs/USAGE.zh-CN.md#4-快速开始自动wsi推理).
