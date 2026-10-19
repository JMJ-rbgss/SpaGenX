# Frozen expression inference

`spagenx_cli.py` supports WSI preprocessing and inference from trusted, matching feature/neighbor files. The current protocol fixes step42000, no expression prompt, gene bias enabled and tech_zero.

```bash
python inference/spagenx_cli.py --help
```

Configure the external assets first. For automatic WSI processing, use [the orchestration script](../scripts/README.md). See [the complete guide](../docs/USAGE.zh-CN.md) for inputs and output fields. Historical experiment source is guarded against direct execution.
