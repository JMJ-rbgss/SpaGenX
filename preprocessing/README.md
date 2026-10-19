# Research data preprocessing

Stages cover sample discovery, frozen UNI2-h features, expression raw-log1p and measured masks, exact rowmap matching, typed neighborhoods, canonical identity/Program correction and active-row HVG384.

```bash
python preprocessing/run.py --help
```

Each stage requires verified inputs and a new output path. The launcher previews commands unless `--execute` is supplied. Preserve measured zeros and do not infer missing gene identities. Historical in-place migration sources remain protected.

[Input contracts](../docs/DATA_CONTRACT.md) · [Step-by-step usage](../docs/USAGE.zh-CN.md)
