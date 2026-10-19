# Native SpaGenX training

The native model, shared modules and recovery trainer preserve the SpaGenX implementation. The launcher uses PyTorch distributed execution with one process per configured GPU.

```bash
python training/run.py --config configs/train.local.json
# Start only after checking all assets and outputs:
python training/run.py --config configs/train.local.json --execute
```

The supplied example requires an external step1000 full-state recovery checkpoint. Final inference weights cannot substitute for optimizer/recovery state. [Training instructions and torchrun template](../docs/USAGE.zh-CN.md#7-pytorch四卡训练).
