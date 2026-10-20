# Native SpaGenX training

The native model, shared modules and recovery trainer preserve the SpaGenX implementation. The launcher uses PyTorch distributed execution with one process per configured GPU.

```bash
python training/run.py --config configs/train.local.json
# Start only after checking all assets and outputs:
python training/run.py --config configs/train.local.json --execute
```

The example continues from an external step1000 full-state checkpoint containing model parameters, optimizer state and recovery metadata. Prepare it separately from inference weights. See [training instructions and torchrun template](../docs/USAGE.zh-CN.md#7-pytorch四卡训练).
