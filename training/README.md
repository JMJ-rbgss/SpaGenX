# Native SpaGenX training

The native model, shared modules and recovery trainer preserve the SpaGenX implementation. The launcher uses PyTorch distributed execution with one process per configured GPU.

```bash
python training/run.py --config configs/train.local.json
# Start only after checking all assets and outputs:
python training/run.py --config configs/train.local.json --execute
```

The default example starts at step 0 with seed-42 random initialization and does not load a checkpoint. Copy `configs/train.example.json` to `configs/train.local.json` and configure the data and output paths before running the commands above.

To resume an interrupted run, use `configs/train.resume.example.json` instead. Set the full-state checkpoint path, its saved step and the remaining training steps, keeping the same GPU count. Recovery is optional, not a prerequisite for training. See [training instructions and torchrun template](../docs/USAGE.zh-CN.md#7-pytorch四卡训练).
