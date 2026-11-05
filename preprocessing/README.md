# Research data preprocessing

Stages cover sample discovery, frozen UNI2-h features, expression raw-log1p and measured masks, exact rowmap matching, typed neighborhoods, canonical identity/Program correction and active-row HVG384.

```bash
python preprocessing/run.py --help
```

Each stage requires verified inputs and a new output path. The launcher previews commands unless `--execute` is supplied. Preserve measured zeros and do not infer missing gene identities. Historical in-place migration sources remain protected.

## Canonical → active-row HVG384 → training

Run the `canonical` stage before `hvg384`. With the example output paths:

| Consumer | Input from the preceding stage |
| --- | --- |
| `hvg384 --manifest` | `runs/new_canonical/corrected_gene_assets/gene_label_chunks_manifest.tsv` |
| `hvg384 --old-hvg` | `runs/new_canonical/corrected_gene_assets/section_hvg384_ids.pt` |
| Training `--section_hvg_path` | `runs/new_hvg384/section_hvg384_active_rows.pt` |

The canonical stage generates the full-chunk HVG list itself; no separately downloaded historical HVG file is needed for this handoff. The native active-row script requires `--old-hvg` only to report overlap and changed slots. It independently selects the new HVG384 from corrected expression rows retained by the rowmap, with measured-gene masking and variance/global-ID ordering. The old list does not influence selection.

The example preprocessing configuration connects both canonical outputs to `hvg384`. If the canonical output directory changes, update both paths. Configure the training gene manifest and Program overlay from that same canonical output, and point training `--section_hvg_path` to the active-row output above; paths are not transferred automatically.

[Input contracts](../docs/DATA_CONTRACT.md) · [Step-by-step usage](../docs/USAGE.zh-CN.md)
