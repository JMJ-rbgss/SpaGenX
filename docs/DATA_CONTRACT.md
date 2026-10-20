# External data and protocol contracts

All real data remain external. Placeholder paths in JSON must be replaced explicitly; relocating the data root does not rewrite paths embedded inside manifests. Do not upload private manifests, images, expression or feature tensors.

## Training corpus and targets

SpaGenX uses organ and technology metadata, 1536-dimensional frozen UNI2-h image features, a canonical gene vocabulary, Program256 with at most16 memberships per gene, and candidate32/homo12/hetero8 typed neighborhoods. Provide the corresponding data manifests and split files as external inputs.

Corrected gene chunks must preserve `target_genes_raw_local_csr`, `observed_gene_mask_local`, and `local_to_global_gene_idx`, with explicit spot/row identities and finite raw-log1p values. A measured zero remains a supervised zero; absence of a probe is not a zero. Canonical ID collisions are handled by the original correction code, not fuzzy matching. Rowmap lookup must retain the stored row order and local-to-global identities.

The active-row HVG384 builder uses rows present in the rowmap, masks unobserved genes, and ranks variance with global-ID tie breaking to select 384 genes per section. Training sampling queries the training subset defined by the split files.

## Required external asset schemas

- Canonical vocabulary: TSV with integer`global_gene_id` and`gene_symbol`. Preserve numbering; duplicate symbols require IDs or explicit rejection.
- Codebook: JSON containing the frozen17-name`organ_id` mapping and technology mapping; do not re-enumerate by alphabetical order.
- Program overlay: trusted torch serialization containing`gene_to_program_ids` and`gene_to_program_mask`; exact buffers must match the SpaGenX weights.
- Training: original train/val indices, section splits, feature store, corrected chunk manifest, rowmap, typed-neighbor overlay and active-row HVG384.
- Experimental inference: trusted per-slide`features` (`N×1536`), patch coordinates and hash, plus candidate/homo/hetero local indices, validity masks, distances and cosine features with identical row identity.

Verify external assets against the hashes in `external_assets.json` and follow their applicable access and licensing terms. Historical migrations additionally require source h5ad files and original Program membership tables.

## Protocol differences

The native training overlay uses spot-based sampling, stored distance units and padding. The WSI application wrapper uses a full-slide grid and normalized slide-local distances. Existing experimental overlays can be passed directly to inference. Inference uses no expression prompt and the SpaGenX weight configuration; training uses its native step5000 prompt/gene-bias schedule and validation metric.
