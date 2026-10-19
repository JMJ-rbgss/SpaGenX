# External data and protocol contracts

All real data remain external. Placeholder paths in JSON must be replaced explicitly; relocating the data root does not rewrite paths embedded inside manifests. Do not upload private manifests, images, expression or feature tensors.

## Training corpus and targets

The native SpaGenX lineage uses 17 organs and four technology IDs, 1536-dimensional frozen UNI2-h image features, canonical vocabulary capacity115055, Program256 with at most16 memberships per gene, and candidate32/homo12/hetero8 typed neighborhoods. Training uses the original 960-section/2339452-row subset; validation63 and historical test182 belong to its source split. This is not the latest162-section/329612-spot expression benchmark, and no claim of that evaluation is made by the packaging tests.

Corrected gene chunks must preserve `target_genes_raw_local_csr`, `observed_gene_mask_local`, and `local_to_global_gene_idx`, with explicit spot/row identities and finite raw-log1p values. A measured zero remains a supervised zero; absence of a probe is not a zero. Canonical ID collisions are handled by the original correction code, not fuzzy matching. Rowmap lookup must retain the stored row order and local-to-global identities.

The actual active-row HVG384 builder uses only rows present in the rowmap, masks unobserved genes, ranks variance with global-ID tie breaking, and preserves its original384-gene requirement. The source asset has1205 sections; training sampling queries only the original training subset. Do not substitute test HVG50, a new organ-level web panel, or the earlier full-chunk HVG list.

## Required external asset schemas

- Canonical vocabulary: TSV with integer`global_gene_id` and`gene_symbol`. Preserve numbering; duplicate symbols require IDs or explicit rejection.
- Codebook: JSON containing the frozen17-name`organ_id` mapping and technology mapping; do not re-enumerate by alphabetical order.
- Program overlay: trusted torch serialization containing`gene_to_program_ids` and`gene_to_program_mask`; exact buffers must match the SpaGenX weights.
- Training: original train/val indices, section splits, feature store, corrected chunk manifest, rowmap, typed-neighbor overlay and active-row HVG384.
- Experimental inference: trusted per-slide`features` (`N×1536`), patch coordinates and hash, plus candidate/homo/hetero local indices, validity masks, distances and cosine features with identical row identity.

External assets are not granted redistribution rights by this schema documentation. Exact verified hashes are listed separately. Additional historical migration inputs include source h5ad files and original Program membership tables; these are not fabricated or bundled.

## Protocol differences

The native training overlay retains its original distance units and padding. The WSI application wrapper builds a new full-slide grid and normalized slide-local distances; it must not be advertised as reproducing training sampling. Frozen experimental overlays can be passed directly to inference without reconstruction. Inference uses no expression prompt and the frozen SpaGenX weight configuration; training retains its native step5000 prompt/gene-bias schedule. No new PCC evaluation or alternative early-stopping metric is introduced by this release.
