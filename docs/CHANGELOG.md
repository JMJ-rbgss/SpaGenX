# Release changes

- Standardized source filenames, imports, classes, configuration keys, comments and exported metadata under the SpaGenX name.
- Default training starts from step 0 with seed-42 random initialization. Optional full-state resume has a separate configuration.
- Model state-dict keys, tensor shapes, forward calculations, sampling, losses and optimization remain unchanged.
- Configurable external Program and neighborhood inputs; the source-table requirements and canonical overlay output are documented.
- Site-specific experiment and launch archives are retained only in the private provenance archive, outside this release.
- Source hashes and technical validation have been updated for this revision.
- Model weights, images, expression matrices and patient data remain external.
