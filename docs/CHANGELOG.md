# Release changes

- Public-facing model naming is SpaGenX throughout the README and usage guide. The example checkpoint filename is `SpaGenX_weights.pt`; its SHA256 and contents are unchanged. Historical filenames, configuration lookup keys, serialized field names and original classes remain compatibility identifiers, not separate model names.

- Native model and shared module byte-identical to the formal SpaGenX source.
- Native training function/class ASTs unchanged; paths mechanically relocatable.
- New dry-run launchers require explicit execution and fresh outputs.
- Validated WSI wrapper retained, with display documentation updated and paths configured externally.
- Historical in-place/site-specific scripts guarded from direct execution.
- Patient literals replaced by synthetic markers in portable source; original copies/diffs containing patient literals are omitted from the local download.
- External asset schemas, source hashes, notices, synthetic/boundary tests and private native-equivalence checks added.
- Test fixture setup corrections are recorded in private logs; production model/trainer unchanged.
- No weights, data, credentials, formal training or GitHub push.
