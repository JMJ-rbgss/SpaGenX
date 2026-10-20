# Test suites

```bash
python tests/test_release.py
python tests/test_automation.py
```

`test_release.py` checks synthetic identity/mask behavior, neighborhood padding and preprocessing boundaries. `test_automation.py` uses only the Python standard library and checks preview safety, missing configuration and existing-output rejection.

The tests run synthetic and automation checks. Model-equivalence checks performed with private test assets are recorded in [the validation report](../docs/validation.json).
