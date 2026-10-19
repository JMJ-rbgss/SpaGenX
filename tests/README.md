# Test suites

```bash
python tests/test_release.py
python tests/test_automation.py
```

`test_release.py` checks synthetic identity/mask behavior, neighborhood padding and preprocessing boundaries. `test_automation.py` uses only the Python standard library and checks preview safety, missing configuration and existing-output rejection.

Tests do not start formal training. Private-asset equivalence checks recorded during packaging are described in [the validation report](../docs/validation.json).
