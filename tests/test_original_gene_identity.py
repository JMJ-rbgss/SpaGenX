#!/usr/bin/env python3
import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPT = Path(__file__).resolve().parents[1] / "preprocessing/prepare_gene_identity_assets.py"
spec = importlib.util.spec_from_file_location("fixer", SCRIPT)
fixer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixer)


def run_tests():
    fixer.init_worker(
        {"ENSG00000000001": "A", "ENSG00000000002": "B"},
        {"ENSG00000000003"},
        {"A": 0, "B": 1, "C": 2},
        {0: "A", 1: "B", 2: "C"},
    )
    raw = np.asarray([[1, 2, 3, 7, 9], [0, 4, 0, 8, 10]], dtype=np.float32)
    csr = sp.csr_matrix(np.log1p(raw))
    symbols = ["A", "ENSG00000000001", "B", "ENSG00000000003", "NOPE"]
    gids = np.asarray([0, 10, 1, 11, 12])
    observed = np.ones(5, dtype=bool)
    out, out_gids, stats = fixer.collapse_chunk_matrix(csr, symbols, gids, observed)
    assert out_gids.tolist() == [0, 1]
    recovered = np.expm1(out.toarray())
    np.testing.assert_allclose(recovered, np.asarray([[3, 3], [4, 0]], dtype=np.float32), rtol=1e-6, atol=1e-6)
    assert stats["n_collapsed_target_ids"] == 1
    assert stats["DROP_CONFLICT"] == 1
    assert stats["DROP_LOCAL_SYMBOL_NOT_IN_VOCAB"] == 1
    assert stats["raw_total_rel_error"] < 1e-7


if __name__ == "__main__":
    run_tests()
    print("PASS")
