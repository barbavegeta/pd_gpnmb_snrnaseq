import numpy as np
import pandas as pd
from scipy import sparse

from pd_gpnmb.pseudobulk import aggregate_sparse_counts


def test_sparse_pseudobulk_sum():
    x = sparse.csr_matrix(np.array([[1, 0], [2, 3], [4, 1]], dtype=int))
    obs = pd.DataFrame({"sample_id": ["s1", "s1", "s2"]})
    counts, meta = aggregate_sparse_counts(x, obs, ["G1", "G2"], ["sample_id"])
    dense = counts.sparse.to_dense()
    assert list(meta["sample_id"]) == ["s1", "s2"]
    assert dense.loc["pb_000", "G1"] == 3
    assert dense.loc["pb_000", "G2"] == 3
    assert dense.loc["pb_001", "G1"] == 4
