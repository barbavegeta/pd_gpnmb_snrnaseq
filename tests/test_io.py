import numpy as np
import pandas as pd
import pytest

from pd_gpnmb.io import check_clinical_join


def _obs(conditions):
    return pd.DataFrame(
        {
            "sample_id": ["s_0001", "s_0001", "s_0002", "s_0003"],
            "condition": conditions,
        }
    )


def test_clinical_join_passes_when_all_donors_matched():
    check_clinical_join(_obs(["PD", "PD", "Control", "Control"]))


def test_clinical_join_fails_on_unmatched_donor():
    with pytest.raises(ValueError, match="s_0003"):
        check_clinical_join(_obs(["PD", "PD", "Control", np.nan]))


def test_clinical_join_fails_on_unmapped_label():
    with pytest.raises(ValueError, match="Unmapped"):
        check_clinical_join(_obs(["PD", "PD", "Control", "iLBD"]))
