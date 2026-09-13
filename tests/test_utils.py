from pd_gpnmb.utils import canonical_cell_id, canonical_condition, canonical_sample_id, sample_from_cell_id


def test_sample_id_normalisation():
    assert canonical_sample_id("s.0096") == "s_0096"
    assert canonical_sample_id("s_0096") == "s_0096"
    assert canonical_sample_id("S0096") == "s_0096"


def test_cell_id_normalisation():
    assert canonical_cell_id("s.0096_AAACCCA-1") == "s_0096_AAACCCA-1"
    assert sample_from_cell_id("s.0096_AAACCCA-1") == "s_0096"


def test_condition_mapping():
    assert canonical_condition("Parkinson's") == "PD"
    assert canonical_condition("Control") == "Control"
