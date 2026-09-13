import gzip

from pd_gpnmb.io import read_clinical


def test_gse243639_semicolon_table_after_prose(tmp_path):
    path = tmp_path / "clinical.csv.gz"
    text = (
        "The presence of Lewy bodies, neuritic plaques and tangles is estimated as follows:\n"
        "1. prose, with, commas; and semicolons\n"
        "N;Brain Bank ID;Sample ID;Clinical diagnosis;Age;Sex;PMI hours;RIN measure;"
        "Lewy bodies presence in midbrain;Lewy bodies presence in limbic regions (amygdala);"
        "Lewy bodies presence in neocortical regions (frontal cortex);"
        "CERAD score for neuritic plaques;Braak stage for neurofibrillary tangles\n"
        "1;2020;s.0096;Parkinson's;90;female;3;7.8;1;1;1;1;4\n"
        "2;2140;s.0098;Control;85;male;5;7.1;0;0;0;0;2\n"
    )
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)

    df = read_clinical(path)
    assert list(df.index) == ["s_0096", "s_0098"]
    assert df.loc["s_0096", "condition"] == "PD"
    assert df.loc["s_0098", "condition"] == "Control"
    assert df.loc["s_0096", "age"] == 90
    assert df.loc["s_0096", "sex"] == "Female"
    assert df.loc["s_0096", "pmi_hours"] == 3
    assert df.loc["s_0096", "rin"] == 7.8
    assert df.loc["s_0096", "cerad_score"] == 1
    assert df.loc["s_0096", "braak_stage"] == 4
