from defectpl.utils import calc_deltaQ


def test_calc_deltaQ():
    struct1_path = "./tests/data/CONTCAR_gs"
    struct2_path = "./tests/data/CONTCAR_es"
    deltaQ = calc_deltaQ(struct1_path, struct2_path)
    expected_deltaQ = 0.5218583063350908
    assert abs(deltaQ - expected_deltaQ) < 1e-4
