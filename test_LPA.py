import pytest

import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt
from LPA import LPA
from algo import KLD_distance


# class TestDataset(unittest.TestCase):
#     def setUp(self):
#         dvr = pd.read_csv("test_dvr.csv")
#         self.lpa = LPA(dvr, epsilon_frac=2)

# def assert_cols(cols, asserted_cols):
#     print()


@pytest.fixture
def freq():
    return pd.read_csv("test_data/test_freq.csv")


@pytest.fixture
def real_dvr():
    return pd.read_csv("test_data/test_dvr.csv")


@pytest.fixture
def real_pvr():
    return pd.read_csv("test_data/test_pvr.csv")


@pytest.fixture
def lpa(real_dvr):
    return LPA(real_dvr, epsilon_frac=2)


def assert_categories(df):
    assert len(df.drop_duplicates("category")) == 3


def assert_probability(df: pd.DataFrame, column: str, groupby: None | str = None):
    if groupby:
        # TODO: perhaps approx?
        assert df.groupby(groupby)[column].sum().all()
    else:
        assert pytest.approx(df[column].sum()) == 1


def test_dvr(freq, real_dvr):
    dvr = LPA.create_dvr(freq)
    pdt.assert_frame_equal(dvr, real_dvr)
    assert set(dvr.columns.tolist()) == {"element", "global_weight"}
    assert_probability(dvr, "global_weight")


def test_pvr(freq, lpa, real_pvr):
    pvr = lpa.create_pvr(freq)
    pdt.assert_frame_equal(pvr, real_pvr)
    assert_probability(pvr, "local_weight", groupby="category")
    assert_categories(pvr)


def test_KLD_distance():
    # np.log(0)?
    P = np.array([1, 0, 3])
    Q = np.array([1, 4, 4])
    npt.assert_array_almost_equal(KLD_distance(P, Q), np.array([0, 0, np.log(4 / 3)]))


def test_signatures(freq, real_dvr):
    lpa = LPA(real_dvr, epsilon_frac=2)
    signatures, max_distances = lpa.create_signatures(
        freq, overused=True, most_significant=30, sig_length=500
    )


# class TestModels(unittest.TestCase):
#     @params(*list(zip(A_cases, A1_expected_values)))
#     def test_A1(self, params, expected):
#         self.assertEqual(Models.A1(**params), expected)
