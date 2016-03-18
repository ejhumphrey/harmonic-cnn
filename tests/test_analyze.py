import numpy as np
import pandas
import pytest


@pytest.fixture
def testdata():
    n_classes = 5
    n_samples = 100
    change_pred_probability = .3
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_pred_new_indx = (np.random.random(size=n_samples) < \
        change_pred_probability).nonzero()[0]
    y_pred = np.array(y_true)
    y_pred[y_pred_new_indx] = np.random.randint(size=len(y_pred_new_indx))
    return y_true, y_pred


@pytest.fixture
def testdata_df(testdata):
    return pandas.DataFrame([y_true, y_pred],
                            columns=["target", "max_likleyhood"])


def test_prediction_analyzer(testdata):
