import numpy as np
import pandas
import pytest

import wcqtlib.train.analyze as analyze


@pytest.fixture
def testdata():
    n_classes = 5
    n_samples = 100
    change_pred_probability = .3
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_pred_new_indx = (np.random.random(size=n_samples) < \
        change_pred_probability).nonzero()[0]
    y_pred = np.array(y_true)
    y_pred[y_pred_new_indx] = np.random.randint(0, n_classes,
                                                size=len(y_pred_new_indx))
    dataset = [("rwc", "uiowa", "philharmonia")[x] for
               x in np.random.randint(0, 3, size=n_samples)]
    return y_true, y_pred, dataset


@pytest.fixture
def testdata_df(testdata):
    target = pandas.Series(testdata[0], name="target")
    max_likelyhood = pandas.Series(testdata[1], name="max_likelyhood")
    dataset = pandas.Series(testdata[2], name="dataset")
    return pandas.concat([target, max_likelyhood, dataset], axis=1)


def test_prediction_analyzer(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    assert analyzer is not None


def test_save_analyzer():
    pass


def test_load_analyzer():
    pass


def test_metrics():
    pass


def test_class_wise_scores():
    pass


def test_summary_scores():
    pass
