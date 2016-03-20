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
    mean_loss = np.random.random(n_samples)
    dataset = [("rwc", "uiowa", "philharmonia")[x] for
               x in np.random.randint(0, 3, size=n_samples)]
    return y_true, y_pred, mean_loss, dataset


@pytest.fixture
def testdata_df(testdata):
    target = pandas.Series(testdata[0], name="target")
    max_likelyhood = pandas.Series(testdata[1], name="max_likelyhood")
    mean_loss = pandas.Series(testdata[2], name="mean_loss")
    dataset = pandas.Series(testdata[3], name="dataset")
    return pandas.concat([target, max_likelyhood, mean_loss, dataset], axis=1)


def test_prediction_analyzer(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    assert analyzer is not None


def test_save_analyzer():
    pass


def test_load_analyzer():
    pass


def test_metrics(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    assert all(analyzer.y_true == testdata_df["target"])
    assert all(analyzer.y_pred == testdata_df["max_likelyhood"])
    assert isinstance(analyzer.mean_loss, float)
    assert np.sum(analyzer.support) == len(testdata_df)
    assert len(analyzer.tps) == len(testdata_df)


def test_class_wise_scores(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    scores = analyzer.class_wise_scores()
    assert set(scores.columns) == set(["precision", "recall",
                                      "f1score", "support"])
    assert all(np.isfinite(scores))


def test_summary_scores(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    scores = analyzer.summary_scores()
    assert set(scores.index) == set(["precision", "recall",
                                    "f1score", "accuracy"])
    assert all(np.isfinite(scores))
