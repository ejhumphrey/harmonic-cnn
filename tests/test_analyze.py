import numpy as np
import os
import pandas
import pytest

import hcnn.evaluate.analyze as analyze


@pytest.fixture
def testdata():
    n_classes = 5
    n_samples = 100
    change_pred_probability = .3
    y_true = np.random.randint(0, n_classes, size=n_samples)
    y_pred_new_indx = (np.random.random(size=n_samples) <
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
    max_likelihood = pandas.Series(testdata[1], name="max_likelihood")
    mean_loss = pandas.Series(testdata[2], name="mean_loss")
    dataset = pandas.Series(testdata[3], name="dataset")
    return pandas.concat([target, max_likelihood, mean_loss, dataset], axis=1)


def test_prediction_analyzer(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set="rwc")
    assert analyzer is not None


def test_save_load_analyzer(workspace, testdata_df):
    test_set = "rwc"
    analyzer = analyze.PredictionAnalyzer(testdata_df, test_set=test_set)
    save_path = os.path.join(workspace, "analyzer.pkl")
    analyzer.save(save_path)
    assert os.path.exists(save_path)

    # Prove you can load it back in, too.
    analyzer2 = analyze.PredictionAnalyzer.load(save_path, test_set)
    assert analyzer.y_true == analyzer2.y_true
    assert analyzer.y_pred == analyzer2.y_pred


def test_metrics(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df)
    assert analyzer.y_true == testdata_df["target"].tolist()
    assert analyzer.y_pred == testdata_df["max_likelihood"].tolist()
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


def test_dataset_class_wise(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df)
    scores = analyzer.dataset_class_wise()
    assert set(scores.index.levels[0]) == \
        set(["overall", "rwc", "uiowa", "philharmonia"])
    assert scores.shape[1] == 4


def test_dataset_summary(testdata_df):
    analyzer = analyze.PredictionAnalyzer(testdata_df)
    scores = analyzer.dataset_summary()
    assert set(scores.index) == \
        set(["overall", "rwc", "uiowa", "philharmonia"])
    assert scores.shape[1] == 4
