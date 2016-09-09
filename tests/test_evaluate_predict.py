"""
Tests for evalute.predict.predict_one, and evalute.predict.predict_many.

Also serves to make sure all the theano-complied functions for training
and evaluating work when passed some real data.
"""

import logging
import os
import pandas
import pytest
import sys

import hcnn.common.config as C
import hcnn.evaluate.analyze
import hcnn.evaluate.predict
import hcnn.train.models as models
import hcnn.train.streams as streams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.load(CONFIG_PATH)


@pytest.fixture(params=[(streams.cqt_slices, models.cqt_iX_c1f1_oY),
                        (streams.cqt_slices, models.cqt_iX_c2f2_oY),
                        # (streams.wcqt_slices, models.wcqt_iX_c1f1_oY),
                        # (streams.wcqt_slices, models.wcqt_iX_c2f2_oY),
                        (streams.hcqt_slices, models.hcqt_iX_c1f1_oY),
                        (streams.hcqt_slices, models.hcqt_iX_c2f2_oY)],
                ids=["cqt_c1f1", "cqt_c2f2",
                     # "wcqt_c1f1", "wcqt_c2f2",
                     "hcqt_c1f1", "hcqt_c2f2"])
def slicer_and_model(request):
    return request.param


@pytest.fixture(scope="module")
def feats_df(tiny_feats):
    return tiny_feats.to_df()


def test_predict_one(slicer_and_model, feats_df):
    # TODO: random seeds for consistency / reproducability.
    print("Running on:")
    # Get a record
    test_record = feats_df.iloc[0]
    other_record = feats_df.iloc[1]
    assert test_record['instrument'] != other_record['instrument']

    # Pick a model
    t_len = 8
    n_classes = 12
    slicer = slicer_and_model[0]
    network_def = slicer_and_model[1](t_len, n_classes)
    model = models.NetworkManager(network_def)

    # Train this a couple frames to bias the eval in favor of this file.
    iter_count = 0
    max_count = 10
    for frames in slicer(test_record, t_len):
        loss = model.train(frames)
        print("Quick train ", iter_count, "loss:", loss)
        iter_count += 1
        if iter_count >= max_count:
            break

    result = hcnn.evaluate.predict.predict_one(
        test_record, model, slicer, t_len)
    for key in ["mean_loss", "mean_acc", "max_likelihood", "vote", "target"]:
        assert key in result
    assert result['max_likelihood'] == result['target']
    assert result['vote'] == result['target']
    print("Test Result:\n", result)

    other = hcnn.evaluate.predict.predict_one(
        other_record, model, slicer, t_len)
    for key in ["mean_loss", "mean_acc", "max_likelihood", "vote", "target"]:
        assert key in other
    print("Other Result:\n", other)


def test_predict_dataframe(slicer_and_model, feats_df):
    # For the purposes of this we don't care too much about what we train with.
    # TODO: random seeds for consistency / reproducability.
    test_df = feats_df.sample(n=(12 * 12), replace=True)

    # Pick a model
    t_len = 8
    n_classes = 12
    slicer = slicer_and_model[0]
    network_def = slicer_and_model[1](t_len, n_classes)
    model = models.NetworkManager(network_def)

    # Create the streamer.
    streamer = streams.InstrumentStreamer(test_df,
                                          record_slicer=slicer,
                                          t_len=t_len,
                                          batch_size=12)

    # Train for a little bit.
    iter_count = 0
    max_count = 100
    for batch in streamer:
        loss = model.train(batch)
        print("Batch ", iter_count, "loss:", loss)
        iter_count += 1
        if iter_count >= max_count:
            break

    # Run evaluation on some number of datapoints (10ish),
    #  and make sure you get a dataframe back
    eval_df = hcnn.evaluate.predict.predict_many(
        test_df, model, slicer, t_len)

    # TODO: why is this even necessary?
    eval_df = eval_df.dropna()

    assert isinstance(eval_df, pandas.DataFrame)
    assert len(eval_df) == len(test_df)

    analyzer = hcnn.evaluate.analyze.PredictionAnalyzer(eval_df, test_df)
    print(analyzer.classification_report)
    print(analyzer.pprint())
