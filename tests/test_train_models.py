import logging
import numpy as np
import os
import pandas
import pytest
import sys

import wcqtlib.config as C
import wcqtlib.train.streams as streams
import wcqtlib.train.models as models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


def test_push_data_through_simple_network_cqt():
    """Prove that the network accepts data in the expected shape,
    and returns data in the expected shape."""
    # Create some random data in the right shape
    t_len = 8
    n_targets = 2
    x_in = np.random.random((1, 1, t_len, 252))
    target = np.asarray((np.random.randint(n_targets),), dtype=np.int32)

    # Create the network
    train, predict = models.cqt_iX_c1f1_oY(t_len, n_targets)

    # Push it through
    train_loss = train(x_in, target)
    assert np.isfinite(train_loss)

    predict_loss, predict_acc = predict(x_in, target)
    assert np.isfinite(predict_loss) and np.isfinite(predict_acc)

    # Create some more data, perhaps of a different batch size
    n = 24
    x_in = np.random.random((n, 1, t_len, 252))
    target = np.asarray(np.random.randint(n_targets, size=n), dtype=np.int32)

    # Push it through
    train_loss = train(x_in, target)
    assert np.isfinite(train_loss)

    predict_loss, predict_acc = predict(x_in, target)
    assert np.isfinite(predict_loss) and np.isfinite(predict_acc)


@pytest.mark.slowtest
def test_overfit_two_samples_cqt():
    """Prove that our network works by training it with two random files
    from our dataset, intentionally overfitting it.

    Warning: not deterministic, but you could make it.
    """

    # Get list of instruments
    instruments = sorted(features_df["instrument"].unique())
    selected_instruments = instruments[:2]

    # Create a dataframe from our dataframe with only two files in it
    test_df = pandas.concat([
        features_df[features_df["instrument"] ==
                    selected_instruments[0]].sample(),
        features_df[features_df["instrument"] ==
                    selected_instruments[1]].sample()])

    t_len = 8
    batch_size = 8
    datasets = ["rwc"]
    n_targets = 2
    # Create a streamer that samples just those two files.
    streamer = streams.InstrumentStreamer(
        test_df, datasets, streams.cqt_slices,
        t_len=t_len, batch_size=batch_size)

    # Create a new model
    train, predict = models.cqt_iX_c1f1_oY(t_len, n_targets)

    # Train the model for N epochs, till it fits the damn thing
    max_batches = 250
    i = 0
    for batch in streamer:
        x, y = batch["x_in"], np.asarray(batch["target"], dtype=np.int32)
        train_loss = train(x, y)

        i += 1
        print("Batch: ", i, "Loss: ", train_loss)
        if i >= max_batches:
            break

    # Evaluate it. On the original files. Should do well.
    eval_batch = next(streamer)
    eval_loss, accuracy = predict(eval_batch["x_in"],
                                  np.asarray(eval_batch["target"],
                                             dtype=np.int32))
    print("Eval Loss:", eval_loss, "Accuracy:", accuracy)
    assert np.isfinite(eval_loss) and np.isfinite(accuracy)
    np.testing.assert_approx_equal(accuracy, 1.0, significant=1)

    # Evaluate it on a random other file. Should do terribly.
