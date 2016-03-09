import numpy as np
import os
import pandas
import pescador
import pytest

import wcqtlib.config as C
import wcqtlib.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_cqt_slices():
    def __test_valid_cqt(cqt_frame, expected_len):
        assert len(cqt_frame.shape) == 4
        assert cqt_frame.shape[1] == 1
        assert cqt_frame.shape[2] == expected_len

    t_len = 5
    # Sample the first avialble featuer file
    generator = streams.cqt_slices(features_df.iloc[0], t_len)
    data = next(generator)
    yield __test_valid_cqt, data['x_in'], t_len

    t_len = 1
    generator = streams.cqt_slices(features_df.iloc[1], t_len)
    data = next(generator)
    yield __test_valid_cqt, data['x_in'], t_len

    # Now for fun, do it for 100 of them just to make sure
    # it keeps working after one cycle of frames.
    t_len = 10
    generator = streams.cqt_slices(features_df.iloc[2], t_len)
    for i in range(50):
        data = next(generator)
        yield __test_valid_cqt, data['x_in'], t_len


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_wcqt_slices():
    pass


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer():
    def __test_streamer(streamer, t_len, batch_size):
        batch = next(streamer)
        import pdb; pdb.set_trace()

    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    streamer = streams.InstrumentStreamer(
        features_df, datasets, streams.cqt_slices,
        t_len=t_len, batch_size=batch_size)
    yield __test_streamer, streamer, t_len, batch_size


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_with_zmq():
    pass
