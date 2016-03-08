import numpy as np
import os
import pandas
import pescador
import pytest

import wcqtlib.train.streams as streams

DATA_ROOT = os.path.expanduser("~/data")
features_path = os.path.expanduser("~/data/ismir2016-wcqt-data/featuresdf.pkl")
features_df = pandas.read_pickle(features_path)


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
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


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_wcqt_slices():
    pass


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_buffer_streams():
    pass


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_zmq_buffered_streams():
    pass
