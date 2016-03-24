import os
import pandas
import pytest

import wcqtlib.common.config as C
import wcqtlib.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path) \
    if os.path.exists(features_path) else pandas.DataFrame()


def test_cqt_slices():
    @pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                                 os.path.exists(features_path),
                                 not features_df.empty]),
                        reason="Data not found.")
    def __test_slicer(t_len, *slicer_args):
        slicer = streams.cqt_slices(features_df.iloc[0], t_len,
                                    *slicer_args)
        for i in range(10):
            data = next(slicer)['x_in']
            assert len(data.shape) == 4
            assert data.shape[1] == 1
            assert data.shape[2] == t_len

    yield __test_slicer, 5
    yield __test_slicer, 1
    yield __test_slicer, 10
    yield __test_slicer, 8, False, False


def test_wcqt_slices():
    @pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                                 os.path.exists(features_path),
                                 not features_df.empty]),
                        reason="Data not found.")
    def __test_slicer(t_len, *slicer_args):
        slicer = streams.wcqt_slices(features_df.iloc[0], t_len,
                                     *slicer_args)
        for i in range(10):
            data = next(slicer)['x_in']
            assert len(data.shape) == 4
            assert data.shape[1] == 1
            assert data.shape[2] == t_len

    yield __test_slicer, 5
    yield __test_slicer, 1
    yield __test_slicer, 10
    yield __test_slicer, 8, False, False


@pytest.mark.xfail(reason="zmq is always generating batches of only one.")
def __test_streamer(streamer, t_len, batch_size):
    counter = 0
    while counter < 5:
        batch = next(streamer)
        assert batch is not None
        assert 'x_in' in batch and 'target' in batch
        assert len(batch['x_in'].shape) == 4
        assert len(batch['target'].shape) == 1
        assert batch['x_in'].shape[2] == t_len
        assert batch['x_in'].shape[0] == batch_size
        assert batch['target'].shape[0] == batch_size
        print("Result batch size:", batch['x_in'].shape[0])
        counter += 1


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_cqt():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    if not features_df.empty:
        streamer = streams.InstrumentStreamer(
            features_df, datasets, streams.cqt_slices,
            t_len=t_len, batch_size=batch_size)
        yield __test_streamer, streamer, t_len, batch_size


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_wcqt():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    if not features_df.empty:
        streamer = streams.InstrumentStreamer(
            features_df, datasets, streams.wcqt_slices,
            t_len=t_len, batch_size=batch_size)
        yield __test_streamer, streamer, t_len, batch_size


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_with_zmq():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    if not features_df.empty:
        streamer = streams.InstrumentStreamer(
            features_df, datasets, streams.cqt_slices,
            t_len=t_len, batch_size=batch_size, use_zmq=True)
        yield __test_streamer, streamer, t_len, batch_size
