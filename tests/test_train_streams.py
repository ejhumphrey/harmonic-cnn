import os
import pytest

import hcnn.common.config as C
import hcnn.data.cqt
import hcnn.data.dataset
import hcnn.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.load(CONFIG_PATH)


def __assert_cqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.cqt_slices(dataset.to_df().iloc[0], t_len,
                                *slicer_args)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] == 1
        assert data.shape[2] == t_len


def __assert_wcqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.wcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] > 1 and data.shape[1] < 10
        assert data.shape[2] == t_len


def __assert_hcqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.hcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] == 6
        assert data.shape[2] == t_len


@pytest.mark.parametrize(
    "slicer_test",
    [__assert_cqt_slicer,
     __assert_wcqt_slicer,
     __assert_hcqt_slicer],
    ids=["cqt", "wcqt", "hcqt"])
def test_slices(slicer_test, tiny_feats):
    slicer_test(tiny_feats, 5)
    slicer_test(tiny_feats, 1)
    slicer_test(tiny_feats, 10)
    slicer_test(tiny_feats, 8, False, False)


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


@pytest.mark.parametrize("slicer", [streams.cqt_slices,
                                    streams.wcqt_slices,
                                    streams.hcqt_slices],
                         ids=["cqt", "wcqt", "hcqt"])
def test_instrument_streamer(slicer, tiny_feats):
    df = tiny_feats.to_df()
    t_len = 10
    batch_size = 12
    if not df.empty:
        streamer = streams.InstrumentStreamer(
            df, slicer, t_len=t_len, batch_size=batch_size)
        __test_streamer(streamer, t_len, batch_size)


@pytest.mark.xfail(reason="zmq is always generating batches of only one.")
def test_instrument_streamer_with_zmq(tiny_feats):
    df = tiny_feats.to_df()
    t_len = 10
    batch_size = 12
    if not df.empty:
        streamer = streams.InstrumentStreamer(
            df, streams.cqt_slices,
            t_len=t_len, batch_size=batch_size, use_zmq=True)
        __test_streamer(streamer, t_len, batch_size)
