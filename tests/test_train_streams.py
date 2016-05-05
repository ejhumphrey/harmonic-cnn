import os
import pytest

import wcqtlib.common.config as C
import wcqtlib.data.cqt
import wcqtlib.data.dataset
import wcqtlib.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="module")
def tiny_feats(module_workspace, tinyds):
    limited_ds = wcqtlib.data.dataset.Dataset(tinyds.observations[0:2])
    return wcqtlib.data.cqt.cqt_from_dataset(
        limited_ds, module_workspace, skip_existing=True)


def __test_cqt_slicer(dataset, t_len, *slicer_args):
    result = []
    slicer = streams.cqt_slices(dataset.to_df().iloc[0], t_len,
                                *slicer_args)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] == 1
        assert data.shape[2] == t_len

    return all(result)


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


def test_cqt_slices(tiny_feats):
    __test_cqt_slicer(tiny_feats, 5)
    __test_cqt_slicer(tiny_feats, 1)
    __test_cqt_slicer(tiny_feats, 10)
    __test_cqt_slicer(tiny_feats, 8, False, False)


def test_wcqt_slices(tiny_feats):
    __assert_wcqt_slicer(tiny_feats, 5)
    __assert_wcqt_slicer(tiny_feats, 1)
    __assert_wcqt_slicer(tiny_feats, 10)
    __assert_wcqt_slicer(tiny_feats, 8, False, False)


def test_hcqt_slices(tiny_feats):
    __assert_hcqt_slicer(tiny_feats, 5)
    __assert_hcqt_slicer(tiny_feats, 1)
    __assert_hcqt_slicer(tiny_feats, 10)
    __assert_hcqt_slicer(tiny_feats, 8, False, False)


# @pytest.mark.xfail(reason="zmq is always generating batches of only one.")
# def __test_streamer(streamer, t_len, batch_size):
#     counter = 0
#     while counter < 5:
#         batch = next(streamer)
#         assert batch is not None
#         assert 'x_in' in batch and 'target' in batch
#         assert len(batch['x_in'].shape) == 4
#         assert len(batch['target'].shape) == 1
#         assert batch['x_in'].shape[2] == t_len
#         assert batch['x_in'].shape[0] == batch_size
#         assert batch['target'].shape[0] == batch_size
#         print("Result batch size:", batch['x_in'].shape[0])
#         counter += 1


# def test_instrument_streamer_cqt(tiny_feats):
#     df = tiny_feats.to_df()
#     t_len = 10
#     batch_size = 12
#     datasets = ["rwc"]
#     if not df.empty:
#         streamer = streams.InstrumentStreamer(
#             df, datasets, streams.cqt_slices,
#             t_len=t_len, batch_size=batch_size)
#         yield __test_streamer, streamer, t_len, batch_size


# def test_instrument_streamer_wcqt(tiny_feats):
#     df = tiny_feats.to_df()
#     t_len = 10
#     batch_size = 12
#     datasets = ["rwc"]
#     if not df.empty:
#         streamer = streams.InstrumentStreamer(
#             df, datasets, streams.wcqt_slices,
#             t_len=t_len, batch_size=batch_size)
#         yield __test_streamer, streamer, t_len, batch_size


# def test_instrument_streamer_with_zmq(tiny_feats):
#     df = tiny_feats.to_df()
#     t_len = 10
#     batch_size = 12
#     datasets = ["rwc"]
#     if not df.empty:
#         streamer = streams.InstrumentStreamer(
#             df, datasets, streams.cqt_slices,
#             t_len=t_len, batch_size=batch_size, use_zmq=True)
#         yield __test_streamer, streamer, t_len, batch_size
