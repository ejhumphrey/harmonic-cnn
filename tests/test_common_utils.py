import numpy as np
import os
import pytest

import hcnn.common.utils as utils


def __eq(a, b):
    assert a == b


def test_create_directory(workspace):
    dname = os.path.join(workspace, "oh_hello")
    assert not os.path.exists(dname)
    assert utils.create_directory(dname)
    assert utils.create_directory(dname)


def test_filebase():
    fnames = ['y', 'y.z', 'x/y.z', 'x.y.z']
    results = ['y', 'y', 'y', 'x.y']
    for fn, res in zip(fnames, results):
        yield __eq, utils.filebase(fn), res


@pytest.mark.skipif(True, reason='todo')
def test_fold_array():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_map_io():
    assert False


def test_unzip():
    zip_files = [os.path.join(os.path.dirname(__file__), "zipped_folder.zip")]
    unzipped_folders = utils.unzip_files(zip_files)
    for dir_path in unzipped_folders:
        assert os.path.exists(dir_path) and os.path.isdir(dir_path)


@pytest.mark.skipif(True, reason='todo')
def test_slice_ndarray():
    assert False


@pytest.mark.skipif(True, reason='todo')
def test_colored():
    assert False


def test_iter_from_params_filepath():
    tests = [
        ("foo/bar/params/params0500.npz", "0500"),
        ("foo/bar/params/params10505.npz", "10505"),
        ("foo/bar/params/params0000.npz", "0000"),
        ("foo/bar/params/final.npz", "final"),
    ]

    def __test(value, expected):
        assert value == expected

    for test_input, expected in tests:
        yield __test, utils.iter_from_params_filepath(test_input), expected


def test_filter_df(tinyds):
    datasets_df = tinyds.to_df()
    # Test filtering on different Instruments
    filtered_df = utils.filter_df(
        datasets_df, instrument="bassoon")
    assert filtered_df["instrument"].unique() == ["bassoon"]

    filtered_df = utils.filter_df(
        datasets_df, instrument="trumpet")
    assert filtered_df["instrument"].unique() == ["trumpet"]

    # Test filtering on different datasets and dataset combinations.
    filtered_df = utils.filter_df(
        datasets_df, datasets=["rwc"])
    assert filtered_df["dataset"].unique() == ["rwc"]

    filtered_df = utils.filter_df(
        datasets_df, datasets=["philharmonia"])
    assert filtered_df["dataset"].unique() == ["philharmonia"]

    filtered_df = utils.filter_df(
        datasets_df, datasets=["philharmonia", "rwc"])
    assert set(filtered_df["dataset"].unique()) == \
           set(["philharmonia", "rwc"])


@pytest.fixture(scope="module", params=[
    (1, 1, 10, 100), (1, 2, 10),
    (1, 4, 10, 100, 5), (1, 8, 10),
    (1, 100, 10), (22, 1, 40, 10)])
def noise_shape(request):
    param = request.param
    return param


def test_backfill_noise(noise_shape):
    noise = np.random.random(noise_shape)
    for t_len in [1, 2, 3, 8, 10, 43, 100]:
        backfilled = utils.backfill_noise(noise, t_len)
        assert backfilled.shape[-2] >= t_len
        assert noise.shape[:-2] == backfilled.shape[:-2]

        if noise.shape[0] > 1:
            # Ideally we want to check that these
            # are not all equal, but it should suffice to check
            # that they're not equal with the first.
            for i in range(1, noise.shape[0]):
                assert not np.array_equal(backfilled[0], backfilled[i])
