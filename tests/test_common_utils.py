import os
import pytest

import wcqtlib.utils as utils


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
