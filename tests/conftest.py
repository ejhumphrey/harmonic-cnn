import os
import pytest
import shutil
import tempfile

import wcqtlib.common.labels
import wcqtlib.data.cqt
import wcqtlib.data.dataset


@pytest.fixture()
def workspace(request):
    test_workspace = tempfile.mkdtemp()

    def fin():
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)

    return test_workspace


@pytest.fixture(scope="module")
def module_workspace(request):
    test_workspace = tempfile.mkdtemp()

    def fin():
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)

    request.addfinalizer(fin)

    return test_workspace


@pytest.fixture
def classmap():
    return wcqtlib.common.labels.InstrumentClassMap()


@pytest.fixture(scope="module")
def tinyds():
    return wcqtlib.data.dataset.TinyDataset.load()


@pytest.fixture(scope="module")
def tiny_feats(module_workspace, tinyds):
    limited_ds = wcqtlib.data.dataset.Dataset(tinyds.observations[0:2])
    return wcqtlib.data.cqt.cqt_from_dataset(
        limited_ds, module_workspace, skip_existing=True)
