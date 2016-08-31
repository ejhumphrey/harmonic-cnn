import os
import pytest
import shutil
import tempfile

import hcnn.common.labels
import hcnn.data.cqt
import hcnn.data.dataset as DS


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
    return hcnn.common.labels.InstrumentClassMap()


@pytest.fixture(scope="module")
def tinyds():
    return DS.TinyDataset.load()


@pytest.fixture(scope="module")
def tiny_feats(module_workspace, tinyds):
    limited_ds = tinyds.sample(12)
    return hcnn.data.cqt.cqt_from_dataset(
        limited_ds, module_workspace, skip_existing=True)


@pytest.fixture(scope="module")
def tiny_feats_csv(module_workspace, tiny_feats):
    file_path = os.path.join(module_workspace, "feats_index.csv")
    tiny_feats.save_csv(file_path)
    return file_path
