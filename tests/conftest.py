import os
import pytest
import shutil
import tempfile

import wcqtlib.common.labels


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
