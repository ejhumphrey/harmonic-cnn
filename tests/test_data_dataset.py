import os
import pytest

import wcqtlib.common.config as C
import wcqtlib.data.dataset as dataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           os.pardir, "data", "master_config.yaml")


@pytest.fixture(scope="module")
def config():
    return C.Config.from_yaml(CONFIG_PATH)


def test_get_remote_schema():
    pass


def test_observation():
    pass


def test_json():
    pass


def test_to_df():
    pass


def test_to_builtin():
    pass


def test_dataset_view():
    pass


def build_tiny_dataset_from_old_dataframe(config):
    tinyds = dataset.build_tiny_dataset_from_old_dataframe(config)

    assert isinstance(tinyds, list)
    assert len(tinyds) == 36

    schema = dataset.get_remote_schema()
    assert all([obs.validate(schema) for obs in tinyds])
