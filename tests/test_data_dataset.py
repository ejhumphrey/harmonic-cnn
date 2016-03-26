import os
import pandas
import pytest

import wcqtlib.common.config as C
import wcqtlib.data.dataset as dataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           os.pardir, "data", "master_config.yaml")


@pytest.fixture(scope="module")
def config():
    return C.Config.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="module")
def test_obs():
    obs = [
        dict(index="abc123", dataset="rwc", audio_file="foo.aiff",
             instrument="horn, french", source_key="001", start_time=None,
             duration=None, note_number="A4", dynamic="pp", partition=None),
        dict(index="abc234", dataset="uiowa", audio_file="foo.aiff",
             instrument="horn, french", source_key="001", start_time=None,
             duration=None, note_number="A4", dynamic="pp", partition=None),
        dict(index="def123", dataset="philharmonia", audio_file="foo.aiff",
             instrument="horn, french", source_key="001", start_time=None,
             duration=None, note_number="A4", dynamic="pp", partition=None)
    ]
    return obs


def test_get_remote_schema():
    remote_schema = dataset.get_remote_schema()
    assert isinstance(remote_schema, dict)


def test_observation(test_obs):
    ob = dataset.Observation(**test_obs[0])
    assert ob is not None
    assert isinstance(ob, dataset.Observation)
    assert isinstance(ob.to_dict(), dict)


def test_json(workspace, test_obs):
    path = os.path.join(workspace, "test.json")
    ds = dataset.Dataset(test_obs)
    ds.save_json(path)
    assert os.path.exists(path)

    newds = dataset.Dataset.read_json(path)
    assert newds is not None
    assert isinstance(newds, dataset.Dataset)
    assert all([isinstance(x, dict) for x in newds.to_builtin()])


def test_to_df(test_obs):
    ds = dataset.Dataset(test_obs)
    df = ds.to_df()
    assert isinstance(df, pandas.DataFrame)
    assert len(test_obs) == len(df)
    assert set(df.columns) == set(ds.items[0].to_dict().keys())


def test_dataset_view(test_obs):
    ds = dataset.Dataset(test_obs)
    rwc_view = ds.view("rwc")
    assert set(rwc_view["dataset"].unique()) == set(["rwc"])


@pytest.mark.xfail(reason="Data is wrong; TODO")
def test_build_tiny_dataset_from_old_dataframe(config):
    tinyds = dataset.build_tiny_dataset_from_old_dataframe(config)

    assert isinstance(tinyds, list)
    assert len(tinyds) == 36
    assert all([obs.validate() for obs in tinyds])

    # See if we can load it into a dataset class.
    ds = dataset.Dataset(tinyds)
    assert ds.validate()
