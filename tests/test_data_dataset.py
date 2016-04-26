import numpy as np
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
    assert isinstance(ob.to_series(), pandas.Series)


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
    safe_columns = set(ds.items[0].to_dict().keys())
    safe_columns = safe_columns - set(["features"])
    assert set(df.columns) == safe_columns


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


def test_construct_training_valid_df():
    def __test_result_df(traindf, validdf, datasets, n_per_inst=None):
        if n_per_inst:
            assert len(traindf) == 12 * n_per_inst
            assert len(validdf)
        else:
            total_len = len(traindf) + len(validdf)
            np.testing.assert_almost_equal((total_len*.8)/100.,
                                           len(traindf)/100.,
                                           decimal=0)
            np.testing.assert_almost_equal((total_len*.2)/100.,
                                           len(validdf)/100.,
                                           decimal=0)
        assert set(datasets) == set(traindf['dataset'])
        assert set(datasets) == set(validdf['dataset'])

    tinyds = dataset.TinyDataset.load()

    test_set = "rwc"
    # on the tinyds, we hope to have one of each in each
    # train and validate, because there are only two to begin with.
    train_df, valid_df = tinyds.get_train_val_split(
        test_set=test_set, train_val_split=1)
    # TODO: Come up with a test here.
    # yield __test_result_df, train_df, valid_df, datasets

    test_set = "philharmonia"
    train_df, valid_df = tinyds.get_train_val_split(
        test_set=test_set, train_val_split=1)
    # TODO: Come up with a test here.

    # TODO: test that some bigger data works.
