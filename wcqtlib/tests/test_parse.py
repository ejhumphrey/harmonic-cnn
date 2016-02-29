""" Test parse.py functions.

(cbj) I'm not really sure how to test these without the files existing, but
for starters, I'm going to do it by creating dummy hierarchies
in the same format.
"""

import os
import pytest

import wcqtlib.data.parse


def __test_df_has_data(df):
    assert not df.empty

def __test_pd_output(pd_output, working_dir, dataset):
    """Make sure all the files in the tree exist"""
    # Check for valid columns
    required_columns = ['audio_file', 'dataset', 'index', 'instrument']
    for column in required_columns:
        assert column in pd_output.columns

    # Check files and per row things.
    for row in pd_output.iterrows():
        assert os.path.exists(row[1]['audio_file'])
        assert row[1]['dataset'] == dataset


@pytest.fixture
def example_rwc_filetree(workspace):
    return "/Users/cjacoby/data/RWC Instruments"
    # return workspace


@pytest.fixture
def example_uiowa_filetree(workspace):
    return "/Users/cjacoby/data/uiowa"
    # return workspace


@pytest.fixture
def example_philharmonia_filetree(workspace):
    return "/Users/cjacoby/data/philharmonia"
    # return workspace


def test_unzip():
    zip_files = [os.path.join(os.path.dirname(__file__), "zipped_folder.zip")]
    unzipped_folders = wcqtlib.data.parse.unzip_files(zip_files)
    for dir_path in unzipped_folders:
        assert os.path.exists(dir_path) and os.path.isdir(dir_path)


def test_rwc_to_dataframe():
    """Test that an input folder with files in rwc format is correctly
    converted to a dataframe."""
    # Todo... don't do this.
    file_root = "/Users/cjacoby/data/RWC Instruments"
    rwc_df = wcqtlib.data.parse.rwc_to_dataframe(file_root)
    yield __test_df_has_data, rwc_df
    yield __test_pd_output, rwc_df, file_root, "rwc"


def test_uiowa_to_dataframe():
    """Test that an input folder with files in uiowa format is correctly
    converted to a dataframe."""
    # Todo... don't do this.
    file_root = "/Users/cjacoby/data/uiowa"
    uiowa_df = wcqtlib.data.parse.uiowa_to_dataframe(file_root)
    yield __test_df_has_data, uiowa_df
    yield __test_pd_output, uiowa_df, example_uiowa_filetree, "uiowa"


def test_philharmonia_to_dataframe():
    """Test that an input folder with files in philharmonia format is correctly
    converted to a dataframe."""
    file_root = "/Users/cjacoby/data/philharmonia"
    philharmonia_df = wcqtlib.data.parse.philharmonia_to_dataframe(file_root)
    yield __test_df_has_data, philharmonia_df
    yield __test_pd_output, philharmonia_df, file_root, "philharmonia"
