""" Test parse.py functions.

(cbj) I'm not really sure how to test these without the files existing, but
for starters, I'm going to do it by creating dummy hierarchies
in the same format.
"""

import pytest

import wcqtlib.data.parse


def __test_pd_output(pd_output, working_dir, dataset):
    """Make sure all the files in the tree exist"""
    pass


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


def test_gen_directory_in():
    pass


def test_gen_files_in():
    pass


def test_rwc_to_dataframe():
    """Test that an input folder with files in rwc format is correctly
    converted to a dataframe."""
    file_root = "/Users/cjacoby/data/RWC Instruments"
    rwc_df = wcqtlib.data.parse.rwc_to_dataframe(file_root)
    yield __test_pd_output, rwc_df, file_root, "rwc"


# def test_uiowa_to_dataframe(example_uiowa_filetree):
#     """Test that an input folder with files in uiowa format is correctly
#     converted to a dataframe."""
#     uiowa_df = wcqtlib.data.parse(example_uiowa_filetree)
#     yield __test_pd_output, uiowa_df, example_uiowa_filetree, "uiowa"


# def test_philharmonia_to_dataframe(example_philharmonia_filetree):
#     """Test that an input folder with files in philharmonia format is correctly
#     converted to a dataframe."""
#     philharmonia_df = wcqtlib.data.parse(example_philharmonia_filetree)
#     yield __test_pd_output, philharmonia_df, \
#         example_philharmonia_filetree, "philharmonia"
