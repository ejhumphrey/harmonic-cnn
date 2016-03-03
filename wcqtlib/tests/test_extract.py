import claudio
import os
import pandas
import pytest
import shutil
import numpy as np

import wcqtlib.data.parse
import wcqtlib.data.extract

THIS_PATH = os.path.dirname(__file__)


DATA_ROOT = os.path.expanduser("~/data")
RWC_ROOT = os.path.join(DATA_ROOT, "RWC Instruments")
UIOWA_ROOT = os.path.join(DATA_ROOT, "uiowa")
PHIL_ROOT = os.path.join(DATA_ROOT, "philharmonia")


def __test_df_has_data(df):
    assert not df.empty


@pytest.fixture
def testfile(workspace):
    """Copies the "mandolin_trem.mp3" file to
    the workspace so we can mess with it,
    and returns the new path."""
    mando_fn = "mandolin_trem.mp3"
    test_mando = os.path.join(THIS_PATH, mando_fn)
    output_file = os.path.join(workspace, mando_fn)

    shutil.copy(test_mando, output_file)
    return output_file


def test_split_and_standardize_examples(testfile, workspace):
    output_files = wcqtlib.data.extract.split_and_standardize_examples(testfile, workspace)
    assert all(map(os.path.exists, output_files))


def test_standardize_one_onset(testfile):
    padding = 0.1
    result = wcqtlib.data.extract.standardize_one(
        testfile, first_onset_start=padding)
    assert result

    audio, sr = claudio.read(testfile)

    # Find what librosa thinks is the first onset
    onset_samples = wcqtlib.data.extract.get_onsets(audio, sr)
    padding_samples = padding * sr
    assert np.testing.assert_almost_equal(onset_samples[0],
                                          padding_samples, decimal=4)

    # Check that the file up to this point
    # is mostly zero.
    starting_mean = audio[:intended_onset_frame].mean()
    np.testing.assert_almost_equal(starting_mean, 0.0, decimal=3)


def test_standardize_one_centroid(testfile):
    with pytest.raises(NotImplementedError):
        result = wcqtlib.data.extract.standardize_one(
            testfile, center_of_mass_alignment=True)
        assert result


def test_standardize_one_final_duration(testfile):
    duration = 1.0
    result = wcqtlib.data.extract.standardize_one(
        testfile, first_onset_start=None,
        final_duration=duration)
    assert result

    audio, sr = claudio.read(testfile)
    assert len(audio) <= duration * sr


def __test_rwc_notes(datasets_df, notes_df):
    for (index, row) in \
            datasets_df[datasets_df["dataset"] == "rwc"].iterrows():

        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) > 1


def __test_uiowa_notes(datasets_df, notes_df):
    for (index, row) in \
            datasets_df[datasets_df["dataset"] == "uiowa"].iterrows():

        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty

        # TODO: not successfully splitting UIOWA files :(
        #  Might need to change the threshold for silence.
        if len(row['note']) > 2:
            assert len(matching_notes) > 1


def __test_phil_notes(datasets_df, notes_df):
    for (index, row) in \
            datasets_df[datasets_df["dataset"] == "philharmonia"].iterrows():

        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) == 1


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(PHIL_ROOT),
                             os.path.exists(UIOWA_ROOT),
                             os.path.exists(RWC_ROOT)]),
                    reason="Data not found.")
@pytest.mark.slowtest
def test_datasets_to_notes():
    """Requires real and existing data."""
    # First, get the datasets_df with all the original files in it
    datasets_df = wcqtlib.data.parse.load_dataframes(DATA_ROOT)
    yield __test_df_has_data, datasets_df
    datasets = datasets_df["dataset"].unique()
    # Grab a small selection of each dataset so
    #  this test doesn't take literally all day.
    small_datasets_df = pandas.concat([
        datasets_df[datasets_df["dataset"] == x][5:8]
        for x in datasets])
    yield __test_df_has_data, small_datasets_df

    # Now, do the conversion into the notes_df
    test_extract_path = os.path.expanduser("~/data/ismir2016-wcqt-data")
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        small_datasets_df, test_extract_path)
    yield __test_df_has_data, notes_df

    # We'll test this by picking known files from each
    # dataset and making sure they were extracted as
    # we expect.
    yield __test_rwc_notes, small_datasets_df, notes_df
    yield __test_uiowa_notes, small_datasets_df, notes_df
    yield __test_phil_notes, small_datasets_df, notes_df
