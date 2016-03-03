import claudio
import librosa
import os
import pytest
import shutil
import numpy as np

import wcqtlib.data.extract as extract

THIS_PATH = os.path.dirname(__file__)


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
    output_files = extract.split_and_standardize_examples(testfile, workspace)
    assert all(map(output_files, os.path.exists))


def test_standardize_one_onset(testfile):
    padding = 0.1
    result = extract.standardize_one(
        testfile, first_onset_start=padding)
    assert result

    audio, sr = claudio.read(testfile)

    # Find what librosa thinks is the first onset
    onset_samples = extract.get_onsets(audio, sr)
    padding_samples = padding * sr
    assert np.testing.assert_almost_equal(onset_samples[0],
                                          padding_samples, decimal=4)

    # Check that the file up to this point
    # is mostly zero.
    starting_mean = audio[:intended_onset_frame].mean()
    np.testing.assert_almost_equal(starting_mean, 0.0, decimal=3)


def test_standardize_one_centroid(testfile):
    with pytest.raises(NotImplementedError):
        result = extract.standardize_one(testfile,
                                         center_of_mass_alignment=True)
        assert result


def test_standardize_one_final_duration(testfile):
    duration = 1.0
    result = extract.standardize_one(
        testfile, first_onset_start=None,
        final_duration=duration)
    assert result

    audio, sr = claudio.read(testfile)
    assert len(audio) == duration * sr
