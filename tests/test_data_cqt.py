import claudio
import os
import pytest
import numpy as np

import wcqtlib.data.cqt as CQT

DIRNAME = os.path.dirname(__file__)


def test_harmonic_cqt(workspace):
    input_file = os.path.join(DIRNAME, "sax_cres.mp3")
    x, fs = claudio.read(input_file, samplerate=22050, channels=1)
    spec = CQT.harmonic_cqt(x, fs, n_harmonics=6, n_bins=144,
                            bins_per_octave=24)
    assert spec.ndim == 4
    assert np.abs(spec).sum() > 0


def test_harmonic_cqt_uneven_length(workspace):
    input_file = os.path.join(DIRNAME, "uneven_hcqt.flac")
    x, fs = claudio.read(input_file, samplerate=22050, channels=1)
    spec = CQT.harmonic_cqt(x, fs, n_harmonics=6, n_bins=144,
                            bins_per_octave=24)
    assert spec.ndim == 4
    assert np.abs(spec).sum() > 0


def test_cqt_one(workspace):
    input_file = os.path.join(DIRNAME, "sax_cres.mp3")
    output_file = os.path.join(workspace, "foo.npz")
    assert CQT.cqt_one(input_file, output_file)

    features = np.load(output_file)
    for key in 'cqt', 'harmonic_cqt', 'time_points':
        assert key in features


def test_download_many(workspace):
    input_files = [os.path.join(DIRNAME, fname)
                   for fname in ("sax_cres.mp3", "mandolin_trem.mp3")]
    output_files = [os.path.join(workspace, fname)
                    for fname in ('foo.npz', 'bar.npz')]
    assert CQT.cqt_many(input_files, output_files)
