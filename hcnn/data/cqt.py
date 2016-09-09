#!/usr/bin/env python
"""Compute CQTs for a collection of audio files.

This script writes the output files under the given output directory:

  "/some/audio/file.mp3" maps to "${output_dir}/file.npz"

Sample Call:
$ python audio_to_cqts.py \
    filelist.json \
    ./cqt_arrays \
    --cqt_params=params.json \
    --num_cpus=2
"""
from __future__ import print_function

import argparse
import claudio
from joblib import delayed
from joblib import Parallel
import json
import librosa
import logging
import numpy as np
import os
import pandas as pd
import sys
import time

import hcnn.common.utils as utils
import hcnn.data.dataset as DS

logger = logging.getLogger(__name__)

CQT_PARAMS = dict(
    hop_length=512, fmin=16.35, n_bins=252, bins_per_octave=36, tuning=0.0,
    filter_scale=1, aggregate=None, norm=1, sparsity=0.0, real=False)

HARMONIC_PARAMS = dict(n_bins=CQT_PARAMS['n_bins'], n_harmonics=3,
                       fmin=CQT_PARAMS['fmin'],
                       bins_per_octave=CQT_PARAMS['bins_per_octave'])

AUDIO_PARAMS = dict(samplerate=22050.0, channels=1, bytedepth=2)


def harmonic_cqt(x_in, sr, hop_length=1024, fmin=27.5, n_bins=72,
                 n_harmonics=5, bins_per_octave=36, tuning=0.0, filter_scale=1,
                 aggregate=None, norm=1, sparsity=0.0, real=False):
    """Harmonically layered CQT.

    Thin wrapper around librosa.cqt; all parameters are the same, except for
    those listed below.

    Parameters
    ----------
    x_in : np.ndarray, ndim=2
        Input signal, with shape (time, channels)

    n_harmonics : int, default=5
        Number of harmonic layers for each fundamental frequency.

    Returns
    -------
    harmonic_spectra : np.ndarray, ndim=4
        Resulting spectra, with shape (channels, harmonics, time, frequency).
    """

    kwargs = dict(n_bins=n_bins, bins_per_octave=bins_per_octave,
                  hop_length=hop_length, sr=sr, tuning=tuning,
                  filter_scale=filter_scale, aggregate=aggregate, norm=norm,
                  sparsity=sparsity, real=real)

    cqt_spectra = []
    min_tdim = np.inf
    for i in range(1, n_harmonics + 1):
        cqt_spectra += [np.array([librosa.cqt(x_c, fmin=i * fmin, **kwargs).T
                                  for x_c in x_in.T])[:, np.newaxis, ...]]
        min_tdim = min([cqt_spectra[-1].shape[2], min_tdim])
    cqt_spectra = [x[:, :, :min_tdim, :] for x in cqt_spectra]

    return np.concatenate(cqt_spectra, axis=1)


def cqt_one(input_file, output_file, cqt_params=None, audio_params=None,
            harmonic_params=None, skip_existing=True):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    input_file : str
        Audio file to apply the CQT

    output_file : str
        Path to write the output.

    cqt_params : dict, default=None
        Parameters for the CQT function. See `librosa.cqt`.

    audio_params : dict, default=None
        Parameters for reading the audio file. See `claudio.read`.

    harmonic_params : dict, default=None
        Parameters for the `harmonic_cqt` function, which will update those in
        cqt_params.

    skip_existing : bool, default=True
        Skip outputs that exist.

    Returns
    -------
    success : bool
        True if the output file was successfully created.
    """
    input_exists, output_exists = [os.path.exists(f)
                                   for f in (input_file, output_file)]
    if not input_exists:
        logger.warning("[{0}] Input file doesn't exist, skipping: {1}"
                       "".format(time.asctime(), input_file))
        return input_exists

    if skip_existing and output_exists:
        logger.info("[{0}] Output file exists, skipping: {1}"
                    "".format(time.asctime(), output_file))
        return output_exists

    logger.debug("[{0}] Starting {1}".format(time.asctime(), input_file))
    if not cqt_params:
        cqt_params = CQT_PARAMS.copy()

    if not audio_params:
        audio_params = AUDIO_PARAMS.copy()

    if not harmonic_params:
        harmonic_params = HARMONIC_PARAMS.copy()

    logger.debug("[{0}] Audio conversion {1}".format(
        time.asctime(), input_file))
    try:
        x, fs = claudio.read(input_file, **audio_params)
        if len(x) <= 0:
            logger.error("Bad Input signal length={} for audio {}".format(
                len(x), input_file))
            return False
        logger.debug("[{0}] Computing features {1}".format(
            time.asctime(), input_file))
        cqt_spectra = np.array([np.abs(librosa.cqt(x_c, sr=fs, **cqt_params).T)
                                for x_c in x.T])

        cqt_params.update(**harmonic_params)
        harm_spectra = harmonic_cqt(x, fs, **cqt_params)

        frame_idx = np.arange(cqt_spectra.shape[1])
        time_points = librosa.frames_to_time(
            frame_idx, sr=fs, hop_length=cqt_params['hop_length'])
        logger.debug("[{0}] Saving: {1}".format(time.asctime(), output_file))
        np.savez(
            output_file, time_points=time_points,
            cqt=np.abs(cqt_spectra).astype(np.float32),
            harmonic_cqt=np.abs(harm_spectra).astype(np.float32))
    except AssertionError as e:
        logger.error("Failed to load audio file: {} with error:\n{}".format(
                     input_file, e))
    logger.debug("[{0}] Finished: {1}".format(time.asctime(), output_file))
    return os.path.exists(output_file)


def cqt_many(audio_files, output_files, cqt_params=None, audio_params=None,
             harmonic_params=None, num_cpus=-1, verbose=50,
             skip_existing=True):
    """Compute CQT representation over a number of audio files.

    Parameters
    ----------
    audio_files : list of str, len=n
        Audio files over which to compute the CQT.

    output_files : list of str, len=n
        File paths for writing outputs.

    cqt_params : dict, default=None
        Parameters to use for CQT computation.

    audio_params : dict, default=None
        Parameters to use for loading the audio file.

    harmonic_params : dict, default=None
        Parameters to use on top of `cqt_params` for the harmonic cqt.

    num_cpus : int, default=-1
        Number of parallel threads to use for computation.

    Returns
    -------
    failed_files : array of audio_files
        Array indicating which files failed to load.
    """
    pool = Parallel(n_jobs=num_cpus, verbose=50)
    dcqt = delayed(cqt_one)
    pairs = zip(audio_files, output_files)
    statuses = pool(dcqt(fin, fout, cqt_params, audio_params,
                         harmonic_params, skip_existing)
                    for fin, fout in pairs)
    return [audio_files[i] for i, x in enumerate(statuses) if not x]


def cqt_from_dataset(dataset, write_dir,
                     cqt_params=None, audio_params=None, harmonic_params=None,
                     num_cpus=-1, verbose=50, skip_existing=True):
    """Compute CQT representation over audio files referenced by
    a dataframe, and return a new dataframe also containing a column
    referencing the cqt files.

    Parameters
    ----------
    dataset : hcnn.data.Dataset
        Dataset containing references to the audio files.

    write_dir : str
        Directory to write to.

    cqt_params : dict, default=None
        Parameters to use for CQT computation.

    audio_params : dict, default=None
        Parameters to use for loading the audio file.

    harmonic_params : dict, default=None
        Parameters to use on top of `cqt_params` for the harmonic cqt.

    num_cpus : int, default=-1
        Number of parallel threads to use for computation.

    verbose : int
        Passed to cqt_many; for "Parallel"

    skip_existing : bool
        If files exist, don't try to extract them.

    Returns
    -------
    updated_dataset : data.dataset.Dataset
        Dataset updated with parameters to the outputed features.
    """
    utils.create_directory(write_dir)

    ####
    ## TODO IF skip_existing, try to reload the dataset with features
    ## And modify it instead of replacing it.

    def features_path_for_audio(audio_path):
        return os.path.join(write_dir,
                            utils.filebase(audio_path) + ".npz")

    audio_paths = dataset.to_df()["audio_file"].tolist()
    cqt_paths = [features_path_for_audio(x) for x in audio_paths]

    failed_files = cqt_many(audio_paths, cqt_paths, cqt_params, audio_params,
                            harmonic_params, num_cpus, verbose, skip_existing)
    logger.warning("{} files failed to extract.".format(len(failed_files)))

    feats_df = dataset.to_df()
    feats_df['cqt'] = pd.Series([None] * len(feats_df), index=feats_df.index)
    # Update the features field if the file was successfully created.
    for i, path in enumerate(cqt_paths):
        if os.path.exists(path):
            feats_df.loc[feats_df.index[i], "cqt"] = path
        else:
            logger.warning("CQT Not successfully created: {}".format(path))

    return DS.Dataset(feats_df, dataset.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("audio_files",
                        metavar="audio_files", type=str,
                        help="A JSON file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    parser.add_argument("--cqt_params",
                        metavar="cqt_params", type=str,
                        default='',
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--audio_params",
                        metavar="audio_params", type=str,
                        default='',
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--num_cpus", type=int,
                        metavar="num_cpus", default=-1,
                        help="Number of CPUs over which to parallelize "
                             "computations.")
    parser.add_argument("--verbose", type=int,
                        metavar="verbose", default=0,
                        help="Verbosity level for joblib.")
    parser.add_argument("--skip_existing", action='store_true',
                        help="If True, will skip existing files.")

    args = parser.parse_args()
    with open(args.audio_files) as fp:
        audio_files = json.load(fp)

    cqt_params = None
    audio_params = None

    cqt_params = json.load(open(args.cqt_params)) if args.cqt_params else None
    audio_params = json.load(open(args.audio_params)) \
        if args.audio_params else None

    output_files = [utils.map_io(fin, args.output_directory)
                    for fin in audio_files]
    success = cqt_many(audio_files, output_files,
                       cqt_params=cqt_params, audio_params=audio_params,
                       num_cpus=args.num_cpus, verbose=args.verbose,
                       skip_existing=args.skip_existing)
    sys.exit(0 if success else 1)
