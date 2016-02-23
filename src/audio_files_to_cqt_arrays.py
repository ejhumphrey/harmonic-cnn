#!/usr/bin/env python
"""Compute CQTs for a collection of audio files listed in a text file.

Calling this script consumes two files:

 1. textlist_file: A text file of newline separated filepaths to audio.
 2. cqt_params: A JSON-encoded text file of parameters for the CQT.

The former can be compiled by any means. The JSON encoded text can be created
by copy-pasting the following and writing the result to a file.

import json
params = {"q": 0.75,
          "freq_min": 27.5,
          "octaves": 8,
          "samplerate": 16000.0,
          "bins_per_octave": 24,
          "framerate": 20,
          "alignment": 'center',
          "channels": 1}
print json.dumps(params, indent=2)
fh = open("cqt_params.txt", "w")
json.dump(params, fh, indent=2)
fh.close()

This script writes the output files under the given output directory:

  "/some/audio/file.mp3" maps to "${output_dir}/file.npz"

Sample Call:
$ python audio_files_to_cqt_arrays.py \
rwc_filelist.txt \
cqt_arrays \
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
import numpy as np
import os
import time

import common.utils as utils

EXT = ".npz"
DEFAULT_PARAMS = dict(
    filepath=None, q=1.0, freq_min=27.5, octaves=7, bins_per_octave=36,
    samplerate=11025.0, channels=1, bytedepth=2, framerate=20.0,
    overlap=None, stride=None, time_points=None, alignment='center',
    offset=0)


def audio_file_to_cqt(input_file, output_directory):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    file_pair : Pair of strings
        The input_file (first) and output file (second) tuple.

    Returns
    -------
    output_file : str
        Newly created file.
    """
    output_file = os.path.join(output_directory,
                               "{}.npz".format(utils.filebase(input_file)))

    # TODO: This isn't quite correct.
    time_points, cqt_spectra = librosa.cqt(input_file, **DEFAULT_PARAMS)

    np.savez(output_file, time_points=time_points, cqt=cqt_spectra)
    print("[{0}] Finished: {1}".format(time.asctime(), output_file))
    return output_file


def cqt_many(audio_files, output_directory, cqt_params=None, num_cpus=-1):
    if cqt_params:
        DEFAULT_PARAMS.update(json.load(open(cqt_params)))

    utils.create_directory(output_directory)
    pool = Parallel(n_jobs=num_cpus)
    dcqt = delayed(audio_file_to_cqt)

    return pool(dcqt(af, output_directory) for af in audio_files)


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
    parser.add_argument("--num_cpus", type=int,
                        metavar="num_cpus", default=-1,
                        help="Number of CPUs over which to parallelize "
                             "computations.")

    args = parser.parse_args()
    with open(args.audio_files) as fp:
        audio_files = json.load(fp)

    # TODO: Catch result, write to JSON.
    cqt_many(audio_files, args.output_directory,
             args.cqt_params, args.num_cpus)
