"""
Generators/Streams for generating data.
"""

import numpy as np

import wcqtlib.data.parse as parse
import wcqtlib.common.utils as utils

instrument_map = parse.InstrumentClassMap()


def cqt_slices(record, t_len):
    """Generate slices from a cqt file.
    Designed to be used in a Pescador stream.

    Thresholds the frames by finding the cqt means, and
    only returning frames with means greater than the lowest
    quarter of the means.

    Parameters
    ----------
    record : pandas.Series
        Single pandas record containing a 'cqt' record
        which points to the cqt file in question.
        Also must contain an "instrument" column
        which contains the ground truth.

    length : int
        Length of the sliced array

    idx : int, or None
        Centered frame index for the slice, or random if not provided.

    Yields
    -------
    sample : dict with fields {cqt, label}
        The windowed observation.
    """
    # Load the npz
    cqt = np.load(record['cqt'])['cqt']
    target = instrument_map.get_index(record["instrument"])

    num_obs = cqt.shape[1] - t_len

    # Get the frame means, and remove the lowest 1/4
    cqt_mu = cqt[0, :num_obs].mean(axis=1)
    threshold = sorted(cqt_mu)[int(len(cqt_mu)*.25)]
    idx = (cqt_mu[:num_obs] > threshold).nonzero()[0]
    np.random.shuffle(idx)

    counter = 0
    while True:
        obs = utils.slice_ndarray(cqt, idx[counter], length=t_len, axis=1)
        data = dict(
            x_in=obs[np.newaxis, ...],
            target=target)
        yield data

        # Once we have used all of the frames once, reshuffle.
        counter += 1
        if counter >= len(idx):
            np.random.shuffle(idx)
            counter = 0


def wcqt_slices(record, t_len, p_len=48, p_stride=36):
    """Generate slices of wrapped CQT observations.

    Parameters
    ----------
    stash : biggie.Stash
        Stash to draw the datapoint from.
    key : str
        Key of the entity to slice.
    t_len : int
        Length of the CQT slice in time.
    p_len : int
        Number of adjacent pitch bins.
    p_stride : int
        Number of pitch bins to stride when wrapping.
    Yields
    ------
    sample : dict with fields {cqt, label}
        The windowed observation.
    """
    pass


def buffer_streams(stream, batch_size=50):
    """Buffer stream into ndarrays.

    Parameters
    ----------
    stream : generator

    batch_size : int

    Yields
    ------
    batch : dict
        Dense ndarrays.
    """
    pass
