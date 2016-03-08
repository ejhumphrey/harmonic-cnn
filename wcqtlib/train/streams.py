"""
Generators/Streams for generating data.
"""

import numpy as np
import pescador

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


class InstrumentStreamer(object):
    """Class wrapping the creation of a pescador streamer
    to sample equally from each instrument class available in
    the features dataframe."""

    def __init__():
        pass


def instrument_streams(features_df, instrument, hold_out_dataset=[],
                       t_len=1):
    """Return a list of generators for all records in the dataframe
    which match the instrument given, but are not in the hold_out_dataset.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Dataframe which points to the features files and has ground truth data.

    instrument : str
        Instrument to select.

    hold_out_dataset : list of str
        Dataset(s) to exclude from searching in.

    t_len : int
        Number of frames in the time dimension to return for each sample.

    Returns
    -------
    streams : list of pescador.Streamer
        One streamer for each instrument file.
    """
    pass


def instrument_mux(features_df, instrument, hold_out_dataset=[],
                   t_len=1, k=10, lam=20, **kwargs):
    """Return a pescador.mux for a single instrument.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Dataframe which points to the features files and has ground truth data.

    instrument : str
        Instrument to select.

    hold_out_dataset : list of str
        Dataset(s) to exclude from searching in.

    t_len : int
        Number of frames in the time dimension to return for each sample.

    k : int
    lam : int
    **kwargs : dict
        See pescador.Mux. kwargs get passed to the mux.

    Returns
    -------
    mux : pescador.mux
        A pescador.mux for a single instrument.
    """
    pass


def all_instrument_streams(features_df, hold_out_dataset,
                           t_len=1, k=10, lam=20, **kwargs):
    """Return a list of pescador.Streamers for each instrument,
    where each Streamer is wrapping a pescador.Mux sampling
    every instrument file.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Dataframe which points to the features files and has ground truth data.

    hold_out_dataset : list of str
        Dataset(s) to exclude from searching in.

    t_len : int
        Number of frames in the time dimension to return for each sample.

    k : int
    lam : int
    **kwargs : dict
        See pescador.Mux. kwargs get passed to the mux.

    Returns
    -------
    streamers : list of pescador.Steamer
        One pescador.Streamer for each instrument type in the dataframe.
    """
    pass


def buffer_stream(stream, batch_size=50):
    """Buffer stream into ndarrays.

    TODO: Try to make this return samples equally from
        different classes.

    Parameters
    ----------
    stream : generator
        Stream of similar pairs, yielding dicts like
            {'x_in', 'target'}

    batch_size : int
        Number of similar pairs to buffer.

    Yields
    ------
    batch : dict
        Dense ndarrays.
    """
    pass


def zmq_buffered_stream(stream, batch_size=50):
    """Buffer stream using zmq in a separate python process.

    Parameters
    ----------
    stream : generator

    batch_size : int

    Yields
    ------
    batch : dict
        Dense ndarrays
    """
    pass
