"""
Generators/Streams for generating data.
"""

import collections
import numpy as np
import pescador

import wcqtlib.data.extract as extract
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


def buffer_stream(stream, batch_size):
    """Buffer stream into ndarrays.

    Parameters
    ----------
    stream : pescador.Stream

    batch_size : int
        Number of elements to return.

    Yields
    ------
    batch : dict
        Dense ndarrays.
    """
    return pescador.buffer_streamer(stream, batch_size)


def zmq_buffered_stream(stream, batch_size):
    """Buffer stream using zmq in a separate python process.

    Parameters
    ----------
    stream : pescador.Stream

    batch_size : int
        Number of elements to return.

    Yields
    ------
    batch : dict
        Dense ndarrays
    """
    return pescador.zmq_stream(stream, max_batches=batch_size)


class InstrumentStreamer(collections.Iterator):
    """Class wrapping the creation of a pescador streamer
    to sample equally from each instrument class available in
    the features dataframe.

    This class will behave like a generator."""

    def __init__(self, features_df, datasets,
                 record_slicer,
                 slicer_kwargs={},
                 t_len=1,
                 instrument_mux_params=dict(k=10, lam=20),
                 master_mux_params=dict(
                    n_samples=None,  # no maximum number
                    k=12,  # this is the number of classes.
                    lam=None,  # only use these 12 streams.
                    with_replacement=False,  # really, only use those 12
                    revive=True  # make sure only one copy of each is active
                                 # at a time.
                    ),
                 batch_size=50,
                 use_zmq=False):
        """
        Parameters
        ----------
        features_df : pandas.DataFrame
            Dataframe which points to the features files and has ground
            truth data.

        datasets : list of str
            Dataset(s) to use in the features_df.

        record_slicer : function
            Function used to load slices from a df record.
            (See {cqt_slices, wcqt} slices above).

        slicer_kwargs : dict
            Any parameters to send to the slider.
            t_len will always get fowarded to the slicer.

        t_len : int
            Number of frames in the time dimension to return for each sample.

        instrument_mux_params : dict
            Arguments to pass to the pescador.mux's which sample from
            each instrument class.

        master_mux_params : dict
            Arguments to pass to the pescador.mux's which sample from
            all of the instrument mux streams.

        batch_size : int
            Size of the batch to return from the buffered generator stream.

        use_zmq : bool
            If true, use a zmq_stream as the final sampler, in order
            to generate the data in parallel.
        """
        self.features_df = features_df
        self.datasets = datasets
        self.record_slicer = record_slicer
        self.slicer_kwargs = slicer_kwargs
        self.t_len = t_len
        self.instrument_mux_params = instrument_mux_params
        self.master_mux_params = master_mux_params
        self.batch_size = batch_size
        self.use_zmq = use_zmq

        self.setup()

    def setup(self):
        """Perform the setup to prepare for streaming."""
        # Instrument names
        instruments = list(self.features_df["instrument"].unique())

        # Get Muxes for each instrument.
        inst_muxes = [self._instrument_mux(i) for i in instruments]

        # Construct the streams for each mux.
        mux_streams = [pescador.Streamer(x) for x in inst_muxes]

        # Construct the master mux
        self.master_mux = pescador.mux(mux_streams, **self.master_mux_params)

        # Now construct the final streamer
        if self.use_zmq:
            self.buffered_streamer = zmq_buffered_stream(
                self.master_mux, self.batch_size)
        else:
            self.buffered_streamer = buffer_stream(
                self.master_mux, self.batch_size)

    def _instrument_streams(self, instrument):
        """Return a list of generators for all records in the dataframe
        which match the instrument given, and are in the datasets.

        Parameters
        ----------
        instrument : str
            Instrument to select.

        Returns
        -------
        streams : list of pescador.Streamer
            One streamer for each instrument file.
        """
        # Get list of instruments
        instrument_records = extract.filter_df(
            self.features_df, instrument=instrument, datasets=self.datasets)
        seed_pool = [pescador.Streamer(self.record_slicer, record, self.t_len,
                                       **self.slicer_kwargs)
                     for record in instrument_records.iterrows()]
        return seed_pool

    def _instrument_mux(self, instrument):
        """Return a pescador.mux for a single instrument.

        Parameters
        ----------
        instrument : str
            Instrument to select.

        Returns
        -------
        mux : pescador.mux
            A pescador.mux for a single instrument.
        """
        streams = self._instrument_streams(instrument)
        return pescador.mux(streams, n_samples=None,
                            **self.instrument_mux_params)

    def __iter__(self):
        yield self.buffered_streamer

    def __next__(self):
        """Generate batches of samples and return them."""
        return next(self.buffered_streamer)
