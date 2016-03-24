"""A class and utilities for managing the dataset and dataframes.
"""

import copy
import os
import pandas


class Dataset(object):
    """A class wrapper for loading the dataset
    from various inputs, and writing to various
    outputs, with utilities for providing
    views over datasets and generating train/val
    sets.

    A dataset contains the following columns
     - audio_file
     - target [normalized target names]
     - [some provenance information]
    """

    def __init__(self):
        pass

    @classmethod
    def read_csv(self, csv_path):
        pass

    @classmethod
    def read_pickle(self, pickle_path):
        pass

    def save_csv(self, csv_path):
        pass

    def save_pickle(self, pickle_path):
        pass

    @property
    def to_df(self):
        """Returns the dataset as a dataframe."""
        return self._df

    @property
    def to_dict(self):
        return self._to_dict

    def view(self, dataset):
        """Returns a copy of the analyzer pointing to the desired dataset."""
        thecopy = copy.copy(self)
        thecopy.set_test_set(dataset)
        return thecopy
