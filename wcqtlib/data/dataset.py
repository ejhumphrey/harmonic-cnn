"""A class and utilities for managing the dataset and dataframes.
"""

import copy
import json
import jsonschema
import os
import pandas
import requests


SCHEMA_PATH = "https://raw.githubusercontent.com/ejhumphrey/minst-dataset/" \
              "master/minst/schema/observation.json"


def get_remote_schema(url=SCHEMA_PATH):
    return requests.get(url).json()


class Observation(object):
    """Document model each item in the collection.
    TODO: Inherit / whatever from minst-dataset repo
    """
    SCHEMA = json.load(open(SCHEMA_PATH))

    def __init__(self, index, dataset, audio_file, instrument, source_key,
                 start_time, duration, note_number, dynamic, partition,
                 features=None):
        self.index = index
        self.dataset = dataset
        self.audio_file = audio_file
        self.instrument = instrument
        self.source_key = source_key
        self.start_time = start_time
        self.duration = duration
        self.note_number = note_number
        self.dynamic = dynamic
        self.partition = partition
        self.features = features if features else dict()

    def to_dict(self):
        return self.__dict__.copy()

    def validate(self, schema=None):
        schema = self.SCHEMA if schema is None else schema
        success = True
        try:
            jsonschema.validate(self.to_dict(), schema)
        except jsonschema.ValidationError:
            success = False
        success &= os.path.exists(self.audio_file)
        # if success:
        #     success &= utils.check_audio_file(self.audio_file)[0]

        return success


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
    def read_json(self, csv_path):
        pass

    @classmethod
    def read_pickle(self, pickle_path):
        pass

    def save_json(self, csv_path):
        pass

    def save_pickle(self, pickle_path):
        pass

    @property
    def to_df(self):
        """Returns the dataset as a dataframe."""
        return self._df

    @property
    def to_builtin(self):
        return self._to_dict

    def view(self, dataset):
        """Returns a copy of the analyzer pointing to the desired dataset."""
        thecopy = copy.copy(self)
        thecopy.set_test_set(dataset)
        return thecopy


def build_tiny_dataset_from_old_dataframe(config):
    notes_df_path = os.path.join(
        os.path.expanduser(config['paths/data_dir']),
        config['dataframes/notes'])
    notes_df = pandas.read_pickle(notes_df_path)

    tiny_dataset = []
    # Get one file for each instrument for each dataset.
    for dataset in notes_df["dataset"].unique():
        for instrument in notes_df["instrument"].unique():
            record = notes_df.loc[(notes_df["dataset"] == dataset) &
                                  (notes_df["instrument"] == instrument)][0]
            tiny_dataset.append(
                Observation(
                    index=record.index,
                    dataset=record['dataset'],
                    audio_file=record['audio_file'],
                    instrument=record['instrument'],
                    source_key=None,
                    start_time=None,
                    duration=None,
                    note_number=None,
                    dynamic=None,
                    partition=None
                    ))
    return tiny_dataset
