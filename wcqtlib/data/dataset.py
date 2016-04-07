"""A class and utilities for managing the dataset and dataframes.
"""

import copy
import json
import jsonschema
import logging
import os
import pandas
import requests
from sklearn.cross_validation import train_test_split
import sys


import wcqtlib.common.config as C
import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)

SCHEMA_PATH = "https://raw.githubusercontent.com/ejhumphrey/minst-dataset/" \
              "master/minst/schema/observation.json"


def get_remote_schema(url=SCHEMA_PATH):
    try:
        return requests.get(url).json()
    except requests.exceptions.ConnectionError:
        logger.error("No internet connection - cannot load remote schema.")
        return {}


class Observation(object):
    """Document model each item in the collection.
    TODO: Inherit / whatever from minst-dataset repo
    """
    SCHEMA = get_remote_schema()

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

    @classmethod
    def from_record(cls, record):
        return cls(index=record.index[0][0],
                   dataset=record['dataset'][0],
                   audio_file=record['audio_file'][0],
                   instrument=record['instrument'][0],
                   source_key="",
                   start_time=0.0,
                   duration=0.0,
                   note_number=record['note'][0],
                   dynamic=record['dynamic'][0],
                   partition="")

    def to_dict(self):
        return self.__dict__.copy()

    def __getitem__(self, key):
        return self.__dict__[key]

    def to_series(self):
        """Convert to a flat series (ie make features a column)

        Returns
        -------
        pandas.Series
        """
        flat_dict = self.to_dict()
        flat_dict.update(**flat_dict.pop("features"))
        return pandas.Series(flat_dict)

    def validate(self, schema=None):
        schema = self.SCHEMA if schema is None else schema
        success = True
        try:
            jsonschema.validate(self.to_dict(), schema)
        except jsonschema.ValidationError:
            success = False
        success &= os.path.exists(self.audio_file)
        if success:
            success &= utils.check_audio_file(self.audio_file,
                                              min_duration=.1)[0]

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

    def __init__(self, observations, data_root=None):
        """
        Parameters
        ----------
        observations : list
            List of Observations (as dicts or Observations.)
            If they're dicts, this will convert them to Observations.

        data_root : str or None
            Path to look for an observation, if not None
        """
        def safe_obs(obs, data_root=None):
            "Get dict from an Observation if an observation, else just dict"
            if not os.path.exists(obs['audio_file']) and data_root:
                new_audio = os.path.join(data_root, obs['audio_file'])
                if os.path.exists(new_audio):
                    obs['audio_file'] = new_audio
            if isinstance(obs, Observation):
                return obs.to_dict()
            else:
                return obs
        self.observations = [Observation(**safe_obs(x, data_root))
                             for x in observations]

    @classmethod
    def read_json(cls, json_path, data_root=None):
        if os.path.exists(json_path):
            with open(json_path, 'r') as fh:
                return cls(json.load(fh), data_root=data_root)
        else:
            logger.error("No dataset available at {}".format(json_path))
            return None

    def save_json(self, json_path):
        with open(json_path, 'w') as fh:
            json.dump(self.to_builtin(), fh)

    def to_df(self):
        """Returns the dataset as a dataframe."""
        return pandas.DataFrame([x.to_series() for x in self.observations])

    def to_builtin(self):
        return [x.to_dict() for x in self.observations]

    @property
    def items(self):
        return self.observations

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return self.observations[index]

    def validate(self):
        if len(self.observations > 0):
            return all([x.validate for x in self.observations])
        else:
            logger.warning("No observations to validate.")
            return False

    def copy(self, deep=True):
        return Dataset(copy.deepcopy(self.observations))

    def view(self, dataset_filter):
        """Returns a copy of the analyzer pointing to the desired dataset.
        Parameters
        ----------
        dataset_filter : str
            String in ["rwc", "uiowa", "philharmonia"] which is
            the items in the dataset to return.

        Returns
        -------
        """
        thecopy = copy.copy(self.to_df())
        ds_view = thecopy[thecopy["dataset"] == dataset_filter]
        return ds_view

    def get_train_val_split(self, test_set, train_val_split=0.2,
                            max_files_per_class=None):
        """Returns Datasets for train and validation constructed
        from the datasets not in the test_set, and split with
        the ratio train_val_split.

         * First selects from only the datasets given in datasets.
         * Then **for each instrument** (so the distribution from
             each instrument doesn't change)
            * train_test_split to generate training and validation sets.
            * if max_files_per_class, also then restrict the training set to
                a maximum of that number of files for each train and test

        Parameters
        ----------
        test_set : str
            String in ["rwc", "uiowa", "philharmonia"] which selects
            the hold-out-set to be used for testing.

        Returns
        -------
        train_df, valid_df : pandas.DataFrame
            DataFrames referencing the files for train and validation.
        """
        df = self.to_df()
        datasets = set(df["dataset"].unique()) - set([test_set])
        search_df = df[df["dataset"].isin(datasets)]

        selected_instruments_train = []
        selected_instruments_valid = []
        for instrument in search_df["instrument"].unique():
            instrument_df = search_df[search_df["instrument"] == instrument]

            if len(instrument_df) < 2:
                logger.warning("Instrument {} doesn't haven enough samples "
                               "to split.".format(instrument))
                continue

            traindf, validdf = train_test_split(
                instrument_df, test_size=train_val_split)

            if max_files_per_class:
                traindf = traindf.sample(n=max_files_per_class)

            selected_instruments_train.append(traindf)
            selected_instruments_valid.append(validdf)

        return pandas.concat(selected_instruments_train), \
            pandas.concat(selected_instruments_valid)


class TinyDataset(Dataset):
    ROOT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
        os.pardir, os.pardir, "tests", "tinydata"))
    DS_FILE = os.path.join(ROOT_PATH, "tinydata.json")

    @classmethod
    def load(cls, json_path=DS_FILE):
        with open(json_path, 'r') as fh:
            data = json.load(fh)
            # Update the paths to full paths.
            for item in data:
                item['audio_file'] = os.path.join(cls.ROOT_PATH,
                                                  item['audio_file'])
        return cls(data)


def build_tiny_dataset_from_old_dataframe(config):
    def sample_record(df, dataset, instrument):
        query_records = df.loc[(df["dataset"] == dataset) &
                               (df["instrument"] == instrument)]
        logger.info("[{} - {} - {} records]".format(
            dataset, instrument, len(query_records)))

        if not query_records.empty:
            is_valid = False
            count_limit = 5  # to prevent infinite loops
            i = 0
            while not is_valid:
                record = query_records.sample()
                obs = Observation.from_record(record)
                is_valid = obs.validate()

                i += 1
                if i >= count_limit and not is_valid:
                    obs = None
                    break
            return obs
        else:
            return None

    df_path = os.path.expanduser(config['paths/extract_dir'])
    notes_df_path = os.path.join(df_path, config['dataframes/notes'])
    notes_df = pandas.read_pickle(notes_df_path)

    tiny_dataset = []
    # Get one file for each instrument for each dataset.
    for dataset in notes_df["dataset"].unique():
        logger.info("Loading Dataset: {}".format(dataset))
        for instrument in notes_df["instrument"].unique():
            logger.info("Loading instrument: {}".format(instrument))
            # Grab it from the notes
            record = sample_record(notes_df, dataset, instrument)
            # If that fails, just grab a file from the original datasets
            if record is None:
                logger.warning("Dataset {} has no instrument '{}'".format(
                    dataset, instrument))
                continue

            tiny_dataset.append(record)

    return tiny_dataset


if __name__ == "__main__":
    utils.setup_logging('INFO')
    logger.info("Building tiny dataset from notes_df.")
    ROOT_DIR = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir)
    CONFIG_PATH = os.path.join(ROOT_DIR, "data", "master_config.yaml")
    config = C.Config.from_yaml(CONFIG_PATH)

    TINY_DATASET_JSON = os.path.normpath(
        os.path.join(ROOT_DIR, config['data/tiny']))

    tinyds = build_tiny_dataset_from_old_dataframe(config)
    tinyds = Dataset(tinyds)
    logger.debug("Tiny Dataset has {} records.".format(len(tinyds)))
    # Save it.
    logger.info("Saving dataset to: {}".format(TINY_DATASET_JSON))
    tinyds.save_json(TINY_DATASET_JSON)
    sys.exit(os.path.exists(TINY_DATASET_JSON) is False)
