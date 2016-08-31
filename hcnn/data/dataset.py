"""A class and utilities for managing the datasets.

A 'dataset' in this context will generally exist on disk as a .csv
file or dataframe.
"""

import json
import jsonschema
import logging
import os
import pandas as pd
import requests
from sklearn.cross_validation import train_test_split
import sys


import hcnn.common.config as C
import hcnn.common.utils as utils

logger = logging.getLogger(__name__)

SCHEMA_PATH = "https://raw.githubusercontent.com/ejhumphrey/minst-dataset/" \
              "master/minst/schema/observation.json"


class MissingDataException(Exception):
    pass


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

    def __init__(self, dataset, audio_file, instrument,
                 index=None, source_key=None,
                 start_time=None, duration=None, note_number=None,
                 dynamic=None, partition=None,
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
        """
        Parameters
        ----------
        record : pandas.Series
        """
        return cls(index=record.name,
                   dataset=record.get('dataset', None),
                   audio_file=record.get('audio_file',
                                         record.get('note_file', None)),
                   instrument=record.get('instrument', None),
                   source_key=record.get('source_key', ""),
                   start_time=record.get('start_time', 0.0),
                   duration=record.get('duration', None),
                   note_number=record.get('note', None),
                   dynamic=record.get('dynamic', None),
                   partition=record.get('partition', ""))

    def to_dict(self):
        return self.__dict__.copy()

    def __getitem__(self, key):
        return self.__dict__[key]

    def to_series(self):
        """Convert to a flat series (ie make features a column)

        Returns
        -------
        pd.Series
        """
        flat_dict = self.to_dict()
        flat_dict.update(**flat_dict.pop("features"))
        return pd.Series(flat_dict)

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


def expand_audio_paths(df, data_root):
    # Update the paths to full paths, and make sure it's
    # saved in 'audio_file'
    new_df = df.copy()
    for idx, item in new_df.iterrows():
        assert 'audio_file' in item
        new_df.loc[idx, 'audio_file'] = os.path.expanduser(
            os.path.join(data_root, item['audio_file']))
    return new_df


class Dataset(object):
    def __init__(self, df, split=None):
        self.df = df
        self.split = split

    @classmethod
    def from_observations(cls, observations):
        obs_series = [x.to_series() for x in observations]
        return cls(pd.DataFrame(obs_series))

    @classmethod
    def load(cls, path, data_root=None):
        _, ext = os.path.splitext(path)
        if ext == '.json':
            return cls.read_json(path, data_root)
        elif ext == '.csv':
            return cls.read_csv(path, data_root)
        else:
            raise NotImplementedError()

    @classmethod
    def read_json(cls, json_path, data_root=None):
        if os.path.exists(json_path):
            df = pd.read_json(json_path)
            if data_root:
                df = expand_audio_paths(df, data_root)
            return cls(df)
        else:
            logger.error("No dataset available at {}".format(json_path))
            return None

    @classmethod
    def read_csv(cls, csv_path, data_root=None):
        df = pd.read_csv(csv_path, index_col=0)
        if data_root:
            df = expand_audio_paths(df, data_root)
        return cls(df)

    def copy(self):
        return Dataset(self.df.copy(), self.split)

    def save_json(self, json_path):
        with open(json_path, 'w') as fh:
            json.dump(self.to_builtin(), fh)

    def save_csv(self, csv_path):
        self.df.to_csv(csv_path)
        return os.path.exists(csv_path)

    def save(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.csv':
            self.save_csv(path)
        elif ext == '.json':
            self.save_json(path)
        else:
            raise NotImplementedError()

    def to_df(self):
        """Returns the dataset as a dataframe."""
        return self.df

    def to_builtin(self):
        """Returns a list of dicts, where the dict
        is functionally equivalent to an observation.
        """
        index_dict = self.df.to_dict(orient='index')
        df_as_list = []
        for k, v in index_dict.items():
            v['index'] = k
            df_as_list.append(v)
        return df_as_list

    def as_observations(self):
        obs = []
        for idx, item in self.df.iterrows():
            obs.append(Observation.from_record(item))
        return obs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return Observation(**self.df.iloc[index].to_dict())

    @property
    def datasets(self):
        return list(self.df['dataset'].unique())

    def sample(self, *args, **kwargs):
        """Thin wrapper on DataFrame.sample, but returns a Dataset."""
        return Dataset(self.df.sample(*args, **kwargs))

    def filter(self, dataset_name=None, instrument=None, invert=False):
        """Return a copy of the dataset, filtering on dataset name
        or instrument
        """
        result = self.df.copy()
        if dataset_name:
            if invert:
                result = result[result['dataset'] != dataset_name]
            else:
                result = result[result['dataset'] == dataset_name]
        if instrument:
            if invert:
                result = result[result['instrument'] != instrument]
            else:
                result = result[result['instrument'] == instrument]

        return Dataset(result)

    def test_set(self, selected_set):
        """Assumption: a "test set" is simply all of the samples from
        one of the datasets.

        Parameters
        ----------
        selected_set : str in ['rwc', 'philharmonia', 'uiowa']
            Subsamples to get for the test set.
        """
        ds = self.filter(dataset_name=selected_set)
        ds.split = 'test'
        return ds

    def train_valid_sets(self, test_set, train_val_split=0.2,
                         max_files_per_class=None):
        """Assumption: a "test set" is simply all of the samples from
        one of the datasets

        Returns Datasets for train and validation constructed
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

        train_val_split : float
            Amount of validation data to split out from the dataset.

        max_files_per_class : int or None
            Limit the data to this number of samples per class.

        Returns
        -------
        train_df, valid_df : pd.DataFrame
            DataFrames referencing the files for train and validation.
        """
        df = self.filter(dataset_name=test_set, invert=True).to_df()

        selected_instruments_train = []
        selected_instruments_valid = []
        for instrument in df["instrument"].unique():
            instrument_df = df[df["instrument"] == instrument]

            if len(instrument_df) < 2:
                logger.warning("Instrument {} doesn't haven enough samples "
                               "to split, so putting it in both [bad practice, but what ya gonna do?]".format(instrument))
                selected_instruments_train.append(instrument_df)
                selected_instruments_valid.append(instrument_df)
            else:

                traindf, validdf = train_test_split(
                    instrument_df, test_size=train_val_split)

                if max_files_per_class:
                    replace = (False if len(traindf) > max_files_per_class
                               else True)
                    traindf = traindf.sample(n=max_files_per_class,
                                             replace=replace)

                selected_instruments_train.append(traindf)
                selected_instruments_valid.append(validdf)

        return Dataset(pd.concat(selected_instruments_train), 'train'), \
            Dataset(pd.concat(selected_instruments_valid), 'valid')


class TinyDataset(Dataset):
    ROOT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.pardir, os.pardir, "data", "tinyset"))
    DS_FILE = os.path.join(ROOT_PATH, "notes_index.csv")

    @classmethod
    def load(cls, data_path=DS_FILE, data_root=ROOT_PATH):
        return Dataset.read_csv(data_path, data_root=data_root)


def build_tiny_dataset_from_old_dataframe(config):
    warings.warn("This function has been deprecated. Remove it ASAP")

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
    notes_df = pd.read_pickle(notes_df_path)

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
