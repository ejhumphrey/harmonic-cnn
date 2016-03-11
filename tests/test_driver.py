import lasagne
import logging
import numpy as np
import os
import pandas
import pytest
import sys

import wcqtlib.config as C
import wcqtlib.train.driver as driver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


def test_construct_training_df():
    def __test_result_df(df, datasets, n_per_inst):
        assert len(df) == 12 * n_per_inst
        assert set(datasets) == set(df['dataset'])
    datasets = ["rwc"]
    n_files_per_inst = 1
    new_df = driver.construct_training_df(
        features_df, datasets, n_files_per_inst)
    yield __test_result_df, new_df, datasets, n_files_per_inst

    datasets = ["rwc", "uiowa"]
    n_files_per_inst = 5
    new_df = driver.construct_training_df(
        features_df, datasets, n_files_per_inst)
    yield __test_result_df, new_df, datasets, n_files_per_inst
