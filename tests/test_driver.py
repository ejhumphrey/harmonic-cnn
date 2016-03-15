import copy
import logging
import logging.config
import numpy as np
import os
import pandas
import pytest

import wcqtlib.config as C
import wcqtlib.train.driver as driver

logger = logging.getLogger(__name__)
logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': "standard"
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


def test_construct_training_valid_df():
    def __test_result_df(traindf, validdf, datasets, n_per_inst):
        if n_per_inst:
            assert len(traindf) == 12 * n_per_inst
            assert len(validdf)
        else:
            total_len = len(traindf) + len(validdf)
            np.testing.assert_almost_equal((total_len*.8)/100.,
                                           len(traindf)/100.,
                                           decimal=0)
            np.testing.assert_almost_equal((total_len*.2)/100.,
                                           len(validdf)/100.,
                                           decimal=0)
        assert set(datasets) == set(traindf['dataset'])
        assert set(datasets) == set(validdf['dataset'])

    datasets = ["rwc"]
    n_files_per_inst = 1
    train_df, valid_df = driver.construct_training_valid_df(
        features_df, datasets, max_files_per_class=n_files_per_inst)
    yield __test_result_df, train_df, valid_df, datasets, n_files_per_inst

    datasets = ["rwc", "uiowa"]
    n_files_per_inst = 5
    train_df, valid_df = driver.construct_training_valid_df(
        features_df, datasets, max_files_per_class=n_files_per_inst)
    yield __test_result_df, train_df, valid_df, datasets, n_files_per_inst

    datasets = ["rwc", "philharmonia"]
    n_files_per_inst = None
    train_df, valid_df = driver.construct_training_valid_df(
        features_df, datasets, max_files_per_class=n_files_per_inst)
    yield __test_result_df, train_df, valid_df, datasets, n_files_per_inst


@pytest.mark.slowtest
def test_train_simple_model(workspace):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['max_iterations'] = 200
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['paths']['model_dir'] = workspace
    experiment_name = "testexperiment"
    hold_out = "rwc"

    driver.train_model(thisconfig, 'wcqt_iX_c1f1_oY',
                       experiment_name, hold_out,
                       max_files_per_class=1)

    # Expected files this should generate
    logger.debug("1")
    new_config = os.path.join(workspace, experiment_name, "config.yaml")
    final_params = os.path.join(workspace, experiment_name, "params",
                                "final.npz")
    train_loss_fp = os.path.join(workspace,
                                 experiment_name, "training_loss.pkl")
    logger.debug("2")
    assert os.path.exists(new_config) and os.path.exists(final_params)
    assert os.path.exists(train_loss_fp)

    # Also make sure the training & validation splits got written out
    logger.debug("3")
    train_fp = os.path.join(workspace, experiment_name,
                            "train_df_{}.pkl".format(hold_out))
    logger.debug("4")
    assert os.path.exists(train_fp)
    logger.debug("5")
    valid_fp = os.path.join(workspace, experiment_name,
                            "train_df_{}.pkl".format(hold_out))
    logger.debug("6")
    assert os.path.exists(valid_fp)
