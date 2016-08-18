import copy
import logging
import logging.config
import numpy as np
import os
import pandas
import pytest

import hcnn.common.config as C
import hcnn.data.dataset as dataset
import hcnn.data.cqt
import hcnn.driver

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
config = C.Config.load(CONFIG_PATH)


@pytest.mark.slowtest
def test_extract_features(module_workspace, tiny_feats):
    for obs in tiny_feats.items:
        assert "cqt" in obs.features
        assert os.path.exists(obs.features['cqt'])


@pytest.mark.runme
@pytest.mark.slowtest
def test_train_simple_model(workspace, tiny_feats):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['max_iterations'] = 200
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['training']['max_files_per_class'] = 1
    thisconfig.data['paths']['model_dir'] = workspace
    experiment_name = "testexperiment"
    hold_out = "rwc"

    driver = hcnn.driver.Driver(thisconfig, experiment_name,
                                   load_features=True)

    result = driver.train_model(hold_out)
    assert result is True

    # Expected files this should generate
    new_config = os.path.join(workspace, experiment_name, "config.yaml")
    train_loss_fp = os.path.join(workspace, experiment_name, hold_out,
                                 "training_loss.pkl")
    assert os.path.exists(new_config)
    assert os.path.exists(train_loss_fp)

    # Also make sure the training & validation splits got written out
    train_fp = os.path.join(workspace, experiment_name, hold_out,
                            "train_df_{}.pkl".format(hold_out))
    assert os.path.exists(train_fp)
    valid_fp = os.path.join(workspace, experiment_name, hold_out,
                            "train_df_{}.pkl".format(hold_out))
    assert os.path.exists(valid_fp)


@pytest.mark.slowtest
def test_find_best_model(workspace):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['iteration_write_frequency'] = 5
    thisconfig.data['training']['iteration_print_frequency'] = 10
    thisconfig.data['training']['max_iterations'] = 200
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['training']['max_files_per_class'] = 3
    thisconfig.data['paths']['model_dir'] = workspace
    thisconfig.data['experiment']['hold_out_set'] = "rwc"
    experiment_name = "testexperiment"
    hold_out = thisconfig['experiment/hold_out_set']

    valid_df_path = os.path.join(
        workspace, experiment_name, hold_out,
        thisconfig['experiment/data_split_format'].format(
            "valid", hold_out))

    driver = hcnn.driver.Driver(thisconfig, experiment_name,
                                   load_features=True)
    result = driver.train_model(hold_out)
    assert result is True

    # This should have been created by the training process.
    assert os.path.exists(valid_df_path)

    # Create a vastly reduced validation dataframe so it'll take less long.
    validation_size = 20
    valid_df = pandas.read_pickle(valid_df_path).sample(n=validation_size,
                                                        replace=True)
    assert len(valid_df) == validation_size

    results_df = driver.find_best_model(valid_df)
    # check that the results_df is ordered by iteration.
    assert all(results_df["model_iteration"] ==
               sorted(results_df["model_iteration"]))

    # Get the best param
    param_iter = driver.select_best_iteration(results_df)
    assert param_iter is not None

    # load it again to test the reloading thing.
    #  Just making sure this runs through
    results_df2 = driver.find_best_model(valid_df)
    assert all(results_df == results_df2)

    predictions_df = driver.predict(param_iter)
    assert not predictions_df.empty
    predictions_df_path = os.path.join(
        workspace, experiment_name, hold_out,
        "model_{}_predictions.pkl".format(param_iter))
    assert os.path.exists(predictions_df_path)
