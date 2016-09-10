import boltons.fileutils
import copy
import datetime
import glob
import json
import logging
import logging.config
import numpy as np
import os
import pandas as pd
import pytest

import hcnn.common.config as C
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
                           "data", "integration_config.yaml")
config = C.Config.load(CONFIG_PATH)


@pytest.fixture
def config_with_workspace(workspace):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['iteration_write_frequency'] = 1
    thisconfig.data['training']['iteration_print_frequency'] = 5
    thisconfig.data['training']['max_iterations'] = 20
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['training']['max_files_per_class'] = 3
    thisconfig.data['paths']['model_dir'] = os.path.join(workspace, "models")
    thisconfig.data['paths']['feature_dir'] = os.path.join(workspace, "cqt")

    return thisconfig


def generate_example_training_loss(config, output_dir, n_iter):
    def training_loss(n):
        return pd.Series({
            'timestamp': datetime.datetime.now(),
            'batch_train_dur': np.random.random(),
            'iteration': n_iter,
            'loss': np.random.random()
        })

    training_loss_fp = config['experiment']['training_loss']
    train_loss_path = os.path.join(output_dir, training_loss_fp)

    train_stats = pd.DataFrame([training_loss(n) for n in range(5)])        
    train_stats.to_pickle(train_loss_path)


def generate_example_validation_loss(config, output_dir, n_iter):
    def validation_series(n):
        return pd.Series({
            'mean_acc': np.random.random(),
            'mean_loss': np.random.random(),
            'model_file': "foobar.npz",
            'model_iteration': n
        })
    validation_loss_fp = config['experiment']['validation_loss']
    validation_loss_path = os.path.join(output_dir, validation_loss_fp)

    validation_df = pd.DataFrame([validation_series(n) for n in range(5)])
    validation_df.to_pickle(validation_loss_path)


def generate_example_predictions(config, output_dir, n_iter, n_classes):
    def a_prediction():
        return pd.Series({
            'y_pred': np.random.randint(n_classes),
            'target': np.random.randint(n_classes)
        })

    predictions_format = config['experiment']['predictions_format']
    predictions_fp = predictions_format.format("00010")
    predictions_path = os.path.join(output_dir, predictions_fp)
    predictions_df = pd.DataFrame([a_prediction() for n in range(5)])
    predictions_df.to_pickle(predictions_path)


@pytest.fixture
def available_datasets():
    return ["rwc", 'uiowa', 'philharmonia']


@pytest.fixture
def pre_existing_experiment(config_with_workspace, available_datasets):
    """Create some template existing experiment data."""
    experiment_name = "testexperiment"
    model_dir = config_with_workspace['paths']['model_dir']
    experiment_dir = os.path.join(model_dir, experiment_name)

    n_iter = 5
    n_classes = 12
    for dataset in available_datasets:
        dataset_dir = os.path.join(experiment_dir, dataset)
        boltons.fileutils.mkdir_p(dataset_dir)

        generate_example_training_loss(config, dataset_dir, n_iter)
        generate_example_validation_loss(config, dataset_dir, n_iter)
        generate_example_predictions(config, dataset_dir, n_iter, n_classes)

    return experiment_name



@pytest.mark.slowtest
def test_extract_features(module_workspace, tiny_feats):
    for idx, obs in tiny_feats.to_df().iterrows():
        assert "cqt" in obs
        assert os.path.exists(obs.cqt)


@pytest.mark.runme
@pytest.mark.slowtest
@pytest.mark.parametrize("model_name",
                         ["cqt_MF_n16",
                          # "wcqt",
                          "cqt_M2_n8",
                          "hcqt_MH_n8"],
                         ids=["cqt_MF_n16",
                              "cqt_M2_n8",
                              "hcqt_MH_n8"])
def test_train_simple_model(model_name, module_workspace, workspace,
                            tiny_feats_csv):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['iteration_write_frequency'] = 2
    thisconfig.data['training']['max_iterations'] = 10
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['training']['max_files_per_class'] = 1
    thisconfig.data['paths']['model_dir'] = workspace
    thisconfig.data['paths']['feature_dir'] = module_workspace
    # The features get loaded by tiny_feats_csv anyway
    thisconfig.data['features']['cqt']['skip_existing'] = True
    experiment_name = "testexperiment"
    hold_out = "rwc"

    driver = hcnn.driver.Driver(thisconfig,
                                model_name=model_name,
                                experiment_name=experiment_name,
                                dataset=tiny_feats_csv, load_features=True)

    driver.setup_partitions(hold_out)
    result = driver.train_model()
    assert result is True

    # Expected files this should generate
    new_config = os.path.join(workspace, experiment_name, "config.yaml")
    train_loss_fp = os.path.join(workspace, experiment_name, hold_out,
                                 "training_loss.pkl")
    assert os.path.exists(new_config)
    assert os.path.exists(train_loss_fp)


@pytest.mark.slowtest
@pytest.mark.parametrize("model_name",
                         ["cqt_MF_n16",
                          # "wcqt",
                          "cqt_M2_n8",
                          "hcqt_MH_n8"],
                         ids=["cqt_MF_n16",
                              "cqt_M2_n8",
                              "hcqt_MH_n8"])
def test_find_best_model(config_with_workspace, model_name, workspace):
    experiment_name = "testexperiment"
    hold_out = "rwc"
    driver = hcnn.driver.Driver(config_with_workspace, model_name=model_name,
                                experiment_name=experiment_name,
                                load_features=True)
    driver.setup_partitions(hold_out)
    result = driver.train_model()
    assert result is True

    # Create a vastly reduced validation dataframe so it'll take less long.
    validation_size = 3
    driver.valid_set.df = driver.valid_set.df.sample(n=validation_size,
                                                     replace=True)
    assert len(driver.valid_set.df) == validation_size
    driver.test_set.df = driver.test_set.df.sample(n=validation_size,
                                                   replace=True)

    results_df = driver.find_best_model()
    # check that the results_df is ordered by iteration.
    assert all(results_df["model_iteration"] ==
               sorted(results_df["model_iteration"]))

    # Get the best param
    param_iter = driver.select_best_iteration(results_df)
    assert param_iter is not None

    # load it again to test the reloading thing.
    #  Just making sure this runs through
    results_df2 = driver.find_best_model()
    assert all(results_df == results_df2)

    # Shrink the dataset so this doesn't take forever.
    driver.dataset.df = driver.dataset.df.sample(n=10, replace=True)
    predictions_df = driver.predict(param_iter)
    assert not predictions_df.empty
    predictions_df_path = os.path.join(
        workspace, experiment_name, hold_out,
        "model_{}_predictions.pkl".format(param_iter))
    assert os.path.exists(predictions_df_path)


def test_collect_results(config_with_workspace, pre_existing_experiment,
                         available_datasets, workspace):
    driver = hcnn.driver.Driver(config_with_workspace,
                                experiment_name=pre_existing_experiment,
                                load_features=False,
                                skip_load_dataset=True)

    destination_dir = os.path.join(workspace, "results")
    result = driver.collect_results(destination_dir)

    assert result is True

    new_experiment_dir = os.path.join(destination_dir, pre_existing_experiment)
    assert os.path.isdir(new_experiment_dir)

    for dataset in available_datasets:
        dataset_results = os.path.join(new_experiment_dir, dataset)
        assert os.path.isdir(dataset_results)

        training_loss_fp = os.path.join(dataset_results, "training_loss.pkl")
        assert os.path.isfile(training_loss_fp)
        training_loss_df = pd.read_pickle(training_loss_fp)
        assert [x in training_loss_df.columns for x in ['iteration', 'loss']]

        validation_loss_fp = os.path.join(dataset_results,
                                          "validation_loss.pkl")
        assert os.path.isfile(validation_loss_fp)
        validation_loss_df = pd.read_pickle(validation_loss_fp)
        assert [x in validation_loss_df.columns
                for x in ['mean_acc', 'mean_loss', 'model_file',
                          'model_iteration']]

        prediction_glob = os.path.join(dataset_results, "*predictions.pkl")
        assert len(prediction_glob) > 0
        prediction_file = glob.glob(prediction_glob)[0]
        prediction_df = pd.read_pickle(prediction_file)
        assert [x in prediction_df.columns for x in ['y_pred', 'y_true']]

    # Finally, collect_results should create an overall analysis of the
    # three-fold validation, and put it in
    overall_results_fp = os.path.join(
        new_experiment_dir, "experiment_results.json")
    assert os.path.isfile(overall_results_fp)
    with open(overall_results_fp, 'r') as fh:
        result_data = json.load(fh)
    for dataset in available_datasets:
        assert dataset in result_data
        assert 'mean_accuracy' in result_data[dataset]
        assert 'mean_precision' in result_data[dataset]
        assert 'mean_recall' in result_data[dataset]
        assert 'mean_f1' in result_data[dataset]
        assert 'class_precision' in result_data[dataset]
        assert 'class_recall' in result_data[dataset]
        assert 'class_f1' in result_data[dataset]
        assert 'sample_weight' in result_data[dataset]
