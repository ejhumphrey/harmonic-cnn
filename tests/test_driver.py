import copy
import logging
import logging.config
import os
import pandas
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
def test_find_best_model(model_name, workspace):
    thisconfig = copy.deepcopy(config)
    thisconfig.data['training']['iteration_write_frequency'] = 2
    thisconfig.data['training']['iteration_print_frequency'] = 10
    thisconfig.data['training']['max_iterations'] = 50
    thisconfig.data['training']['batch_size'] = 12
    thisconfig.data['training']['max_files_per_class'] = 3
    thisconfig.data['paths']['model_dir'] = workspace
    thisconfig.data['experiment']['hold_out_set'] = "rwc"
    experiment_name = "testexperiment"
    hold_out = thisconfig['experiment/hold_out_set']

    driver = hcnn.driver.Driver(thisconfig, model_name=model_name,
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
