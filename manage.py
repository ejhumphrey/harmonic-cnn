"""Master script to run Harmonic-CNN Experiments.

Usage:
 manage.py [options] run
 manage.py [options] run <model>
 manage.py [options] extract_features
 manage.py [options] experiment (train|predict|fit_and_predict|analyze) <experiment_name> <test_set> <model>
 manage.py [options] test [(data|model|unit)]
 manage.py [options] collect_results <results_destination> [<experiment_name>] [--integration]

Arguments:
 run           Run all of the the experiments end-to-end.
 extract_features  Manually extract features from the dataset audio files.
               (This will happen automatically in a 'run'.)
 experiment    fit_and_predict  Train over a specified partition, and
                      immediately runs the predictions over the test set.
               train       Run only the training compoment for a specified
                           partition.
               predict     Run only the prediction component over a specified
                           partition.
               analyze     Create a report on the run for analysis.
 test          Run tests.
               'data' tests to make sure the data is setup to run.
               'model' runs simple train and predict on a small subset of data.
               'unit' runs unit tests.
 collect_results  Collect all the relevant results files from experiments
               into the results_destination. Most likely used to transfer
               the results into this git repository.
               Use 'experiment_name' to only copy results from one experiment.
               'results_destination' is the root destination for all results.
               Recommended path is the './results' folder in this repo.

Options:
 -v --verbose  Increase verbosity.
 --skip_features  Prevent the driver from trying to create features.
 --skip_training  Skip the trianing process, and just do the model selection
               and prediction for each model given.
"""

from docopt import docopt
import logging
import os
import pytest
import shutil
import sys
import theano

import hcnn.common.config as C
import hcnn.common.utils as utils
import hcnn.data.dataset
import hcnn.data.cqt
import hcnn.driver
import hcnn.logger

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "data", "master_config.yaml")
INT_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                               "data", "integration_config.yaml")

logger = logging.getLogger(__name__)

# theano debug values. probably remove these later.
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'


def run_process_if_not_exists(process, filepath, **kwargs):
    if not os.path.exists(filepath):
        return process(**kwargs)
    else:
        return True


def clean(config_path, force=False):
    """Clean dataframes and extracted audio/features."""
    config = C.Config.load(config_path)

    data_path = os.path.expanduser(config['paths/feature_dir'])
    # Clean data
    if not force:
        answer = input("Are you sure you want to delete {} (y|s to skip): "
                       .format(data_path))
        if answer in ['y', 'Y']:
            pass
        elif answer in ['s', 'S']:
            return True
        else:
            print("Exiting")
            sys.exit(1)

    shutil.rmtree(data_path)
    logger.info("clean done.")
    return True


def extract_features(master_config):
    """Extract CQTs from all files collected in collect."""
    config = C.Config.load(master_config)
    print(utils.colored("Extracting CQTs from note audio."))

    driver = hcnn.driver.Driver(config, load_features=False)
    result = driver.extract_features()
    print("Extraction {}".format(utils.result_colored(result)))
    return result


def fit_and_predict(config, experiment_name, test_set, model_name):
    """Runs:
    - train
    - model_selection_df
    - predict
    - analyze
    """
    run_name = "fit_and_predict:{}:{}:{}".format(
        experiment_name, test_set, model_name)

    config = C.Config.load(config)
    print(utils.colored("Running {} end-to-end.".format(run_name)))

    timer = utils.TimerHolder()
    timer.start(run_name)
    logger.debug("Running model={} with experiment_name={} at {}"
                 .format(model_name, experiment_name,
                         timer.get_start(run_name)))
    driver = hcnn.driver.Driver(config,
                                model_name=model_name,
                                experiment_name=experiment_name,
                                load_features=True)
    result = driver.fit_and_predict_one(test_set)
    print("{} - {} complted in duration {}".format(
        run_name, utils.result_colored(result), timer.end(run_name)))
    return result


def train(config,
          experiment_name,
          test_set,
          model_name):
    """Run training loop.

    Parameters
    ----------
    config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    test_set : str
        String in ["rwc", "uiowa", "philharmonia"] specifying which
        dataset to use as the test set.

    model_name : str
        Name of the model to use for training.
    """
    print(utils.colored("Training experiment: {}".format(experiment_name)))
    logger.info("Training model '{}' with test_set '{}'"
                .format(model_name, test_set))
    driver = hcnn.driver.Driver(config, test_set,
                                model_name=model_name,
                                experiment_name=experiment_name,
                                load_features=True)

    return driver.train_model()


def predict(config,
            experiment_name,
            test_set,
            model_name,
            select_epoch=None):
    """Predict results on all datasets and report results.

    Parameters
    ----------
    config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    model_name : str
        Name of the model to use for training. Must match the training
        configuration.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Evaluating"))
    config = C.Config.load(config)

    driver = hcnn.driver.Driver(config, model_name=model_name,
                                experiment_name=experiment_name,
                                load_features=True)
    results = driver.predict(select_epoch)
    logger.info("Generated results for {} files.".format(len(results)))


def analyze(master_config,
            experiment_name,
            select_epoch=None):
    """Predict results on all datasets and report results.

    Parameters
    ----------
    master_config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    dataset : str
        Dataset to select results from for analysis.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Analyzing"))
    config = C.Config.load(master_config)

    hold_out_set = config["experiment/hold_out_set"]

    driver = hcnn.driver.Driver(config, experiment_name=experiment_name,
                                load_features=True)

    driver.analyze(select_epoch, hold_out_set)
    return 0


def run_all_experiments(config, experiment_root=None,
                        skip_features=False, skip_training=False):
    MODELS_TO_RUN = [
        'cqt_MF_n16',
        'cqt_M2_n8',
        'hcqt_MH_n8'
    ]

    # MODELS_TO_RUN = [
    #     'cqt_MF_n32',
    #     'cqt_MF_n64',
    #     'cqt_M2_n16',
    #     'cqt_M2_n32',
    #     'cqt_M2_n64'
    #     'hcqt_MH_n16',
    #     'hcqt_MH_n32',
    #     'hcqt_MH_n64'
    # ]

    results = []
    for model_name in MODELS_TO_RUN:
        results.append(run_experiment(model_name, config, experiment_root,
                                      skip_features=skip_features,
                                      skip_training=skip_training))

    return all(results)


def run_experiment(model_name, config, experiment_root=None,
                   skip_features=False,
                   skip_training=False):
    """Run an experiment using the specified input feature

    Parameters
    ----------
    model_name : str
        Name of the NN model configuration [in models.py].
    """
    logger.info("run_experiment(model_name='{}')".format(model_name))
    config = C.Config.load(config)
    experiment_name = "{}{}".format(
        "{}_".format(experiment_root) if experiment_root else "",
        model_name)
    logger.info("Running Experiment: {}".format(
        utils.colored(experiment_name, 'magenta')))

    driver = hcnn.driver.Driver(config,
                                model_name=model_name,
                                experiment_name=experiment_name,
                                load_features=True,
                                skip_features=skip_features,
                                skip_training=skip_training,
                                skip_cleaning=skip_training)
    result = driver.fit_and_predict_cross_validation()

    return result


def run_tests(mode):
    logger.info("run_tests(mode='{}')".format(mode))

    config = INT_CONFIG_PATH

    results = []
    if mode in ['all', 'unit']:
        run_unit_tests()
    if mode in ['data']:
        results.append(test_data(config))
    if mode in ['all', 'model']:
        results.append(integration_test(config))

    return all(results)


def run_unit_tests():
    return 0 == pytest.main('.')


def test_data(config, show_full=False):
    driver = hcnn.driver.Driver(
        config, experiment_name="data_test", load_features=False)
    return driver.validate_data()


def integration_test(config):
    """AKA "model" test.
    This is equivalent to running
    python manage.py -c data/integrationtest_config.yaml run_all_experiments
    """
    # Load integrationtest config
    experiment_name = "integrationtest"

    print(utils.colored("Extracting features from tinydata set."))
    print(utils.colored(
        "Running integration test on tinydata set : {}."
        .format(config)))
    # Begin by cleaning the feature data
    result = clean(config, force=True)
    if result:
        result = run_all_experiments(config, experiment_root=experiment_name)

    print("IntegrationTest Result: {}".format(utils.result_colored(result)))
    return result


def collect_results(config, destination, experiment_name=None,
                    use_integration=False):
    print(utils.colored("Collecting Results"))
    if use_integration:
        config = INT_CONFIG_PATH

    if experiment_name is None:
        experiments = hcnn.driver.Driver.available_experiments(config)
    else:
        experiments = [experiment_name]

    results = []
    for experiment_name in experiments:
        if not use_integration and "integrationtest" in experiment_name:
            continue
        elif (use_integration and
                "integrationtest" not in experiment_name):
            continue

        print("Collecting experiment", utils.colored(experiment_name, 'cyan'))

        driver = hcnn.driver.Driver(config, experiment_name=experiment_name,
                                    load_features=False,
                                    skip_load_dataset=True)

        results.append(driver.collect_results(destination))

    return all(results)


def handle_arguments(arguments):
    config = CONFIG_PATH
    logger.debug(arguments)

    # Run modes
    if arguments['run']:
        model = arguments['<model>']
        skip_training = arguments['--skip_training']
        skip_features = arguments['--skip_features']

        logger.info("Run Mode; model={}".format(model))
        if model:
            result = run_experiment(model, config,
                                    skip_features=skip_features,
                                    skip_training=skip_training)
        else:
            result = run_all_experiments(config,
                                         skip_features=skip_features,
                                         skip_training=skip_training)

    elif arguments['extract_features']:
        logger.info('Extracting features.')
        result = extract_features(config)

    # Basic Experiment modes
    elif arguments['experiment']:
        if arguments['fit_and_predict']:
            mode = 'fit_and_predict'
        elif arguments['train']:
            mode = 'train'
        elif arguments['predict']:
            mode = 'predict'
        elif arguments['analyze']:
            mode = 'analyze'
        else:
            # docopt should not allow us to get here.
            raise ValueError("No valid experiment mode set.")

        experiment_name = arguments['<experiment_name>']
        test_set = arguments['<test_set>']
        model = arguments['<model>']

        logger.info("Running experiment '{}' with test_set '{}' "
                    "using model '{}'".format(
                        experiment_name, test_set, model))

        # Use the 'mode' to select the function to call.
        result = globals().get(mode)(config, experiment_name, test_set, model)

    # Test modes
    elif arguments['test']:
        test_type = 'all'
        if arguments['data']:
            test_type = 'data'
        elif arguments['model']:
            test_type = 'model'
        elif arguments['unit']:
            test_type = 'unit'

        logger.info('Running {} tests'.format(test_type))

        result = run_tests(test_type)

    elif arguments['collect_results']:
        experiment_name = arguments.get('<experiment_name>', None)
        destination = arguments['<results_destination>']
        integration_mode = arguments['--integration']

        result = collect_results(config, destination, experiment_name,
                                 use_integration=integration_mode)

    return result


if __name__ == "__main__":
    # parser.add_argument("-c", "--master_config", default=CONFIG_PATH)

    arguments = docopt(__doc__)
    utils.setup_logging(logging.DEBUG if arguments['--verbose']
                        else logging.INFO)

    handle_arguments(arguments)
