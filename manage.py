"""Master script to run Harmonic-CNN Experiments.

Usage:
 manage.py [options] run
 manage.py [options] run [<cqt> | <wcqt> | <hcqt>]
 manage.py [options] extract_features
 manage.py [options] experiment (train|predict|fit_and_predict|analyze) <experiment_name> <test_set> <model>
 manage.py [options] test [(data|model|unit)]

Arguments:
 run           Run all of the the experiments end-to-end.
               'cqt' runs only on cqt features.
               'wcqt' runs only on wcqt features.
               'hcqt' runs only on hcqt features.
 extract_features  Manually extract features from the dataset audio files.
               (This will happen automatically in a full 'run'.)
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

Options:
 -v --verbose  Increase verbosity.
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


def clean(master_config):
    """Clean dataframes and extracted audio/features."""
    config = C.Config.load(master_config)

    data_path = os.path.expanduser(config['paths/extract_dir'])
    # Clean data
    answer = input("Are you sure you want to delete {} (y|s to skip): "
                   .format(data_path))
    if answer in ['y', 'Y']:
        shutil.rmtree(data_path)
        logger.info("clean done.")
    elif answer in ['s', 'S']:
        return True
    else:
        print("Exiting")
        sys.exit(1)


def extract_features(master_config):
    """Extract CQTs from all files collected in collect."""
    config = C.Config.load(master_config)
    print(utils.colored("Extracting CQTs from note audio."))

    driver = hcnn.driver.Driver(config, load_features=False)
    result = driver.extract_features()
    print("Extraction {}".format(utils.result_colored(result)))
    return result


def run(master_config, experiment_name):
    """Run an experiment end-to-end with cross validation.
    Note: requires extracted features.
    """
    config = C.Config.load(master_config)
    print(utils.colored("Running experiment end-to-end."))

    timer = utils.TimerHolder()
    timer.start("run")
    logger.debug("Running with experiment_name={} at {}"
                 .format(experiment_name, timer.get_start("run")))
    driver = hcnn.driver.Driver(config, experiment_name,
                                load_features=True)
    result = driver.fit_and_predict_cross_validation()
    print("Experiment {} in duration {}".format(
        utils.result_colored(result), timer.end("run")))
    return result


def fit_and_predict(config, experiment_name, test_set, feature_mode):
    """Runs:
    - train
    - model_selection_df
    - predict
    - analyze
    """
    run_name = "fit_and_predict:{}:{}".format(experiment_name, test_set)

    config = C.Config.load(config)
    print(utils.colored("Running {} end-to-end.".format(run_name)))

    timer = utils.TimerHolder()
    timer.start(run_name)
    logger.debug("Running with experiment_name={} at {}"
                 .format(experiment_name, timer.get_start(run_name)))
    driver = hcnn.driver.Driver(config, experiment_name,
                                load_features=True)
    result = driver.fit_and_predict_one(test_set)
    print("{} - {} complted in duration {}".format(
        run_name, utils.result_colored(result), timer.end(run_name)))
    return result


def train(config,
          experiment_name,
          test_set,
          feature_mode):
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

    feature_mode : str in ['cqt', 'wcqt', 'hcqt']
        What type of features to use (and associated model) in training
        the network.
    """
    print(utils.colored("Training experiment: {}".format(experiment_name)))
    logger.info("Training with test_set '{}', and features '{}'".format(
        test_set, feature_mode))
    driver = hcnn.driver.Driver(config, test_set, feature_mode,
                                experiment_name=experiment_name,
                                load_features=True)

    return driver.train_model()


def predict(config,
            experiment_name,
            test_set,
            feature_mode,
            select_epoch=None):
    """Predict results on all datasets and report results.

    Parameters
    ----------
    config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    feature_mode : str in ['cqt', 'wcqt', 'hcqt']
        What type of features to use (and associated model) in evaluating
        the network. Must match the training configuration.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Evaluating"))
    config = C.Config.load(config)

    driver = hcnn.driver.Driver(config, experiment_name,
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

    driver.analyze(config, experiment_name, select_epoch, hold_out_set)
    return 0


def run_all_experiments(config, experiment_root=None):
    results = []
    for features in ['cqt', 'wcqt', 'hcqt']:
        results.append(run_experiment(features, config, experiment_root))

    return all(results)


def run_experiment(input_feature, config, experiment_root=None):
    """Run an experiment using the specified input feature

    Parameters
    ----------
    input_feature : ['cqt', 'wcqt', 'hcqt']
    """
    logger.info("run_experiment(input_feature='{}')".format(input_feature))
    config = C.Config.load(config)
    experiment_name = "{}{}".format(
        "{}_".format(experiment_root) if experiment_root else "",
        input_feature)
    logger.info("Running Experiment: {}".format(
        utils.colored(experiment_name, 'magenta')))

    driver = hcnn.driver.Driver(config, experiment_name=experiment_name,
                                load_features=True)
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
    driver = hcnn.driver.Driver(config, "data_test", load_features=False)
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
    result = run_all_experiments(config, experiment_root=experiment_name)

    print("IntegrationTest Result: {}".format(utils.result_colored(result)))
    return result


def handle_arguments(arguments):
    config = CONFIG_PATH
    logger.debug(arguments)

    # Run modes
    if arguments['run']:
        feature = None
        if arguments['<cqt>']:
            feature = 'cqt'
        elif arguments['<wcqt>']:
            feature = 'wcqt'
        elif arguments['<hcqt>']:
            feature = 'hcqt'

        logger.info("Run Mode; features={}".format(
            feature if feature else "all"))
        if feature:
            result = run_experiment(feature, config)
        else:
            result = run_all_experiments(config)

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

    return result


if __name__ == "__main__":
    # parser.add_argument("-c", "--master_config", default=CONFIG_PATH)

    # extract_features_parser = subparsers.add_parser('extract_features')
    # extract_features_parser.set_defaults(func=extract_features)

    # fit_and_predict_parser = subparsers.add_parser('fit_and_predict')
    # fit_and_predict_parser.add_argument('experiment_name',
    #                                     help="Name of the experiment. "
    #                                     "Files go in a directory of "
    #                                     "this name.")
    # fit_and_predict_parser.add_argument('test_set',
    #                                     help="Dataset to use for hold-out.")
    # fit_and_predict_parser.set_defaults(func=fit_and_predict)

    # train_parser = subparsers.add_parser('train')
    # train_parser.add_argument('experiment_name',
    #                           help="Name of the experiment. "
    #                                "Files go in a directory of this name.")
    # train_parser.add_argument('test_set',
    #                           help="Dataset to use for hold-out.")
    # train_parser.set_defaults(func=train)
    # predict_parser = subparsers.add_parser('predict')
    # predict_parser.add_argument('experiment_name',
    #                             help="Name of the experiment. "
    #                                  "Files go in a directory of this name.")
    # predict_parser.add_argument('test_set',
    #                             help="Dataset to use for hold-out.")
    # predict_parser.add_argument('-s', '--select_epoch',
    #                             default=None, type=int)
    # predict_parser.set_defaults(func=predict)
    # analyze_parser = subparsers.add_parser('analyze')
    # analyze_parser.add_argument('experiment_name',
    #                             help="Name of the experiment. "
    #                                  "Files go in a directory of this name.")
    # analyze_parser.add_argument('-s', '--select_epoch',
    #                             default=None)
    # analyze_parser.set_defaults(func=analyze)

    # # Tests
    # datatest_parser = subparsers.add_parser('datatest')
    # datatest_parser.add_argument('-p', '--show_full', action="store_true",
    #                              help="Print the full diff to screen.")
    # datatest_parser.set_defaults(func=datatest)
    # datastats_parser = subparsers.add_parser('datastats')
    # datastats_parser.set_defaults(func=datastats)
    # test_parser = subparsers.add_parser('test')
    # test_parser.set_defaults(func=test)

    arguments = docopt(__doc__)
    utils.setup_logging(logging.DEBUG if arguments['--verbose']
                        else logging.INFO)

    handle_arguments(arguments)
