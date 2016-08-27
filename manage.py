import argparse
import logging
import numpy as np
import os
import pandas
import shutil
import sys

import hcnn.common.config as C
import hcnn.common.utils as utils
import hcnn.data.dataset
import hcnn.data.cqt
import hcnn.driver

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)

import theano
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


def fit_and_predict(master_config, experiment_name, test_set):
    """Runs:
    - train
    - model_selection_df
    - predict
    - analyze
    """
    run_name = "fit_and_predict:{}:{}".format(experiment_name, test_set)

    config = C.Config.load(master_config)
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


def train(master_config,
          experiment_name,
          test_set):
    """Run training loop.

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    test_set : str
        String in ["rwc", "uiowa", "philharmonia"] specifying which
        dataset to use as the test set.
    """
    print(utils.colored("Training experiment: {}".format(experiment_name)))
    logger.info("Training with test set {}".format(test_set))
    config = C.Config.load(master_config)
    driver = hcnn.driver.Driver(config, experiment_name,
                                load_features=True)

    driver.setup_data_splits(test_set)
    return driver.train_model()


def predict(master_config,
            experiment_name,
            test_set,
            select_epoch=None):
    """Predict results on all datasets and report results.

    Parameters
    ----------
    master_config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Evaluating"))
    config = C.Config.load(master_config)

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


def datatest(master_config, show_full=False):
    # TODO
    pass


def datastats(master_config):
    config = C.Config.load(master_config)
    print(utils.colored("Printing Stats."))

    driver = hcnn.driver.Driver(config, load_features=False)
    driver.print_stats()
    return True


def test(master_config):
    """Runs integration test.
    This is equivalent to running
    python manage.py -c data/integrationtest_config.yaml run
    """
    # Load integrationtest config
    results = []

    INT_CONFIG_PATHS = [
        ("./data/integrationtest_config_cqt.yaml", "integration_test_cqt"),
        ("./data/integrationtest_config_wcqt.yaml", "integration_test_wcqt"),
        ("./data/integrationtest_config_hcqt.yaml", "integration_test_hcqt")
    ]

    print(utils.colored("Extracting features from tinydata set."))
    results.append(extract_features(INT_CONFIG_PATHS[0][0]))

    if results[-1]:
        for config, experiment_name in INT_CONFIG_PATHS:
            print(utils.colored(
                "Running regression test on tinydata set : {}."
                .format(config)))
            results.append(
                run(config, experiment_name=experiment_name))

    result = all(results)
    print("IntegrationTest {}".format(utils.result_colored(result)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    extract_features_parser = subparsers.add_parser('extract_features')
    extract_features_parser.set_defaults(func=extract_features)

    fit_and_predict_parser = subparsers.add_parser('fit_and_predict')
    fit_and_predict_parser.add_argument('experiment_name',
                                        help="Name of the experiment. "
                                        "Files go in a directory of "
                                        "this name.")
    fit_and_predict_parser.add_argument('test_set',
                                        help="Dataset to use for hold-out.")
    fit_and_predict_parser.set_defaults(func=fit_and_predict)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('experiment_name',
                              help="Name of the experiment. "
                                   "Files go in a directory of this name.")
    train_parser.add_argument('test_set',
                              help="Dataset to use for hold-out.")
    train_parser.set_defaults(func=train)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('experiment_name',
                                help="Name of the experiment. "
                                     "Files go in a directory of this name.")
    predict_parser.add_argument('test_set',
                                help="Dataset to use for hold-out.")
    predict_parser.add_argument('-s', '--select_epoch',
                                default=None, type=int)
    predict_parser.set_defaults(func=predict)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('experiment_name',
                                help="Name of the experiment. "
                                     "Files go in a directory of this name.")
    analyze_parser.add_argument('-s', '--select_epoch',
                                default=None)
    analyze_parser.set_defaults(func=analyze)

    # Tests
    datatest_parser = subparsers.add_parser('datatest')
    datatest_parser.add_argument('-p', '--show_full', action="store_true",
                                 help="Print the full diff to screen.")
    datatest_parser.set_defaults(func=datatest)
    datastats_parser = subparsers.add_parser('datastats')
    datastats_parser.set_defaults(func=datastats)
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    utils.setup_logging(logging.INFO)

    args = vars(parser.parse_args())
    fx = args.pop('func', None)
    if fx:
        success = fx(**args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
