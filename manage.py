import argparse
import logging
import numpy as np
import os
import pandas
import shutil
import sys

import wcqtlib.common.config as C
import wcqtlib.common.utils as utils
import wcqtlib.data.dataset
import wcqtlib.data.parse as parse
import wcqtlib.data.extract as E
import wcqtlib.data.cqt
import wcqtlib.driver

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


def run_process_if_not_exists(process, filepath, **kwargs):
    if not os.path.exists(filepath):
        return process(**kwargs)
    else:
        return True


def save_canonical_files(master_config):
    """Create the canonical_files.json file."""
    config = C.Config.from_yaml(master_config)
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])
    datasets_df = pandas.read_json(datasets_path)
    logger.info("Generating canonical datasets file from {} records".format(
        len(datasets_df)))
    success = parse.generate_canonical_files(datasets_df)
    logger.info("Success: {}".format(success))
    return success


def clean(master_config):
    """Clean dataframes and extracted audio/features."""
    config = C.Config.from_yaml(master_config)

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
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Extracting CQTs from note audio."))

    driver = wcqtlib.driver.Driver(config, load_features=False)
    result = driver.extract_features()
    print("Extraction {}".format(utils.result_colored(result)))
    return result


def run(master_config, experiment_name):
    """Run an experiment end-to-end with cross validation.
    Note: requires extracted features.
    """
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Running experiment end-to-end."))

    timer = utils.TimerHolder()
    timer.start("run")
    logger.debug("Running with experiment_name={} at {}"
                 .format(experiment_name, timer.get_start("run")))
    driver = wcqtlib.driver.Driver(config, experiment_name,
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

    config = C.Config.from_yaml(master_config)
    print(utils.colored("Running {} end-to-end.".format(run_name)))

    timer = utils.TimerHolder()
    timer.start(run_name)
    logger.debug("Running with experiment_name={} at {}"
                 .format(experiment_name, timer.get_start("run")))
    driver = wcqtlib.driver.Driver(config, experiment_name,
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
    config = C.Config.from_yaml(master_config)

    if dataset:
        driver.train_model(config,
                           dataset,
                           model_selector=model_definition,
                           experiment_name=experiment_name,
                           test_set=test_set,
                           max_files_per_class=max_files_per_class)
        return True
    else:
        logger.error("Dataset load failed.")
        return False


def model_selection(master_config,
                    experiment_name,
                    test_set,
                    plot_loss=False):
    """Perform model selection on the validation set.

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    test_set : str
        String in ["rwc", "uiowa", "philharmonia"] specifying which
        dataset to use as the test set.

    plot_loss : bool
        If true, uses matplotlib to non-blocking plot the loss
        at each validation.
    """
    print(utils.colored("Model Selection"))
    config = C.Config.from_yaml(master_config)

    return driver.find_best_model(config,
                                  experiment_name=experiment_name,
                                  validation_df=valid_df,
                                  plot_loss=plot_loss)


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
    config = C.Config.from_yaml(master_config)

    max_iterations = config['training/max_iterations']
    params_zero_pad = int(np.ceil(np.log10(max_iterations)))
    param_format_str = config['experiment/params_format']
    param_format_str = param_format_str.format(params_zero_pad)
    selected_model_file = param_format_str.format(select_epoch) \
        if str(select_epoch).isdigit() else "{}.npz".format(
            select_epoch)

    results = driver.predict(
        config, experiment_name, selected_model_file)
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
    config = C.Config.from_yaml(master_config)

    hold_out_set = config["experiment/hold_out_set"]

    driver.analyze(config, experiment_name, select_epoch, hold_out_set)
    return 0


def datatest(master_config, show_full=False):
    """Check your generated data."""
    config = C.Config.from_yaml(master_config)
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])

    # Regenerate the datasets_df to make sure it's not stale.
    if not collect(master_config, clean_data=False):
        logger.error("Failed to regenerate datasets_df.")

    datasets_df = pandas.read_json(datasets_path)
    logger.info("Your datasets_df has {} records".format(len(datasets_df)))

    canonical_df = parse.load_canonical_files()
    logger.info("Cannonical dataset has {} records".format(len(canonical_df)))

    diff_df = parse.diff_datasets_files(canonical_df, datasets_df)
    if len(diff_df) and show_full:
        print(utils.colored(
            "The following files are missing from your machine", "red"))

        print("Filename\tdirname\tdataset")
        for index, row in diff_df.iterrows():
            print("{:<40}\t{:<20}\t{:<15}".format(
                index, row['dirnamecan'], row['datasetcan']))
    else:
        print(utils.colored("You're all set; your dataset matches.", "green"))

    print(utils.colored("Now checking all files for validity."))

    classmap = wcqtlib.data.parse.InstrumentClassMap()
    filtered_df = datasets_df[datasets_df["instrument"].isin(
        classmap.allnames)]
    bad_files = E.check_valid_audio_files(filtered_df,
                                          write_path="bad_file_reads.txt")
    print(utils.colored("{} files could not be opened".format(
                        len(bad_files)), "red"))
    if bad_files:
        print("Unable to open the following audio files:")
        for filepath in bad_files:
            print(filepath)


def datastats(master_config):
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Printing Stats."))

    driver = wcqtlib.driver.Driver(config, load_features=False)
    driver.print_stats()
    return True


def test(master_config):
    """Runs integration test.
    This is equivalent to running
    python manage.py -c data/integrationtest_config.yaml run
    """
    # Load integrationtest config
    features_result, run_result = False, False

    CONFIG_PATH = "./data/integrationtest_config.yaml"
    print(utils.colored("Extracting features from tinydata set."))
    features_result = extract_features(CONFIG_PATH)

    if features_result:
        print(utils.colored("Running regression test on tinydata set."))
        run_result = run(CONFIG_PATH, experiment_name="integration_test")

    result = all([features_result, run_result])
    print("IntegrationTest {}".format(utils.result_colored(result)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    save_canonical_parser = subparsers.add_parser('save_canonical_files')
    save_canonical_parser.set_defaults(func=save_canonical_files)

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
