import argparse
import logging
import logging.config
import pandas
import os
import sys

import wcqtlib.config as C
import wcqtlib.data.parse as parse
import wcqtlib.data.extract as E
import wcqtlib.data.cqt
import wcqtlib.train.driver as driver
import wcqtlib.common.utils as utils

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


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


def collect(master_config):
    """Prepare dataframes of notes for experiments."""
    config = C.Config.from_yaml(master_config)

    print(utils.colored("Parsing directories to collect datasets"))
    parse_result = parse.parse_files_to_dataframe(config)

    print(utils.colored("Spliting audio files to notes."))
    extract_result = E.extract_notes(config)

    return all([parse_result,
                extract_result])


def extract_features(master_config):
    """Extract CQTs from all files collected in collect."""
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Extracting CQTs from note audio."))
    success = wcqtlib.data.cqt.cqt_from_df(config, **config["features/cqt"])
    return success


def train(master_config,
          experiment_name):
    """Run training loop.

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.
    """
    print(utils.colored("Training"))
    config = C.Config.from_yaml(master_config)

    model_definition = config["model"]
    hold_out_set = config["experiment/hold_out_set"]
    max_files_per_class = config.get(
        "training/max_files_per_class", None)

    driver.train_model(config,
                       model_selector=model_definition,
                       experiment_name=experiment_name,
                       hold_out_set=hold_out_set,
                       max_files_per_class=max_files_per_class)


def evaluate(master_config,
             experiment_name,
             select_epoch=None):
    """Evaluate datasets and report results.

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

    selected_model_file = "params{}.npz".format(select_epoch) \
        if select_epoch else "final.npz"

    driver.evaluate_and_analyze(
        config, experiment_name, selected_model_file)


def analyze(master_config):
    """Analyze results from an experiment."""
    print(utils.colored("Analyzing"))
    raise NotImplementedError("analyze not yet implemented")


def notebook(master_config):
    """Launch the associated notebook."""
    print(utils.colored("Launching notebook."))
    raise NotImplementedError("notebook not yet implemented")


def datatest(master_config):
    """Check your generated data."""
    config = C.Config.from_yaml(master_config)
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])
    datasets_df = pandas.read_json(datasets_path)
    logger.info("Your datasets_df has {} records".format(len(datasets_df)))

    canonical_df = parse.load_canonical_files()
    logger.info("Cannonical dataset has {} records".format(len(canonical_df)))

    diff_df = parse.diff_datasets_files(canonical_df, datasets_df)
    if len(diff_df):
        print(utils.colored(
            "The following files are missing from your machine", "red"))

        print("Filename\tdirname\tdataset")
        for index, row in diff_df.iterrows():
            print("{:<40}\t{:<20}\t{:<15}".format(
                index, row['dirnamecan'], row['datasetcan']))
        return 1
    else:
        print(utils.colored("You're all set; your dataset matches.", "green"))
        return 0


def test(master_config):
    """Launch all unit tests."""
    print(utils.colored("Running unit tests."))
    raise NotImplementedError("notebook not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    save_canonical_parser = subparsers.add_parser('save_canonical_files')
    save_canonical_parser.set_defaults(func=save_canonical_files)

    collect_parser = subparsers.add_parser('collect')
    collect_parser.set_defaults(func=collect)
    extract_features_parser = subparsers.add_parser('extract_features')
    extract_features_parser.set_defaults(func=extract_features)
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('experiment_name',
                              help="Name of the experiment. "
                                   "Files go in a directory of this name.")
    train_parser.set_defaults(func=train)
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('experiment_name',
                                 help="Name of the experiment. "
                                      "Files go in a directory of this name.")
    evaluate_parser.add_argument('-s', '--select_epoch',
                                 default=None, type=int)
    evaluate_parser.set_defaults(func=evaluate)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.set_defaults(func=analyze)
    notebook_parser = subparsers.add_parser('notebook')
    notebook_parser.set_defaults(func=notebook)

    # Tests
    datatest_parser = subparsers.add_parser('datatest')
    datatest_parser.set_defaults(func=datatest)
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    # TODO MOve this to a file config.
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
                'level': 'INFO',
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
    # logging.basicConfig(level=logging.DEBUG)

    args = vars(parser.parse_args())
    fx = args.pop('func', None)
    if fx:
        success = fx(**args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
