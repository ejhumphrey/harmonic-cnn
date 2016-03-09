import argparse
import logging
import os
import sys

import wcqtlib.config as C
import wcqtlib.data.parse as parse
import wcqtlib.data.extract as E
import wcqtlib.data.cqt
import wcqtlib.common.utils as utils

CONFIG_PATH = os.path.join(os.path.dirname(__name__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


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


def train(master_config):
    """Run training loop."""
    print(utils.colored("Training"))
    raise NotImplementedError("train not yet implemented")


def evaluate(master_config):
    """Evaluate datasets and report results."""
    print(utils.colored("Evaluating"))
    raise NotImplementedError("evaluate not yet implemented")


def analyze(master_config):
    """Analyze results from an experiment."""
    print(utils.colored("Analyzing"))
    raise NotImplementedError("analyze not yet implemented")


def notebook(master_config):
    """Launch the associated notebook."""
    print(utils.colored("Launching notebook."))
    raise NotImplementedError("notebook not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    collect_parser = subparsers.add_parser('collect')
    collect_parser.set_defaults(func=collect)
    extract_features_parser = subparsers.add_parser('extract_features')
    extract_features_parser.set_defaults(func=extract_features)
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.set_defaults(func=evaluate)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.set_defaults(func=analyze)
    notebook_parser = subparsers.add_parser('notebook')
    notebook_parser.set_defaults(func=notebook)

    logging.basicConfig(level=logging.DEBUG)

    args = vars(parser.parse_args())
    fx = args.pop('func', None)
    if fx:
        success = fx(**args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

