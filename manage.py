import argparse
import logging
import os

import wcqtlib.config as C
import wcqtlib.data.parse as parse
import wcqtlib.data.extract as E
import wcqtlib.common.utils as utils

CONFIG_PATH = os.path.join(os.path.dirname(__name__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


def extract(master_config):
    """Prepare data for experiments."""
    print(utils.colored("Parsing directories to collect datasets"))
    # Load the config.
    config = C.Config.from_yaml(master_config)
    parse_result = parse.parse_files_to_dataframe(config)

    print(utils.colored("Spliting audio files to notes."))
    extract_result = E.extract_notes(config)

    return not all([parse_result,
                    extract_result])


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
    extract_parser = subparsers.add_parser('extract')
    extract_parser.set_defaults(func=extract)
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
        fx(**args)
    else:
        parser.print_help()
OSError
