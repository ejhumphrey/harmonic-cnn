import argparse
import logging
import os

import wcqtlib.config as C
import wcqtlib.data.parse as parse

CONFIG_PATH = os.path.join(os.path.dirname(__name__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


def extract(master_config):
    """Prepare data for experiments."""
    # Load the config.
    config = C.Config.from_yaml(master_config)
    return parse.parse_files_to_dataframe(config)


def train(master_config):
    """Run training loop."""
    pass


def evaluate(master_config):
    """Evaluate datasets and report results."""
    pass


def analyze(master_config):
    """Analyze results from an experiment."""
    pass


def notebook(master_config):
    """Launch the associated notebook."""
    pass


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
