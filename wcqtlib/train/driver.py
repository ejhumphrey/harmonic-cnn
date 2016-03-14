import logging
import numpy as np
import os
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import datetime

import wcqtlib.config as C
import wcqtlib.common.utils as utils
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams
import wcqtlib.train.evaluate as evaluate

logger = logging.getLogger(__name__)


def conditional_colored(value, minval, formatstr="{:0.3f}", color="green"):
    val_str = formatstr.format(value)
    if value < minval:
        val_str = utils.colored(val_str, color)
    return val_str


def construct_training_valid_df(features_df, datasets,
                                validation_split=0.2,
                                max_files_per_class=None):
    """Prepare training and validation dataframes from the features df.
     * First selects from only the datasets given in datasets.
     * Then **for each instrument** (so the distribution from each instrument
        doesn't change)
        * train_test_split to generate training and validation sets.
        * if max_files_per_class, also then restrict it to
            a maximum of that number of files for each train and test

    Parameters
    ----------
    features_df : pandas.DataFrame

    datasets : list of str
        List of datasets to use.

    validation_spilt : float
        Train/validation split.

    max_files_per_class : int
        Number of files to restrict the dataset to.

    """
    search_df = features_df[features_df["dataset"].isin(datasets)]

    selected_instruments_train = []
    selected_instruments_valid = []
    for instrument in search_df["instrument"].unique():
        instrument_df = search_df[search_df["instrument"] == instrument]
        traindf, validdf = train_test_split(
            instrument_df, test_size=validation_split)

        if max_files_per_class:
            traindf = traindf.sample(n=max_files_per_class)
            validdf = traindf.sample(n=max_files_per_class)

        selected_instruments_train.append(traindf)
        selected_instruments_valid.append(validdf)

    return pandas.concat(selected_instruments_train), \
        pandas.concat(selected_instruments_valid)


def get_slicer_from_network_def(network_def_name):
    if 'wcqt' in network_def_name:
        slicer = streams.wcqt_slices
    else:
        slicer = streams.cqt_slices
    return slicer


def train_model(config, model_selector, experiment_name,
                hold_out_set,
                max_files_per_class=None):
    """
    Train a model, writing intermediate params
    to disk.

    Trains for max_iterations or max_time, whichever is fewer.
    [Specified in the config.]

    config: wcqtlib.config.Config
        Instantiated config.

    model_selector : str
        Name of the model to use.
        (This is a function name from models.py)

    experiment_name : str
        Name of the experiment. This is used to
        name the files/parameters saved.

    hold_out_set : str or list of str
        Which dataset to leave out in training.

    max_files_per_class : int or None
        Used for overfitting the network during testing;
        limit the training set to this number of files
        per class.
    """
    logger.info("Starting training for experiment:", experiment_name)
    # Important paths & things to load.
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.read_pickle(features_path)
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    params_dir = os.path.join(model_dir, "params")
    utils.create_directory(model_dir)
    utils.create_directory(params_dir)

    # Get the datasets to use excluding the holdout set.
    exclude_set = set(hold_out_set)
    datasets = set(features_df["dataset"].unique())
    datasets = datasets - exclude_set

    # Set up the dataframe we're going to train with.
    logger.info("[{}] Constructing training df".format(experiment_name))
    training_df, valid_df = construct_training_valid_df(
        features_df, datasets, max_files_per_class=max_files_per_class)
    logger.debug("[{}] training_df : {} rows".format(experiment_name,
                                                     len(training_df)))

    # Save the config we used in the model directory, just in case.
    config.save(os.path.join(model_dir, "config.yaml"))

    # Duration parameters
    max_iterations = config['training/max_iterations']
    max_time = config['training/max_time']  # in seconds

    # Collect various necessary parameters
    t_len = config['training/t_len']
    batch_size = config['training/batch_size']
    n_targets = config['training/n_targets']
    logger.debug("Hyperparams:\nt_len: {}\nbatch_size: {}\n"
                 "n_targets: {}\max_iterations: {}\nmax_time: {}s or {}h"
                 .format(t_len, batch_size, n_targets, max_iterations,
                         max_time, (max_time / 60. / 60.)))

    slicer = get_slicer_from_network_def(model_selector)

    # Set up our streamer
    logger.info("[{}] Setting up streamer".format(experiment_name))
    streamer = streams.InstrumentStreamer(
        training_df, datasets, slicer, t_len=t_len,
        batch_size=batch_size)

    # create our model
    logger.info("[{}] Setting up model: {}".format(experiment_name,
                                                   model_selector))
    network_def = getattr(models, model_selector)(t_len, n_targets)
    model = models.NetworkManager(network_def)

    iter_print_freq = config.get('training/iteration_print_frequency', None)
    iter_write_freq = config.get('training/iteration_write_frequency', None)

    train_t0 = datetime.datetime.now()
    logger.info("[{}] Beginning training loop at {}".format(
        experiment_name, train_t0))
    iter_count = 0
    train_losses = []
    min_train_loss = np.inf
    batch_start, batch_end = [None]*2
    try:
        for batch in streamer:
            batch_start = datetime.datetime.now()
            train_losses += [model.train(batch)]
            train_end = datetime.datetime.now()

            # Time Logging
            logger.debug("[Iter timing] iter: {} | loss: {} | "
                         "stream: {} | train: {}".format(
                            iter_count, train_losses[-1],
                            (batch_start - batch_end) if batch_end else
                            (batch_start - train_t0),
                            (train_end - batch_start)
                            ))
            # Print status
            if iter_print_freq and (iter_count % iter_print_freq == 0):
                print("Iteration: {} | | Train_loss: {}"
                      .format(iter_count,
                              conditional_colored(train_losses[-1],
                                                  min_train_loss)))
                min_train_loss = min(train_losses[-1], min_train_loss)

            # save model, maybe
            if iter_write_freq and (iter_count % iter_write_freq == 0):
                save_path = os.path.join(
                    params_dir, "params{0:0>4}.npz".format(iter_count))
                model.save(save_path)

            iter_count += 1

            # Stopping conditions
            batch_end = datetime.datetime.now()
            if (iter_count >= max_iterations) or \
                    batch_end > (train_t0 + max_time):
                break
    except KeyboardInterrupt:
        print("User cancelled training at epoch:", iter_count)

    # Print final training loss
    print("Last Epoch:", iter_count)
    print("Final training loss:", train_losses[-1])

    # Make sure to save the final model.
    save_path = os.path.join(params_dir, "final.npz".format(iter_count))
    model.save(save_path)
    logger.info("Completed training for experiment:", experiment_name)


def evaluate_and_analyze(config, experiment_name, selected_model_file):
    print("Evaluating experient {} with params from {}".format(
        utils.colored(experiment_name, "magenta"),
        utils.colored(selected_model_file, "cyan")))

    print("Loading DataFrame...")
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.read_pickle(features_path)

    experiment_dir = os.path.join(
        os.path.expanduser(config['paths/model_dir']),
        experiment_name)
    experiment_config_path = os.path.join(experiment_dir, "config.yaml")
    original_config = C.Config.from_yaml(experiment_config_path)
    params_file = os.path.join(experiment_dir, "params", selected_model_file)
    slicer = get_slicer_from_network_def(original_config['model'])

    print("Deserializing Network & Params...")
    model = models.NetworkManager.deserialize_npz(params_file)

    t_len = original_config['training/t_len']
    print("Running evaluation on all files...")
    eval_df = evaluate.evaluate_dataframe(features_df, model, slicer, t_len,
                                          show_progress=True)
    print("Calculating results...")
    results = evaluate.analyze_results(eval_df, experiment_name)
    print("{:*^30}".format(utils.colored("Results", "green")))
    print(results)

    print("File Class Predictions", np.bincount(eval_df["max_likelyhood"]))
    print("File Class Targets", np.bincount(eval_df["target"]))
    print(classification_report(eval_df["max_likelyhood"].tolist(),
                                eval_df["target"].tolist()))

    print("Random baseline should be: {:0.3f}".format(
          1.0 / original_config['training/n_targets']))
