import lasagne
import logging
import os
import pandas

import wcqtlib.config as C
import wcqtlib.common.utils as utils
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams

logger = logging.getLogger(__name__)


def construct_training_df(features_df, datasets, max_files_per_class):
    if max_files_per_class:
        search_df = features_df[features_df["dataset"].isin(datasets)]
        selected_instruments = []
        for instrument in search_df["instrument"].unique():
            selected_instruments.append(
                search_df[search_df["instrument"] == instrument].sample(
                    n=max_files_per_class))
        return pandas.concat(selected_instruments)
    else:
        return features_df


def train_model(config, model_selector, experiment_name,
                hold_out_set,
                max_files_per_class=None):
    """
    Train a model, writing intermediate params
    to disk.

    Trains for max_epochs epochs, where an epoch is:
     # 44 is the approximate number of average frames in one file.
     total_dataset_frames = n_training_files * (44 / t_len)
     epoch_size = total_dataset_frames / batch_size

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
    # Important paths & things to load.
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.load_pickle(features_path)
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    utils.create_directory(model_dir)

    # Get the datasets to use excluding the holdout set.
    exclude_set = set(hold_out_set)
    datasets = set(features_df["dataset"].unique())
    datasets = datasets - exclude_set

    # Set up the dataframe we're going to train with.
    training_df = construct_training_df(features_df,
                                        datasets,
                                        max_files_per_class)

    # Save the config we used in the model directory, just in case.
    config.save(os.path.join(model_dir, "config.yaml"))

    # Collect various necessary parameters
    t_len = config['training/t_len']
    batch_size = config['training/batch_size']
    n_targets = config['training/n_targets']
    max_epochs = config['training/max_epochs']
    epoch_length = int(len(training_df) * (44 / float(t_len)) /
                       float(batch_size))

    # Set up our streamer
    streamer = streams.InstrumentStreamer(
        training_df, datasets, streams.cqt_slices,
        t_len=t_len,
        batch_size=batch_size)

    # create our model
    network_def = getattr(models, model_selector)(t_len, n_targets)
    model = models.NetworkManager(network_def)

    epoch_count = 0
    try:
        while epoch_count < max_epochs:
            logger.debug("Beginning epoch: ", epoch_count)
            # train, storing loss for each batchself.
            batch_count = 0
            for batch in streamer:
                logger.debug("Beginning ")
                train_loss = model.train(batch)
                # print, maybe

                batch_count += 1
                if batch_count >= epoch_length:
                    break
            # print valid, maybe
            # save model, maybe
            epoch_count += 1
    except KeyboardInterrupt:
        print("User cancelled training at epoch:", epoch_count)

    # Print final training & validation loss & acc
    # Make sure to save the final model.
