"""Top-level routines, including:

* extract_features
* train_model
* find_best_model
* predict
* analyze
* fit_and_predict_one
* fit_and_predict_cross_validation
"""

import datetime
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import shutil

import wcqtlib.common.config as C
import wcqtlib.common.utils as utils
import wcqtlib.data.cqt
import wcqtlib.data.dataset
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams
import wcqtlib.evaluate.analyze
import wcqtlib.evaluate.model_selection as MS
import wcqtlib.evaluate.predict

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


def get_slicer_from_network_def(network_def_name):
    if 'wcqt' in network_def_name:
        slicer = streams.wcqt_slices
    else:
        slicer = streams.cqt_slices
    return slicer


class Driver(object):
    "Controller class for handling state."

    def __init__(self, config, experiment_name=None, load_features=True):
        """
        Parameters
        ----------
        config: wcqtlib.config.Config
            Instantiated config.

        experiment_name : str or None
            Name of the experiment. This is used to
            name the files/parameters saved.

            It is required for many but not all functions.
        """
        self.config = config
        self.experiment_name = experiment_name

        # Initialize common paths 'n variables.
        self._init()

        self.load_dataset(load_features=load_features)

    def _init(self):
        self._selected_ds = self.config['data/selected']
        self._dataset_file = self.config['data/{}'.format(self._selected_ds)]
        self._feature_dir = os.path.expanduser(
            self.config['paths/feature_dir'])

        self.model_definition = self.config["model"]
        self.max_files_per_class = self.config.get(
            "training/max_files_per_class", None)

        self.dataset = None

        if self.experiment_name:
            self._model_dir = os.path.join(
                os.path.expanduser(self.config["paths/model_dir"]),
                self.experiment_name)

            self._params_dir = os.path.join(
                self.model_dir, self.config["experiment/params_dir"])
            self._experiment_config_path = os.path.join(
                self.model_dir, self.config['experiment/config_path'])
            self._training_loss_path = os.path.join(
                self.model_dir, self.config['experiment/training_loss'])

    def _init_cross_validation(self, test_set):
        self._training_df_save_path = os.path.join(
            self.model_dir,
            self.config['experiment/data_split_format'].format(
                "train", test_set))
        self._valid_df_save_path = os.path.join(
            self._model_dir,
            self.config['experiment/data_split_format'].format(
                "valid", test_set))

        utils.create_directory(self.model_dir)
        utils.create_directory(self.params_dir)

    def check_cross_validation_state(self):
        """Check to make sure everything's ready to run, including:
         * existing features
        """
        if "cqt" not in self.dataset.to_df().columns:
            logger.error("No features for input data; please extract first.")
            return False

    def load_dataset(self, load_features=True):
        """Load the selected dataset in specified in the config file.

        Parameters
        ----------
        load_features : bool
            If true, tries to load the features version of the dataset,
            else just loads the original specified version.
        """
        if not load_features:
            self.dataset = wcqtlib.data.dataset.Dataset.read_json(
                self._dataset_file, data_root=self.config['paths/data_dir'])
        else:
            dataset_fn = os.path.basename(self.dataset_file)
            feature_ds_path = os.path.join(self._feature_dir, dataset_fn)
            self.dataset = wcqtlib.data.dataset.Dataset.read_json(
                feature_ds_path, data_root=self.config['paths/data_dir'])

    def extract_features(self):
        """Extract CQTs from all files collected in collect."""
        updated_ds = wcqtlib.data.cqt.cqt_from_dataset(
            self.dataset, self._feature_dir, **self.config["features/cqt"])

        success = False
        if updated_ds is not None and \
                len(updated_ds) == len(self.dataset):
            write_path = os.path.join(
                self._feature_dir, os.path.basename(self._dataset_file))
            updated_ds.save_json(write_path)
            success = os.path.exists(write_path)
        return success

    def _get_validation_splits(self, training_df, test_set):
        logger.info("[{}] Constructing training df".format(
            self.experiment_name))
        training_df, valid_df = self.dataset.get_train_val_split(
            test_set, max_files_per_class=self.max_files_per_class)
        logger.debug("[{}] training_df : {} rows".format(
            self.experiment_name, len(training_df)))
        # Save the dfs to disk so we can use them for validation later.
        training_df.to_pickle(self._training_df_save_path)
        valid_df.to_pickle(self._valid_df_save_path)

        return training_df, valid_df

    def train_model(self, test_set):
        """
        Train a model, writing intermediate params
        to disk.

        Trains for max_iterations or max_time, whichever is fewer.
        [Specified in the config.]

        test_set : str
            Which dataset to leave out in training.

        """
        logger.info("Starting training for experiment: {}".format(
            self.experiment_name))

        # Important paths & things to load.
        self._init_cross_validation(test_set)
        if not self.check_cross_validation_state():
            logger.error("train_model setup state invalid.")
            return False

        dataset_df = self.dataset.to_df()
        # Set up the dataframes we're going to train with.
        training_df, valid_df = self._get_validation_splits(
            dataset_df, test_set)

        # Save the config we used in the model directory, just in case.
        self.config.save(self._experiment_config_path)

        # Duration parameters
        max_iterations = self.config['training/max_iterations']
        params_zero_pad = int(np.ceil(np.log10(max_iterations)))
        max_time = self.config['training/max_time']  # in seconds
        param_format_str = self.config['experiment/params_format']
        # insert the zero padding into the format string.
        param_format_str = param_format_str.format(params_zero_pad)

        # Collect various necessary parameters
        t_len = self.config['training/t_len']
        batch_size = self.config['training/batch_size']
        n_targets = self.config['training/n_targets']
        logger.debug("Hyperparams:\nt_len: {}\nbatch_size: {}\n"
                     "n_targets: {}\nmax_iterations: {}\nmax_time: {}s or {}h"
                     .format(t_len, batch_size, n_targets, max_iterations,
                             max_time, (max_time / 60. / 60.)))

        slicer = get_slicer_from_network_def(self.model_definition)

        # Set up our streamer
        logger.info("[{}] Setting up streamer".format(experiment_name))
        streamer = streams.InstrumentStreamer(
            training_df, slicer, t_len=t_len, batch_size=batch_size)

        # create our model
        logger.info("[{}] Setting up model: {}".format(experiment_name,
                                                       model_selector))
        network_def = getattr(models, model_selector)(t_len, n_targets)
        model = models.NetworkManager(network_def)

        iter_print_freq = config.get('training/iteration_print_frequency', None)
        iter_write_freq = config.get('training/iteration_write_frequency', None)

        timers = utils.TimerHolder()
        iter_count = 0
        train_stats = pandas.DataFrame(columns=['timestamp',
                                                'batch_train_dur',
                                                'iteration', 'loss'])
        min_train_loss = np.inf

        timers.start("train")
        logger.info("[{}] Beginning training loop at {}".format(
            experiment_name, timers.get("train")))
        try:
            timers.start(("stream", iter_count))
            for batch in streamer:
                timers.end(("stream", iter_count))
                timers.start(("batch_train", iter_count))
                loss = model.train(batch)
                timers.end(("batch_train", iter_count))
                row = dict(timestamp=timers.get_end(("batch_train", iter_count)),
                           batch_train_dur=timers.get(("batch_train", iter_count)),
                           iteration=iter_count,
                           loss=loss)
                train_stats.loc[len(train_stats)] = row

                # Time Logging
                logger.debug("[Iter timing] iter: {} | loss: {} | "
                             "stream: {} | train: {}".format(
                                iter_count, loss,
                                timers.get(("stream", iter_count)),
                                timers.get(("batch_train", iter_count))
                                ))
                # Print status
                if iter_print_freq and (iter_count % iter_print_freq == 0):
                    mean_train_loss = train_stats["loss"][-iter_print_freq:].mean()
                    logger.info("Iteration: {} | Mean_Train_loss: {}"
                                .format(iter_count,
                                        utils.conditional_colored(
                                            mean_train_loss, min_train_loss)))
                    min_train_loss = min(mean_train_loss, min_train_loss)
                    # Print the mean times for the last n frames
                    logger.info("Mean stream time: {}, Mean train time: {}".format(
                        timers.mean("stream", iter_count - iter_print_freq,
                                    iter_count),
                        timers.mean("batch_train", iter_count - iter_print_freq,
                                    iter_count)))

                # save model, maybe
                if iter_write_freq and (iter_count % iter_write_freq == 0):
                    save_path = os.path.join(
                        params_dir, param_format_str.format(iter_count))
                    logger.debug("Writing params to {}".format(save_path))
                    model.save(save_path)

                if datetime.datetime.now() > \
                        (timers.get("train") + datetime.timedelta(
                            seconds=max_time)):
                    raise EarlyStoppingException("Max Time reached")

                iter_count += 1
                timers.start(("stream", iter_count))
                # Stopping conditions
                if (iter_count >= max_iterations):
                    raise EarlyStoppingException("Max Iterations Reached")

        except KeyboardInterrupt:
            logger.warn(utils.colored("Training Cancelled", "red"))
            print("User cancelled training at epoch:", iter_count)
        except EarlyStoppingException as e:
            logger.warn(utils.colored("Training Stopped for {}".format(e), "red"))
            print("Training halted for: ", e)
        timers.end("train")

        # Print final training loss
        logger.info("Total iterations:".format(iter_count))
        logger.info("Trained for ".format(timers.get("train")))
        logger.info("Final training loss: {}".format(train_stats["loss"].iloc[-1]))

        # Make sure to save the final model.
        save_path = os.path.join(params_dir, "final.npz".format(iter_count))
        model.save(save_path)
        logger.info("Completed training for experiment: {}".format(
            experiment_name))

        # Save training loss
        logger.info("Writing training stats to {}".format(training_loss_path))
        train_stats.to_pickle(training_loss_path)


    def find_best_model(self, config, experiment_name, validation_df,
                        plot_loss=False):
        """Perform model selection on the validation set with a binary search
        for minimum validation loss.

        (Bayesean optimization might be another approach?)

        Parameters
        ----------
        master_config : str
            Full path

        experiment_name : str
            Name of the experiment. Files are saved in a folder of this name.

        validation_df : pandas.DataFrame
            Name of the held out dataset (used to specify the valid file)

        plot_loss : bool
            If true, uses matplotlib to non-blocking plot the loss
            at each validation.

        Returns
        -------
        results_df : pandas.DataFrame
            DataFrame containing the resulting losses.
        """
        logger.info("Finding best model for {}".format(
            utils.colored(experiment_name, "magenta")))
        # load the experiment config
        experiment_dir = os.path.join(
            os.path.expanduser(config['paths/model_dir']),
            experiment_name)
        model_dir = os.path.join(
            os.path.expanduser(config["paths/model_dir"]),
            experiment_name)
        params_dir = os.path.join(experiment_dir,
                                  config['experiment/params_dir'])

        # load all necessary config parameters
        experiment_config_path = os.path.join(experiment_dir,
                                              config['experiment/config_path'])
        original_config = C.Config.from_yaml(experiment_config_path)
        slicer = get_slicer_from_network_def(original_config['model'])
        t_len = original_config['training/t_len']

        # Load the training loss
        training_loss_path = os.path.join(model_dir,
                                          config['experiment/training_loss'])
        training_loss = pandas.read_pickle(training_loss_path)
        validation_error_file = os.path.join(
            model_dir, original_config['experiment/validation_loss'])

        if not os.path.exists(validation_error_file):
            model_files = glob.glob(os.path.join(params_dir, "params*.npz"))
            # model_files.extend(glob.glob(os.path.join(params_dir, "final.npz")))

            result_df, best_model = MS.BinarySearchModelSelector(
                model_files, validation_df, slicer, t_len, show_progress=True)()

            result_df.to_pickle(validation_error_file)
            best_path = os.path.join(params_dir,
                                     original_config['experiment/best_params'])
            shutil.copyfile(best_model['model_file'], best_path)
        else:
            logger.info("Model Search already done; printing previous results")
            result_df = pandas.read_pickle(validation_error_file)
            #make sure model_iteration is an int so sorting makes sense.
            result_df["model_iteration"].apply(int)
            logger.info("\n{}".format(result_df.sort_values("model_iteration")))

        if plot_loss:
            fig = plt.figure()
            ax = plt.plot(training_loss["iteration"], training_loss["loss"])
            ax_val = plt.plot(result_df["model_iteration"], result_df["mean_loss"])
            plt.draw()
            plt.show()

        return result_df

    def select_best_iteration(self, model_selection_df):
        """Given the model selection df, return the iteration
        which produced the best model.

        Returns
        -------
        best_model : int
            The iteration number which produced the best model.
        """
        best = model_selection_df.loc[model_selection_df["mean_acc"].argmax()]

        return best["model_iteration"], os.path.basename(best["model_file"])

    def predict(self, config, experiment_name, selected_model_file,
                features_df_override=None):
        """Generates a prediction for *all* files, and writes them to disk
        as a dataframe.

        If features_df_override, replace the features_df with this
        dataframe (for testing)
        """
        logger.info("Evaluating experient {} with params from iter {}".format(
            utils.colored(experiment_name, "magenta"),
            utils.colored(selected_model_file, "cyan")))
        model_iter = utils.iter_from_params_filepath(selected_model_file)

        logger.info("Loading DataFrame...")
        if features_df_override is None:
            features_path = os.path.join(
                os.path.expanduser(config["paths/extract_dir"]),
                config["dataframes/features"])
            features_df = pandas.read_pickle(features_path)
        else:
            features_df = features_df_override

        experiment_dir = os.path.join(
            os.path.expanduser(config['paths/model_dir']),
            experiment_name)
        experiment_config_path = os.path.join(experiment_dir,
                                              config['experiment/config_path'])
        original_config = C.Config.from_yaml(experiment_config_path)
        params_file = os.path.join(experiment_dir,
                                   config['experiment/params_dir'],
                                   selected_model_file)
        slicer = get_slicer_from_network_def(original_config['model'])

        logger.info("Deserializing Network & Params...")
        model = models.NetworkManager.deserialize_npz(params_file)

        t_len = original_config['training/t_len']
        logger.info("Running evaluation on all files...")
        predictions_df = wcqtlib.evaluate.predict.predict_many(
            features_df, model, slicer, t_len, show_progress=True)
        predictions_df_path = os.path.join(
            experiment_dir,
            original_config.get('experiment/predictions_format',
                                config.get('experiment/predictions_format',
                                           None))
            .format(model_iter))
        predictions_df.to_pickle(predictions_df_path)
        return predictions_df

    def analyze(config, experiment_name, model_name, test_set):
        logger.info("Evaluating experient {} with params from {}".format(
            utils.colored(experiment_name, "magenta"),
            utils.colored(model_name, "cyan")))

        experiment_dir = os.path.join(
            os.path.expanduser(config['paths/model_dir']),
            experiment_name)
        experiment_config_path = os.path.join(experiment_dir,
                                              config['experiment/config_path'])
        original_config = C.Config.from_yaml(experiment_config_path)

        analyzer = wcqtlib.analyze.PredictionAnalyzer.from_config(
            original_config, experiment_name, model_name, test_set)
        analysis_path = os.path.join(
            experiment_dir,
            original_config.get('experiment/analysis_format',
                                config.get('experiment/analysis_format', None))
            .format(model_name))
        logger.info("Saving analysis to:", analysis_path)
        analyzer.save(analysis_path)
        return analyzer

    def fit_and_predict_one(self, dataset, model, test_set):
        """On a particular model, with a given set
        * train
        * model_selection
        * predict
        * analyze
        * Write all outputs to a file
        """
        pass

    def fit_and_predict_cross_validation(self, datsaet, model):
        """Master loop for running cross validation across
        all datasets.
        """
        pass
